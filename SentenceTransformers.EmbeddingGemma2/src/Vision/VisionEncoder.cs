using System.Numerics.Tensors;
using SentenceTransformers.EmbeddingGemma2.LiteRt;
using SentenceTransformers.EmbeddingGemma2.Numerics;

namespace SentenceTransformers.EmbeddingGemma2.Vision;

/// <summary>A fully quantized int8 <c>FULLY_CONNECTED</c> of the vision tower: static input scale,
/// per-channel int8 weights and a static output scale (requantized, then dequantized).</summary>
internal sealed class StaticQuantizedLinear
{
    public QuantizedMatrix Weights { get; }
    public float InputScale { get; }
    public int InputZeroPoint { get; }
    public QGemm.Requantization Requant { get; }

    public StaticQuantizedLinear(QuantizedMatrix weights, TfLiteQuantization input, TfLiteQuantization output)
    {
        Weights = weights;
        InputScale = input.Scale[0];
        InputZeroPoint = (int)input.ZeroPoint[0];
        if (output.ZeroPoint.Length > 0 && output.ZeroPoint[0] != 0)
        {
            throw new NotSupportedException("Asymmetric output quantization is not supported.");
        }
        Requant = new QGemm.Requantization(InputScale, weights.Scale, output.Scale[0]);
    }
}

/// <summary>One block of the EmbeddingGemma 2 vision transformer.</summary>
internal sealed class VisionLayer
{
    public float[] InputNorm, QNorm, KNorm, VNorm, PostAttentionNorm, PreFeedForwardNorm, PostFeedForwardNorm;
    public StaticQuantizedLinear Q, K, V, O, Gate, Up, Down;
}

/// <summary>Output of <see cref="VisionEncoder.Encode"/>: one 512-d embedding per valid soft token.</summary>
internal sealed record VisionSoftTokens(int Count, float[] Embeddings, int Rows, int Cols);

/// <summary>
/// The EmbeddingGemma 2 vision tower (<c>tf_lite_vision_encoder</c> + <c>tf_lite_vision_adapter</c> +
/// <c>tf_lite_end_of_vision</c>), ported op-for-op:
/// <code>
/// x   = patches·Wᵀ + b + pos_x[px] + pos_y[py]                 (16×16×3 patches, learned 2-D positions)
/// per layer (×16, static int8 activations):
///   h = rms_norm(x); q,k,v = int8 FC(h); q,k,v = rms_norm per head (64)
///   2-D RoPE (θ = 100): dims [0,32) rotate by the x position, [32,64) by the y position
///   x += rms_norm(int8 FC_o(softmax(q·kᵀ)·v))                (12 heads, no mask, no 1/√d)
///   x += rms_norm(int8 FC_down(gelu(FC_gate(rms_norm(x))) · FC_up(...)))
/// soft = √768 · mean over each 3×3 patch neighbourhood         (pooling, + validity mask)
/// adapter: FC(rms_norm(rms_norm(soft))) -> 512                  (text embedding space)
/// </code>
/// Every int8 FC quantizes its input with a calibrated scale, accumulates in int32, requantizes the
/// output to int8 and dequantizes it, exactly as the LiteRT graph does. Padding patches (position -1)
/// are fed through the encoder like the reference does (they take part in attention).
/// </summary>
internal sealed class VisionEncoder
{
    public const string EncoderModelType = "tf_lite_vision_encoder";
    public const string AdapterModelType = "tf_lite_vision_adapter";
    public const string EndOfVisionModelType = "tf_lite_end_of_vision";

    public int Hidden { get; }
    public int NumHeads { get; }
    public int HeadDim { get; }
    public int PatchSize { get; }
    public int PoolingKernel { get; }
    public int OutputSize => _adapterT.Length / Hidden;
    public VisionLayer[] Layers { get; }
    /// <summary>Supported soft-token budgets (signatures), e.g. 70 and 140.</summary>
    public int[] TokenBudgets { get; }
    public float[] EndOfVision { get; }

    private readonly float[] _patchT;        // [patchDim, hidden] (transposed FC weight)
    private readonly float[] _patchBias;
    private readonly float[] _posX;          // [positions, hidden]
    private readonly float[] _posY;
    private readonly int _positions;
    private readonly float[] _invFreq;       // RoPE inverse frequencies (per 32-dim half: 16)
    private readonly float _poolScale;       // 1 / kernel²
    private readonly float _outputScale;     // √hidden
    private readonly float[] _adapterNorm1, _adapterNorm2;
    private readonly float[] _adapterT;      // [hidden, out]

    private VisionEncoder(int hidden, int heads, int patchSize, int pool, VisionLayer[] layers, int[] budgets, float[] eoi,
                          float[] patchT, float[] patchBias, float[] posX, float[] posY, int positions, float[] invFreq,
                          float poolScale, float outputScale, float[] adapterNorm1, float[] adapterNorm2, float[] adapterT)
    {
        Hidden = hidden;
        NumHeads = heads;
        HeadDim = hidden / heads;
        PatchSize = patchSize;
        PoolingKernel = pool;
        Layers = layers;
        TokenBudgets = budgets;
        EndOfVision = eoi;
        _patchT = patchT;
        _patchBias = patchBias;
        _posX = posX;
        _posY = posY;
        _positions = positions;
        _invFreq = invFreq;
        _poolScale = poolScale;
        _outputScale = outputScale;
        _adapterNorm1 = adapterNorm1;
        _adapterNorm2 = adapterNorm2;
        _adapterT = adapterT;
    }

    public long WeightBytes => (_patchT.LongLength + _posX.LongLength + _posY.LongLength + _adapterT.LongLength) * 4 +
                               Layers.Sum(l => new[] { l.Q, l.K, l.V, l.O, l.Gate, l.Up, l.Down }.Sum(f => f.Weights.ByteSize));

    public static VisionEncoder Load(LiteRtLmFile file)
    {
        var model = file.ReadModel(EncoderModelType);
        var budgets = model.Signatures.Where(s => s.Key.StartsWith("vision_", StringComparison.Ordinal))
                                      .Select(s => model.Subgraph(s.SubgraphIndex).Tensors[s.Outputs["features"]].Shape[1])
                                      .OrderBy(t => t).ToArray();
        // All signatures share the weights; walk the first one.
        var sg = model.Subgraph(model.Signatures[0].SubgraphIndex);
        var floatFcs = new List<(TfLiteTensor W, TfLiteTensor B)>();
        var intFcs = new List<StaticQuantizedLinear>();
        var norms = new List<float[]>();
        float[] invFreq = null;
        float poolScale = 0, outputScale = 0;
        foreach (var op in sg.Operators)
        {
            switch (op.Opcode)
            {
                case TfLiteOp.FullyConnected:
                {
                    var w = sg.Tensors[op.Inputs[1]];
                    if (w.Type == TfLiteType.Float32)
                    {
                        floatFcs.Add((w, op.Inputs.Length > 2 && op.Inputs[2] >= 0 ? sg.Tensors[op.Inputs[2]] : null));
                    }
                    else
                    {
                        intFcs.Add(new StaticQuantizedLinear(QuantizedMatrix.FromTensor(model, w), sg.Tensors[op.Inputs[0]].Quantization, sg.Tensors[op.Outputs[0]].Quantization));
                    }
                    break;
                }
                case TfLiteOp.StableHloComposite when op.CompositeName == "odml.rms_norm":
                    norms.Add(model.ReadFloats(sg.Tensors[op.Inputs[1]]));
                    break;
                case TfLiteOp.Mul:
                {
                    var t = sg.Tensors[op.Inputs[1]];
                    if (!model.IsConstant(t) || t.Type != TfLiteType.Float32)
                    {
                        break;
                    }
                    if (invFreq is null && t.Shape.Length == 3 && t.Shape[0] == 1 && t.Shape[1] == 1)
                    {
                        invFreq = model.ReadFloats(t);
                    }
                    else if (t.ElementCount == 1)
                    {
                        // Pooling: one-hot · (1/k²), then · √hidden; the latter comes last.
                        float v = model.ReadScalar(t);
                        if (poolScale == 0)
                        {
                            poolScale = v;
                        }
                        else
                        {
                            outputScale = v;
                        }
                    }
                    break;
                }
            }
        }
        if (floatFcs.Count != 3 || intFcs.Count % 7 != 0 || norms.Count != intFcs.Count || invFreq is null || poolScale == 0 || outputScale == 0)
        {
            throw new InvalidDataException($"Unexpected vision encoder graph structure ({floatFcs.Count} float FC, {intFcs.Count} int8 FC, {norms.Count} norms).");
        }

        var (patchW, patchB) = floatFcs[0];
        int hidden = patchW.Shape[0];
        int patchDim = patchW.Shape[1];
        var patchT = Transpose(model.ReadFloats(patchW), hidden, patchDim);
        var patchBias = model.ReadFloats(patchB);
        // Position tables: FC weights [hidden, positions] applied to one-hot positions -> keep as [positions, hidden] rows.
        int positions = floatFcs[1].W.Shape[1];
        var posX = Transpose(model.ReadFloats(floatFcs[1].W), hidden, positions);
        var posY = Transpose(model.ReadFloats(floatFcs[2].W), hidden, positions);

        int numLayers = intFcs.Count / 7;
        var layers = new VisionLayer[numLayers];
        for (int l = 0; l < numLayers; l++)
        {
            int f = 7 * l, n = 7 * l;
            layers[l] = new VisionLayer
            {
                InputNorm = norms[n], QNorm = norms[n + 1], KNorm = norms[n + 2], VNorm = norms[n + 3],
                PostAttentionNorm = norms[n + 4], PreFeedForwardNorm = norms[n + 5], PostFeedForwardNorm = norms[n + 6],
                Q = intFcs[f], K = intFcs[f + 1], V = intFcs[f + 2], O = intFcs[f + 3], Gate = intFcs[f + 4], Up = intFcs[f + 5], Down = intFcs[f + 6],
            };
        }
        int headDim = layers[0].QNorm.Length;
        int heads = layers[0].Q.Weights.Rows / headDim;
        if (invFreq.Length * 4 != headDim)
        {
            throw new InvalidDataException($"Unexpected 2-D RoPE table size {invFreq.Length} for head dim {headDim}.");
        }
        int patchSize = (int)Math.Round(Math.Sqrt(patchDim / 3.0));
        int pool = (int)Math.Round(Math.Sqrt(1.0 / poolScale));

        // Adapter: rms_norm, rms_norm, FC (float) -> text hidden size.
        var adapter = file.ReadModel(AdapterModelType);
        var asg = adapter.Subgraph(adapter.Signatures[0].SubgraphIndex);
        var adapterNorms = asg.Operators.Where(o => o.IsComposite("odml.rms_norm")).Select(o => adapter.ReadFloats(asg.Tensors[o.Inputs[1]])).ToArray();
        var adapterFc = asg.Operators.Single(o => o.Opcode == TfLiteOp.FullyConnected);
        var adapterW = asg.Tensors[adapterFc.Inputs[1]];
        if (adapterNorms.Length != 2 || adapterW.Type != TfLiteType.Float32)
        {
            throw new InvalidDataException("Unexpected vision adapter graph structure.");
        }
        var adapterT = Transpose(adapter.ReadFloats(adapterW), adapterW.Shape[0], adapterW.Shape[1]);

        var eoiModel = file.ReadModel(EndOfVisionModelType);
        var esg = eoiModel.Subgraph(eoiModel.Signatures[0].SubgraphIndex);
        var eoi = eoiModel.ReadFloats(esg.Tensors[eoiModel.Signatures[0].Outputs.Values.First()]);

        return new VisionEncoder(hidden, heads, patchSize, pool, layers, budgets, eoi, patchT, patchBias, posX, posY, positions, invFreq,
                                 poolScale, outputScale, adapterNorms[0], adapterNorms[1], adapterT);
    }

    private static float[] Transpose(float[] m, int rows, int cols)
    {
        var t = new float[m.Length];
        for (int r = 0; r < rows; r++)
        {
            for (int c = 0; c < cols; c++)
            {
                t[c * rows + r] = m[r * cols + c];
            }
        }
        return t;
    }

    /// <summary>Runs the encoder on a padded patch sequence and returns the valid soft tokens projected by
    /// the adapter into the text embedding space.</summary>
    /// <param name="patches">[numPatches, 3·p·p] pixel values in [0, 1] (zero for padding patches).</param>
    /// <param name="positions">[numPatches, 2] (x, y) patch coordinates, -1 for padding.</param>
    public VisionSoftTokens Encode(float[] patches, int[] positions, int numPatches, ParallelOptions po, Action<int, float[]> layerHook = null)
    {
        int d = Hidden;
        int n = numPatches;
        var x = new float[n * d];
        using (Profiler.Measure("vision.embed"))
        {
            SGemm.Multiply(patches, _patchT.Length / d, _patchT, d, x, d, n, d, _patchT.Length / d, 1f, po);
            Span<float> pos = stackalloc float[d];
            for (int i = 0; i < n; i++)
            {
                var row = x.AsSpan(i * d, d);
                TensorPrimitives.Add(row, _patchBias, row);
                int px = positions[2 * i], py = positions[2 * i + 1];
                if (px >= 0 && py >= 0 && px < _positions && py < _positions)
                {
                    // patch + (pos_x + pos_y), the graph's summation order.
                    TensorPrimitives.Add(_posX.AsSpan(px * d, d), _posY.AsSpan(py * d, d), pos);
                    TensorPrimitives.Add(row, pos, row);
                }
            }
        }
        layerHook?.Invoke(-1, x);

        var sc = new Scratch(n, d, Layers.Max(l => l.Gate.Weights.Rows));
        var rope = BuildRope(positions, n);
        for (int l = 0; l < Layers.Length; l++)
        {
            RunLayer(Layers[l], x, n, sc, rope, po);
            layerHook?.Invoke(l, x);
        }
        return Pool(x, positions, n, po);
    }

    private sealed class Scratch
    {
        public readonly QuantizedActivations Qa = new();
        public readonly float[] H, Q, K, V, Attn, Tmp, Ff1, Ff2;

        public Scratch(int n, int d, int ff)
        {
            H = new float[n * d];
            Q = new float[n * d];
            K = new float[n * d];
            V = new float[n * d];
            Attn = new float[n * d];
            Tmp = new float[n * d];
            Ff1 = new float[n * ff];
            Ff2 = new float[n * ff];
        }
    }

    /// <summary>cos/sin per patch for the x and y halves: [n, 2 (x|y), 16].</summary>
    private (float[] Cos, float[] Sin) BuildRope(int[] positions, int n)
    {
        int half = _invFreq.Length;
        var cos = new float[n * 2 * half];
        var sin = new float[n * 2 * half];
        for (int i = 0; i < n; i++)
        {
            for (int axis = 0; axis < 2; axis++)
            {
                // Padding patches carry position -1 and are rotated by it, like the graph.
                float p = positions[2 * i + axis];
                for (int j = 0; j < half; j++)
                {
                    float ang = p * _invFreq[j];
                    cos[(i * 2 + axis) * half + j] = MathF.Cos(ang);
                    sin[(i * 2 + axis) * half + j] = MathF.Sin(ang);
                }
            }
        }
        return (cos, sin);
    }

    private void ApplyRope(float[] t, int n, (float[] Cos, float[] Sin) rope, ParallelOptions po)
    {
        int half = _invFreq.Length;
        int hd = HeadDim;
        ParallelRows.For(n, Hidden, po, (r0, r1) =>
        {
            for (int i = r0; i < r1; i++)
            {
                for (int h = 0; h < NumHeads; h++)
                {
                    var head = t.AsSpan(i * Hidden + h * hd, hd);
                    Ops.Rope(head.Slice(0, 2 * half), rope.Cos.AsSpan((i * 2) * half, half), rope.Sin.AsSpan((i * 2) * half, half));
                    Ops.Rope(head.Slice(2 * half, 2 * half), rope.Cos.AsSpan((i * 2 + 1) * half, half), rope.Sin.AsSpan((i * 2 + 1) * half, half));
                }
            }
        });
    }

    private static void Fc(Scratch sc, float[] input, int rows, StaticQuantizedLinear fc, float[] output, ParallelOptions po)
    {
        sc.Qa.QuantizeStatic(input, rows, fc.Weights.Cols, fc.Weights.Cols, fc.InputScale, fc.InputZeroPoint, po);
        QGemm.Multiply(sc.Qa, fc.Weights, output, fc.Weights.Rows, po, fc.Requant);
    }

    private void RunLayer(VisionLayer layer, float[] x, int n, Scratch sc, (float[] Cos, float[] Sin) rope, ParallelOptions po)
    {
        int d = Hidden, hd = HeadDim;
        Ops.RmsNorm(x, layer.InputNorm, sc.H, n, d, po);
        // q, k and v share the same quantized input (same static scale).
        sc.Qa.QuantizeStatic(sc.H, n, d, d, layer.Q.InputScale, layer.Q.InputZeroPoint, po);
        QGemm.Multiply(sc.Qa, layer.Q.Weights, sc.Q, d, po, layer.Q.Requant);
        QGemm.Multiply(sc.Qa, layer.K.Weights, sc.K, d, po, layer.K.Requant);
        QGemm.Multiply(sc.Qa, layer.V.Weights, sc.V, d, po, layer.V.Requant);
        Ops.RmsNorm(sc.Q, layer.QNorm, sc.Q, n * NumHeads, hd, po);
        Ops.RmsNorm(sc.K, layer.KNorm, sc.K, n * NumHeads, hd, po);
        Ops.RmsNorm(sc.V, layer.VNorm, sc.V, n * NumHeads, hd, po);
        ApplyRope(sc.Q, n, rope, po);
        ApplyRope(sc.K, n, rope, po);
        Attention.Run(sc.Q, sc.K, sc.V, sc.Attn, new[] { 0, n }, NumHeads, NumHeads, hd, scale: 1f, po);
        Fc(sc, sc.Attn, n, layer.O, sc.Tmp, po);
        Ops.AddRmsNorm(x, sc.Tmp, layer.PostAttentionNorm, n, d, po);

        int ff = layer.Gate.Weights.Rows;
        Ops.RmsNorm(x, layer.PreFeedForwardNorm, sc.H, n, d, po);
        sc.Qa.QuantizeStatic(sc.H, n, d, d, layer.Gate.InputScale, layer.Gate.InputZeroPoint, po);
        QGemm.Multiply(sc.Qa, layer.Gate.Weights, sc.Ff1, ff, po, layer.Gate.Requant);
        QGemm.Multiply(sc.Qa, layer.Up.Weights, sc.Ff2, ff, po, layer.Up.Requant);
        Ops.GeluMul(sc.Ff1, sc.Ff2, n * ff, po);
        Fc(sc, sc.Ff1, n, layer.Down, sc.Tmp, po);
        Ops.AddRmsNorm(x, sc.Tmp, layer.PostFeedForwardNorm, n, d, po);
    }

    /// <summary>k×k average pooling into the soft-token grid, √d scaling, validity mask, then the adapter.</summary>
    private VisionSoftTokens Pool(float[] x, int[] positions, int n, ParallelOptions po)
    {
        int d = Hidden, k = PoolingKernel;
        int maxTokens = n / (k * k);
        int maxX = -1;
        for (int i = 0; i < n; i++)
        {
            maxX = Math.Max(maxX, positions[2 * i]);
        }
        int gridW = FloorDiv(maxX + 1, k);
        var pooled = new float[maxTokens * d];
        var valid = new bool[maxTokens];
        Span<float> tmp = stackalloc float[d];
        for (int i = 0; i < n; i++)
        {
            int t = FloorDiv(positions[2 * i], k) + gridW * FloorDiv(positions[2 * i + 1], k);
            if ((uint)t >= (uint)maxTokens)
            {
                continue;
            }
            valid[t] = true;
            TensorPrimitives.Multiply(x.AsSpan(i * d, d), _poolScale, tmp);
            TensorPrimitives.Add(pooled.AsSpan(t * d, d), tmp, pooled.AsSpan(t * d, d));
        }
        int count = 0;
        while (count < maxTokens && valid[count])
        {
            count++;
        }
        var soft = pooled.AsSpan(0, count * d).ToArray();
        TensorPrimitives.Multiply(soft, _outputScale, soft);

        // Adapter (rows are independent, so only the valid ones are computed).
        Ops.RmsNorm(soft, _adapterNorm1, soft, count, d, po);
        Ops.RmsNorm(soft, _adapterNorm2, soft, count, d, po);
        int outDim = OutputSize;
        var emb = new float[count * outDim];
        SGemm.Multiply(soft, d, _adapterT, outDim, emb, outDim, count, outDim, d, 1f, po);
        return new VisionSoftTokens(count, emb, count, outDim);
    }

    private static int FloorDiv(int a, int b) => (int)Math.Floor((double)a / b);

    /// <summary>Test hook: runs one layer on <paramref name="x"/> in place (teacher-forced verification).</summary>
    internal void RunSingleLayer(int layer, float[] x, int[] positions, int n, ParallelOptions po)
        => RunLayer(Layers[layer], x, n, new Scratch(n, Hidden, Layers.Max(l => l.Gate.Weights.Rows)), BuildRope(positions, n), po);

    /// <summary>Test hook: pooling + adapter on final hidden states.</summary>
    internal VisionSoftTokens PoolForTest(float[] x, int[] positions, int n, ParallelOptions po) => Pool((float[])x.Clone(), positions, n, po);
}
