using System.Numerics.Tensors;
using SentenceTransformers.EmbeddingGemma2.LiteRt;
using SentenceTransformers.EmbeddingGemma2.Numerics;

namespace SentenceTransformers.EmbeddingGemma2.Model;

/// <summary>One transformer block of the EmbeddingGemma 2 text encoder (Gemma 4 architecture).</summary>
internal sealed class TextLayer
{
    public int Index;
    public int NumHeads;
    public int NumKvHeads;
    public int HeadDim;
    public bool IsGlobal;
    public float[] InputNorm;
    public QuantizedMatrix Q, K, V, O;
    public float[] QNorm, KNorm, VNorm;
    public RopeTable Rope;
    public float[] PostAttentionNorm;
    public float[] PreFeedForwardNorm;
    public QuantizedMatrix Gate, Up, Down;
    public float[] PostFeedForwardNorm;
    public QuantizedMatrix PerLayerGate, PerLayerProjection;
    public float[] PerLayerPostNorm;
    public float LayerScalar;
}

/// <summary>
/// The text tower of EmbeddingGemma 2, ported op-for-op from the LiteRT <c>tf_lite_text_encoder</c> graph:
/// <code>
/// ple   = rms_norm(FC(x) reshaped to [L, 24, 512])                (per-layer inputs)
/// per layer:
///   h   = rms_norm(x);  q,k,v = FC(h);  q,k,v = rms_norm per head;  RoPE(q, k)
///   x  += rms_norm(FC_o(softmax(q·kᵀ + mask)·v))                  (GQA, no 1/√d scaling)
///   x  += rms_norm(FC_down(gelu(FC_gate(rms_norm(x))) · FC_up(...)))
///   x  += rms_norm(FC_b(gelu(FC_a(x)) · ple[layer]))
///   x  *= layer_scalar
/// out   = FC_proj(mean_pool(rms_norm(x)))                          (512 -> 768)
/// </code>
/// Attention is bidirectional over the real tokens (the graph masks only padding). Local layers use
/// 256-dim heads with θ = 10⁴ RoPE, every 6th (global) layer 512-dim heads with a single KV head and
/// θ = 10⁶. All projections are int4 weights driven through the dynamically quantized int8 GEMM.
/// </summary>
internal sealed class TextEncoder
{
    public const string EmbedderModelType = "tf_lite_embedder";
    public const string EncoderModelType = "tf_lite_text_encoder";

    public int HiddenSize { get; }
    public int OutputSize { get; }
    public int NumLayers => Layers.Length;
    public int PerLayerSize { get; }
    public TextLayer[] Layers { get; }
    public TextEmbedder Embedder { get; }

    private readonly QuantizedMatrix _perLayerProjection;
    private readonly float[] _perLayerNorm;
    private readonly float[] _finalNorm;
    private readonly QuantizedMatrix _outputProjection;
    private readonly float _poolEpsilon;

    private TextEncoder(TextEmbedder embedder, TextLayer[] layers, QuantizedMatrix perLayerProjection, float[] perLayerNorm, float[] finalNorm, QuantizedMatrix outputProjection, float poolEpsilon)
    {
        Embedder = embedder;
        Layers = layers;
        _perLayerProjection = perLayerProjection;
        _perLayerNorm = perLayerNorm;
        _finalNorm = finalNorm;
        _outputProjection = outputProjection;
        _poolEpsilon = poolEpsilon;
        HiddenSize = perLayerProjection.Cols;
        PerLayerSize = perLayerNorm.Length;
        OutputSize = outputProjection.Rows;
    }

    public long WeightBytes =>
        Embedder.ByteSize + _perLayerProjection.ByteSize + _outputProjection.ByteSize +
        Layers.Sum(l => l.Q.ByteSize + l.K.ByteSize + l.V.ByteSize + l.O.ByteSize + l.Gate.ByteSize + l.Up.ByteSize + l.Down.ByteSize + l.PerLayerGate.ByteSize + l.PerLayerProjection.ByteSize);

    public static TextEncoder Load(LiteRtLmFile file)
    {
        var embedder = TextEmbedder.Load(file.ReadModel(EmbedderModelType));
        var model = file.ReadModel(EncoderModelType);
        // Every signature shares the same weights; walk the smallest one.
        var sig = model.Signatures.OrderBy(s => model.Subgraph(s.SubgraphIndex).Tensors[s.Inputs["embeddings"]].Shape[1]).First();
        return FromGraph(embedder, model, model.Subgraph(sig.SubgraphIndex));
    }

    private static TextEncoder FromGraph(TextEmbedder embedder, TfLiteModel model, TfLiteSubgraph sg)
    {
        var fcs = new List<QuantizedMatrix>();
        var norms = new List<float[]>();
        var scalars = new List<float>();
        var ropes = new List<float[]>();
        float poolEps = 0;
        foreach (var op in sg.Operators)
        {
            if (op.Opcode == TfLiteOp.FullyConnected)
            {
                fcs.Add(QuantizedMatrix.FromTensor(model, sg.Tensors[op.Inputs[1]]));
            }
            else if (op.IsComposite("odml.rms_norm"))
            {
                Vision.VisionEncoder.RequireDefaultEpsilon(model, op);
                norms.Add(model.ReadFloats(sg.Tensors[op.Inputs[1]]));
            }
            else if (op.IsComposite("odml.rope"))
            {
                ropes.Add(ReadRopeInverseFrequencies(model, op));
            }
            else if (op.Opcode == TfLiteOp.Mul && op.Inputs.Length == 2)
            {
                var t = sg.Tensors[op.Inputs[1]];
                if (t.Shape.Length == 3 && t.ElementCount == 1 && model.IsConstant(t))
                {
                    scalars.Add(model.ReadScalar(t));
                }
            }
            else if (op.Opcode == TfLiteOp.Add && op.Inputs.Length == 2)
            {
                var t = sg.Tensors[op.Inputs[1]];
                if (t.Shape.Length == 2 && t.ElementCount == 1 && model.IsConstant(t))
                {
                    poolEps = model.ReadScalar(t);   // mean-pool denominator epsilon
                }
            }
        }

        // 1 per-layer projection + 9 FCs per layer + 1 output projection.
        int numLayers = (fcs.Count - 2) / 9;
        if (fcs.Count != numLayers * 9 + 2 || norms.Count != numLayers * 8 + 2 || scalars.Count != numLayers || ropes.Count != numLayers * 2)
        {
            throw new InvalidDataException($"Unexpected text encoder graph structure: {fcs.Count} FC, {norms.Count} norms, {scalars.Count} scalars, {ropes.Count} RoPE ops.");
        }

        int fi = 1, ni = 1;
        var layers = new TextLayer[numLayers];
        for (int l = 0; l < numLayers; l++)
        {
            var layer = new TextLayer { Index = l };
            layer.InputNorm = norms[ni++];
            layer.Q = fcs[fi++];
            layer.K = fcs[fi++];
            layer.V = fcs[fi++];
            layer.QNorm = norms[ni++];
            layer.KNorm = norms[ni++];
            layer.VNorm = norms[ni++];
            layer.O = fcs[fi++];
            layer.PostAttentionNorm = norms[ni++];
            layer.PreFeedForwardNorm = norms[ni++];
            layer.Gate = fcs[fi++];
            layer.Up = fcs[fi++];
            layer.Down = fcs[fi++];
            layer.PostFeedForwardNorm = norms[ni++];
            layer.PerLayerGate = fcs[fi++];
            layer.PerLayerProjection = fcs[fi++];
            layer.PerLayerPostNorm = norms[ni++];
            layer.LayerScalar = scalars[l];

            layer.HeadDim = layer.QNorm.Length;
            layer.NumHeads = layer.Q.Rows / layer.HeadDim;
            layer.NumKvHeads = layer.K.Rows / layer.HeadDim;
            var inv = ropes[2 * l];
            if (inv.Length * 2 != layer.HeadDim)
            {
                throw new InvalidDataException($"Layer {l}: RoPE covers {inv.Length * 2} of {layer.HeadDim} dims (partial rotary is not supported).");
            }
            layer.Rope = new RopeTable(inv);
            layer.IsGlobal = layer.NumKvHeads == 1 && layer.HeadDim > layers.FirstOrDefault()?.HeadDim;
            if (layer.NumHeads % layer.NumKvHeads != 0 || layer.V.Rows != layer.K.Rows || layer.O.Cols != layer.Q.Rows)
            {
                throw new InvalidDataException($"Layer {l}: inconsistent attention shapes.");
            }
            layers[l] = layer;
        }
        // Layers that share a θ share one table.
        for (int l = 1; l < numLayers; l++)
        {
            for (int p = 0; p < l; p++)
            {
                if (layers[p].Rope.Half == layers[l].Rope.Half && ropes[2 * p].AsSpan().SequenceEqual(ropes[2 * l]))
                {
                    layers[l].Rope = layers[p].Rope;
                    break;
                }
            }
        }

        return new TextEncoder(embedder, layers, fcs[0], norms[0], norms[ni], fcs[fi], poolEps);
    }

    /// <summary>The <c>odml.rope</c> decomposition multiplies positions by a constant inverse-frequency
    /// vector of shape [1, 1, head_dim/2]; read it so the angles match the reference bit-for-bit.</summary>
    internal static float[] ReadRopeInverseFrequencies(TfLiteModel model, TfLiteOperator op)
    {
        var dg = model.Subgraph(op.DecompositionSubgraph);
        foreach (var dop in dg.Operators)
        {
            if (dop.Opcode != TfLiteOp.Mul)
            {
                continue;
            }
            foreach (var input in dop.Inputs)
            {
                var t = dg.Tensors[input];
                if (t.Type == TfLiteType.Float32 && t.Shape.Length == 3 && t.Shape[0] == 1 && t.Shape[1] == 1 && model.IsConstant(t))
                {
                    return model.ReadFloats(t);
                }
            }
        }
        throw new InvalidDataException("Could not find the RoPE inverse frequencies in the decomposition subgraph.");
    }

    /// <summary>Scratch buffers for one forward pass (sized to the stacked token count).</summary>
    private sealed class Scratch
    {
        public readonly QuantizedActivations Qa = new();
        public float[] H, Q, K, V, Attn, Tmp, Ffn1, Ffn2, Ple;

        /// <summary><paramref name="a"/> if it holds at least <paramref name="n"/> floats, else a new uninitialized
        /// array: every scratch buffer is fully written (by explicit row counts) before it is read.</summary>
        public static float[] Grow(float[] a, int n) => a is not null && a.Length >= n ? a : GC.AllocateUninitializedArray<float>(n);
    }

    /// <summary>Scratch buffers of finished forward passes, reused by the next ones (one per concurrent caller):
    /// a 1000-token document needs ~70 MB of activations, and fresh arrays cost zeroing and page faults.</summary>
    private readonly System.Collections.Concurrent.ConcurrentBag<Scratch> _scratch = new();

    /// <summary>
    /// Runs the encoder over a batch of token sequences (each already wrapped in BOS/EOS) and returns the
    /// pooled, projected embedding of each (not normalized). <paramref name="inputEmbeddings"/> optionally
    /// overrides the token embeddings (used for multimodal soft tokens); rows are stacked in batch order.
    /// </summary>
    public float[][] Forward(IReadOnlyList<int[]> sequences, ParallelOptions po, float[] inputEmbeddings = null, Action<int, float[]> layerHook = null)
    {
        int count = sequences.Count;
        var offsets = new int[count + 1];
        for (int s = 0; s < count; s++)
        {
            offsets[s + 1] = offsets[s] + sequences[s].Length;
        }
        int total = offsets[count];
        int d = HiddenSize;

        var x = inputEmbeddings ?? new float[total * d];
        if (inputEmbeddings is null)
        {
            for (int s = 0; s < count; s++)
            {
                for (int t = 0; t < sequences[s].Length; t++)
                {
                    Embedder.Lookup(sequences[s][t], x.AsSpan((offsets[s] + t) * d, d));
                }
            }
        }
        else if (inputEmbeddings.Length != total * d)
        {
            throw new ArgumentException("Input embedding size does not match the token count.");
        }
        return ForwardEmbeddings(x, offsets, po, layerHook);
    }

    /// <summary>Encoder forward from precomputed input embeddings <c>x[total, d]</c> (modified in place).</summary>
    public float[][] ForwardEmbeddings(float[] x, int[] offsets, ParallelOptions po, Action<int, float[]> layerHook = null)
    {
        int count = offsets.Length - 1;
        int total = offsets[^1];
        int d = HiddenSize;
        int maxLen = 0;
        for (int s = 0; s < count; s++)
        {
            maxLen = Math.Max(maxLen, offsets[s + 1] - offsets[s]);
        }

        var sc = _scratch.TryTake(out var cached) ? cached : new Scratch();
        try
        {
            return ForwardEmbeddings(x, offsets, count, total, d, maxLen, sc, po, layerHook);
        }
        finally
        {
            _scratch.Add(sc);
        }
    }

    private float[][] ForwardEmbeddings(float[] x, int[] offsets, int count, int total, int d, int maxLen, Scratch sc, ParallelOptions po, Action<int, float[]> layerHook)
    {
        int maxQ = Layers.Max(l => l.Q.Rows), maxKv = Layers.Max(l => l.K.Rows), maxFf = Layers.Max(l => l.Gate.Rows);
        sc.H = Scratch.Grow(sc.H, total * d);
        sc.Q = Scratch.Grow(sc.Q, total * maxQ);
        sc.K = Scratch.Grow(sc.K, total * maxKv);
        sc.V = Scratch.Grow(sc.V, total * maxKv);
        sc.Attn = Scratch.Grow(sc.Attn, total * maxQ);
        sc.Tmp = Scratch.Grow(sc.Tmp, total * d);
        sc.Ffn1 = Scratch.Grow(sc.Ffn1, total * maxFf);
        sc.Ffn2 = Scratch.Grow(sc.Ffn2, total * maxFf);

        // Per-layer inputs: FC(x) -> [total, layers * P] then RMS-normalized per P-sized slice.
        int pl = _perLayerProjection.Rows;
        sc.Ple = Scratch.Grow(sc.Ple, total * pl);
        Fc(sc, x, total, _perLayerProjection, sc.Ple, po);
        Ops.RmsNorm(sc.Ple, _perLayerNorm, sc.Ple, total * pl / PerLayerSize, PerLayerSize, po);
        layerHook?.Invoke(-1, sc.Ple);

        foreach (var layer in Layers)
        {
            layer.Rope.Ensure(maxLen);
            RunLayer(layer, x, offsets, total, sc, po);
            layerHook?.Invoke(layer.Index, x);
        }

        // Final norm, masked mean pool (the graph divides by count + |count| + eps), projection.
        Ops.RmsNorm(x, _finalNorm, x, total, d, po);
        layerHook?.Invoke(NumLayers, x);
        var pooled = new float[count * d];
        for (int s = 0; s < count; s++)
        {
            int n = offsets[s + 1] - offsets[s];
            var acc = pooled.AsSpan(s * d, d);
            for (int t = offsets[s]; t < offsets[s + 1]; t++)
            {
                TensorPrimitives.Add(acc, x.AsSpan(t * d, d), acc);
            }
            float denom = (float)n + MathF.Abs(n) + _poolEpsilon;
            TensorPrimitives.Divide(acc, denom, acc);
        }
        var output = new float[count * OutputSize];
        Fc(sc, pooled, count, _outputProjection, output, po);
        var result = new float[count][];
        for (int s = 0; s < count; s++)
        {
            result[s] = output.AsSpan(s * OutputSize, OutputSize).ToArray();
        }
        return result;
    }

    private static int SequenceOf(int[] offsets, int token)
    {
        int idx = Array.BinarySearch(offsets, token);
        if (idx < 0)
        {
            return ~idx - 1;
        }
        // Several offsets can be equal when a sequence is empty; pick the last sequence starting here.
        while (idx + 1 < offsets.Length - 1 && offsets[idx + 1] == token)
        {
            idx++;
        }
        return idx;
    }

    /// <summary>Test hook: runs a single layer on <paramref name="x"/> (one sequence, modified in place)
    /// given precomputed per-layer inputs - used for teacher-forced, layer-by-layer verification.</summary>
    internal void RunSingleLayer(int layerIndex, float[] x, float[] perLayerInputs, ParallelOptions po)
    {
        int total = x.Length / HiddenSize;
        var offsets = new[] { 0, total };
        var sc = new Scratch();
        int maxQ = Layers.Max(l => l.Q.Rows), maxKv = Layers.Max(l => l.K.Rows), maxFf = Layers.Max(l => l.Gate.Rows);
        sc.H = new float[total * HiddenSize];
        sc.Q = new float[total * maxQ];
        sc.K = new float[total * maxKv];
        sc.V = new float[total * maxKv];
        sc.Attn = new float[total * maxQ];
        sc.Tmp = new float[total * HiddenSize];
        sc.Ffn1 = new float[total * maxFf];
        sc.Ffn2 = new float[total * maxFf];
        sc.Ple = perLayerInputs;
        Layers[layerIndex].Rope.Ensure(total);
        RunLayer(Layers[layerIndex], x, offsets, total, sc, po);
    }

    private static void Fc(Scratch sc, float[] input, int rows, QuantizedMatrix w, Span<float> output, ParallelOptions po)
    {
        sc.Qa.Quantize(input, rows, w.Cols, w.Cols, po);
        QGemm.Multiply(sc.Qa, w, output, w.Rows, po);
    }

    /// <summary>Debug hook for op-level verification: (layer, step name, buffer) after each sub-step.</summary>
    internal Action<int, string, float[]> StepHook;

    private void RunLayer(TextLayer layer, float[] x, int[] offsets, int total, Scratch sc, ParallelOptions po)
    {
        var hook = StepHook;
        void Step(string name, float[] buffer) => hook?.Invoke(layer.Index, name, buffer);
        int d = HiddenSize;
        int hd = layer.HeadDim;
        int qDim = layer.Q.Rows, kvDim = layer.K.Rows;

        // --- attention block ---
        Ops.RmsNorm(x, layer.InputNorm, sc.H, total, d, po);
        Step("in_norm", sc.H);
        sc.Qa.Quantize(sc.H, total, d, d, po);   // q, k and v share the same quantized input
        QGemm.Multiply(sc.Qa, layer.Q, sc.Q, qDim, po);
        QGemm.Multiply(sc.Qa, layer.K, sc.K, kvDim, po);
        QGemm.Multiply(sc.Qa, layer.V, sc.V, kvDim, po);
        Step("q", sc.Q); Step("k", sc.K); Step("v", sc.V);
        Ops.RmsNorm(sc.Q, layer.QNorm, sc.Q, total * qDim / hd, hd, po);
        Ops.RmsNorm(sc.K, layer.KNorm, sc.K, total * kvDim / hd, hd, po);
        Ops.RmsNorm(sc.V, layer.VNorm, sc.V, total * kvDim / hd, hd, po);
        Step("qn", sc.Q); Step("kn", sc.K); Step("vn", sc.V);
        ParallelRows.For(total, qDim + kvDim, po, (t0, t1) =>
        {
            for (int t = t0; t < t1; t++)
            {
                int pos = t - offsets[SequenceOf(offsets, t)];
                var cos = layer.Rope.Cos(pos);
                var sin = layer.Rope.Sin(pos);
                for (int h = 0; h < layer.NumHeads; h++)
                {
                    Ops.Rope(sc.Q.AsSpan(t * qDim + h * hd, hd), cos, sin);
                }
                for (int h = 0; h < layer.NumKvHeads; h++)
                {
                    Ops.Rope(sc.K.AsSpan(t * kvDim + h * hd, hd), cos, sin);
                }
            }
        });
        Step("q_rope", sc.Q); Step("k_rope", sc.K);
        Attention.Run(sc.Q, sc.K, sc.V, sc.Attn, offsets, layer.NumHeads, layer.NumKvHeads, hd, scale: 1f, po, maskedKeys: true);
        Step("attn", sc.Attn);
        Fc(sc, sc.Attn, total, layer.O, sc.Tmp, po);
        Step("o", sc.Tmp);
        Ops.AddRmsNorm(x, sc.Tmp, layer.PostAttentionNorm, total, d, po);
        Step("x_attn", x);

        // --- GeGLU feed-forward ---
        int ff = layer.Gate.Rows;
        Ops.RmsNorm(x, layer.PreFeedForwardNorm, sc.H, total, d, po);
        Step("ffn_norm", sc.H);
        sc.Qa.Quantize(sc.H, total, d, d, po);
        QGemm.Multiply(sc.Qa, layer.Gate, sc.Ffn1, ff, po);
        if (hook is null)
        {
            // gelu(gate) · up computed in the up projection's epilogue (same arithmetic, one pass less).
            QGemm.MultiplyGeluGated(sc.Qa, layer.Up, sc.Ffn1, ff, po);
        }
        else
        {
            QGemm.Multiply(sc.Qa, layer.Up, sc.Ffn2, ff, po);
            Step("gate", sc.Ffn1); Step("up", sc.Ffn2);
            Ops.GeluMul(sc.Ffn1, sc.Ffn2, total * ff, po);
        }
        Step("geglu", sc.Ffn1);
        Fc(sc, sc.Ffn1, total, layer.Down, sc.Tmp, po);
        Step("down", sc.Tmp);
        Ops.AddRmsNorm(x, sc.Tmp, layer.PostFeedForwardNorm, total, d, po);
        Step("x_ffn", x);

        // --- per-layer input gate ---
        int p = layer.PerLayerGate.Rows;
        Fc(sc, x, total, layer.PerLayerGate, sc.Ffn1, po);
        Step("ple_gate", sc.Ffn1);
        int stride = _perLayerProjection.Rows;
        ParallelRows.For(total, p, po, (t0, t1) =>
        {
            for (int t = t0; t < t1; t++)
            {
                var row = sc.Ffn1.AsSpan(t * p, p);
                Ops.GeluTanh(row);
                TensorPrimitives.Multiply(row, sc.Ple.AsSpan(t * stride + layer.Index * PerLayerSize, PerLayerSize), row);
            }
        });
        Step("ple_mul", sc.Ffn1);
        Fc(sc, sc.Ffn1, total, layer.PerLayerProjection, sc.Tmp, po);
        Step("ple_proj", sc.Tmp);
        if (hook is null)
        {
            // x = (x + norm(ple)) · layer scalar in one sweep.
            Ops.AddRmsNorm(x, sc.Tmp, layer.PerLayerPostNorm, total, d, po, scale: layer.LayerScalar);
        }
        else
        {
            Ops.AddRmsNorm(x, sc.Tmp, layer.PerLayerPostNorm, total, d, po);
            Step("x_ple", x);
            TensorPrimitives.Multiply(x.AsSpan(0, total * d), layer.LayerScalar, x.AsSpan(0, total * d));
        }
    }
}

/// <summary>Token embedding table: int4 rows with a per-row scale, times the constant √d multiplier.</summary>
internal sealed class TextEmbedder
{
    private readonly byte[] _packed;     // int4, two values per byte, low nibble first
    private readonly float[] _rowScale;
    private readonly float _multiplier;

    public int VocabSize { get; }
    public int Dim { get; }
    public long ByteSize => _packed.LongLength + _rowScale.LongLength * 4;

    private TextEmbedder(byte[] packed, float[] rowScale, float multiplier, int vocab, int dim)
    {
        _packed = packed;
        _rowScale = rowScale;
        _multiplier = multiplier;
        VocabSize = vocab;
        Dim = dim;
    }

    public static TextEmbedder Load(TfLiteModel model)
    {
        var sg = model.Subgraph(model.Signatures[0].SubgraphIndex);
        TfLiteTensor table = null;
        float multiplier = 1f;
        foreach (var op in sg.Operators)
        {
            if (op.Opcode == TfLiteOp.EmbeddingLookup)
            {
                table = sg.Tensors[op.Inputs[1]];
            }
            else if (op.Opcode == TfLiteOp.Mul && table is not null)
            {
                multiplier = model.ReadScalar(sg.Tensors[op.Inputs[1]]);
            }
        }
        if (table is null || table.Type != TfLiteType.Int4 || table.Shape.Length != 2 || table.Shape[1] % 2 != 0)
        {
            throw new InvalidDataException("Unexpected embedder graph (expected an int4 EMBEDDING_LOOKUP table).");
        }
        var q = table.Quantization;
        return new TextEmbedder(model.RawData(table).ToArray(), q.Scale, multiplier, table.Shape[0], table.Shape[1]);
    }

    /// <summary>Writes <c>dequant(table[id]) · multiplier</c> into <paramref name="dst"/>. Out-of-range ids
    /// map to row 0, as the graph's range check does.</summary>
    public void Lookup(int id, Span<float> dst)
    {
        if ((uint)id >= (uint)VocabSize)
        {
            id = 0;
        }
        var row = _packed.AsSpan(id * (Dim / 2), Dim / 2);
        float s = _rowScale[id];
        for (int i = 0; i < row.Length; i++)
        {
            byte b = row[i];
            dst[2 * i] = (float)((sbyte)(b << 4) >> 4) * s * _multiplier;
            dst[2 * i + 1] = (float)((sbyte)b >> 4) * s * _multiplier;
        }
    }
}
