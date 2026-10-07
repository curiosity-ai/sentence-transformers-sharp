using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using SentenceTransformers.EmbeddingGemma2.Numerics;

namespace SentenceTransformers.EmbeddingGemma2.LiteRt;

/// <summary>A runtime tensor of the <see cref="GraphExecutor"/>. Quantized (int8) tensors are held in
/// "fake-quant" form: float values that are exact multiples of the tensor scale, so data-movement ops
/// (reshape, slice, concat, pad, transpose) need no special casing and <c>DEQUANTIZE</c> is free.</summary>
internal sealed class GraphTensor
{
    public int[] Shape;
    public float[] F;
    public int[] I;
    public bool[] B;

    public int Length
    {
        get
        {
            int n = 1;
            foreach (var d in Shape)
            {
                n *= d;
            }
            return n;
        }
    }

    /// <summary>The arena of the <see cref="GraphExecutor.Run"/> executing on this thread (null elsewhere).</summary>
    [ThreadStatic]
    internal static TensorArena CurrentArena;

    public static GraphTensor Float(int[] shape, float[] data = null) => new() { Shape = shape, F = data ?? NewFloats(Count(shape)) };

    private static float[] NewFloats(int n) => CurrentArena?.Rent(n) ?? new float[n];
    public static GraphTensor Int(int[] shape, int[] data = null) => new() { Shape = shape, I = data ?? new int[Count(shape)] };
    public static GraphTensor Bool(int[] shape, bool[] data = null) => new() { Shape = shape, B = data ?? new bool[Count(shape)] };

    public static int Count(int[] shape)
    {
        int n = 1;
        foreach (var d in shape)
        {
            n *= d;
        }
        return n;
    }

    public GraphTensor WithShape(int[] shape) => new() { Shape = shape, F = F, I = I, B = B };

    /// <summary>A copy that shares no storage with this tensor (graph outputs may alias each other through
    /// reshapes, so carried-over state must not be passed back by reference).</summary>
    public GraphTensor Clone() => new() { Shape = (int[])Shape.Clone(), F = (float[])F?.Clone(), I = (int[])I?.Clone(), B = (bool[])B?.Clone() };
}

/// <summary>
/// Recycles the float buffers of a <see cref="GraphExecutor"/> run: every activation is rented from exact-size free
/// lists and returned as soon as the last op reading it has run (reference counted per array, because reshapes
/// alias their input's storage). A streaming encoder therefore reuses the same few cache-warm buffers for every
/// chunk instead of allocating hundreds of megabytes of short-lived arrays.
/// </summary>
internal sealed class TensorArena
{
    private readonly Dictionary<int, Stack<float[]>> _free = new();
    private readonly Dictionary<float[], int> _refs = new(ReferenceEqualityComparer.Instance);

    /// <summary>A zeroed buffer of exactly <paramref name="length"/> floats.</summary>
    public float[] Rent(int length)
    {
        float[] a;
        if (_free.TryGetValue(length, out var stack) && stack.Count > 0)
        {
            a = stack.Pop();
            Array.Clear(a);
        }
        else
        {
            a = new float[length];
        }
        _refs[a] = 0;
        return a;
    }

    /// <summary>Records one more tensor slot holding <paramref name="a"/> (no-op for arrays not rented here).</summary>
    public void AddRef(float[] a)
    {
        if (a is not null && _refs.TryGetValue(a, out int c))
        {
            _refs[a] = c + 1;
        }
    }

    /// <summary>Drops one slot's reference; the buffer is recycled when none remain.</summary>
    public void Release(float[] a)
    {
        if (a is null || !_refs.TryGetValue(a, out int c))
        {
            return;
        }
        if (c > 1)
        {
            _refs[a] = c - 1;
            return;
        }
        _refs.Remove(a);
        Recycle(a);
    }

    /// <summary>End of a run: buffers still referenced (op temporaries, graph outputs already copied out) are
    /// recycled for the next run.</summary>
    public void EndRun()
    {
        foreach (var a in _refs.Keys)
        {
            Recycle(a);
        }
        _refs.Clear();
    }

    private void Recycle(float[] a)
    {
        if (!_free.TryGetValue(a.Length, out var stack))
        {
            _free[a.Length] = stack = new Stack<float[]>();
        }
        stack.Push(a);
    }
}

/// <summary>
/// A small, self-contained executor for the TFLite graphs in a LiteRT-LM bundle - used for the
/// streaming audio encoder, whose 1,600-op graph threads per-layer streaming state (cached keys/values,
/// convolution padding, lookback features) through dozens of slices and concatenations. Executing the
/// graph op-for-op mirrors those semantics exactly; the heavy operators run on the same SIMD kernels as
/// the hand-written text and vision towers:
/// <list type="bullet">
/// <item>int8-activation <c>FULLY_CONNECTED</c> with int2/int4/int8 weights -> <see cref="QGemm"/> (static
/// input scale, int32 accumulation, int8 output requantization);</item>
/// <item>float <c>FULLY_CONNECTED</c> / <c>BATCH_MATMUL</c> -> <see cref="SGemm"/>;</item>
/// <item><c>odml.rms_norm</c> composites -> fused RMSNorm;</item>
/// <item>elementwise ops -> <see cref="TensorPrimitives"/> with N-d broadcasting.</item>
/// </list>
/// Every op is "compiled" once into a closure with its constants pre-decoded, so a run only allocates
/// activations.
/// </summary>
internal sealed class GraphExecutor
{
    private readonly TfLiteModel _model;
    private readonly TfLiteSubgraph _sg;
    private readonly GraphTensor[] _constants;
    private readonly Action<GraphTensor[], ParallelOptions>[] _ops;
    private readonly int[][] _opOutputs;
    private readonly int[][] _deadAfter;   // tensors whose last reader is op i
    private readonly System.Collections.Concurrent.ConcurrentBag<TensorArena> _arenas = new();
    public TfLiteSignature Signature { get; }

    public GraphExecutor(TfLiteModel model, string signatureKey)
    {
        _model = model;
        Signature = model.Signature(signatureKey);
        _sg = model.Subgraph(Signature.SubgraphIndex);
        _constants = new GraphTensor[_sg.Tensors.Length];
        for (int i = 0; i < _sg.Tensors.Length; i++)
        {
            var t = _sg.Tensors[i];
            if (_model.IsConstant(t))
            {
                _constants[i] = LoadConstant(t);
            }
        }
        _ops = _sg.Operators.Select(Compile).ToArray();

        // Liveness: a tensor dies after the last op that reads it (or right after its producer if nothing does);
        // signature outputs and inputs never die inside a run.
        var lastUse = new int[_sg.Tensors.Length];
        Array.Fill(lastUse, -1);
        _opOutputs = new int[_sg.Operators.Length][];
        for (int i = 0; i < _sg.Operators.Length; i++)
        {
            var op = _sg.Operators[i];
            _opOutputs[i] = op.Outputs.Where(t => t >= 0).ToArray();
            foreach (int t in _opOutputs[i])
            {
                lastUse[t] = Math.Max(lastUse[t], i);
            }
            foreach (int t in op.Inputs)
            {
                if (t >= 0)
                {
                    lastUse[t] = Math.Max(lastUse[t], i);
                }
            }
        }
        foreach (int t in Signature.Outputs.Values.Concat(Signature.Inputs.Values))
        {
            lastUse[t] = -1;
        }
        var dead = Enumerable.Range(0, _sg.Operators.Length).Select(_ => new List<int>()).ToArray();
        for (int t = 0; t < lastUse.Length; t++)
        {
            if (lastUse[t] >= 0 && _constants[t] is null)
            {
                dead[lastUse[t]].Add(t);
            }
        }
        _deadAfter = dead.Select(l => l.ToArray()).ToArray();
    }

    public IReadOnlyDictionary<string, int> InputIndices => Signature.Inputs;
    public IReadOnlyDictionary<string, int> OutputIndices => Signature.Outputs;
    public TfLiteTensor TensorInfo(int index) => _sg.Tensors[index];

    private GraphTensor LoadConstant(TfLiteTensor t)
    {
        var shape = t.Shape.Length == 0 ? Array.Empty<int>() : t.Shape;
        switch (t.Type)
        {
            case TfLiteType.Float32:
            case TfLiteType.Float16:
                return GraphTensor.Float(shape, _model.ReadFloats(t));
            case TfLiteType.Int32:
            case TfLiteType.Int64:
                return GraphTensor.Int(shape, _model.ReadInts(t));
            case TfLiteType.Bool:
            {
                var raw = _model.RawData(t);
                var b = new bool[Math.Max(1, GraphTensor.Count(shape))];
                for (int i = 0; i < b.Length; i++)
                {
                    b[i] = raw[i] != 0;
                }
                return GraphTensor.Bool(shape, b);
            }
            case TfLiteType.Int8:
            case TfLiteType.Int4:
            case TfLiteType.Int2:
                // Quantized weights are consumed directly by the FC kernels; keep a dequantized copy only if
                // something else needs it (rare: small constant int8 tensors).
                return GraphTensor.Float(shape, _model.ReadFloats(t));
            default:
                throw new NotSupportedException($"Constant {t} of type {t.Type} is not supported.");
        }
    }

    /// <summary>Runs the signature. Inputs are keyed by signature input name; returns tensors keyed by output name.</summary>
    public Dictionary<string, GraphTensor> Run(IReadOnlyDictionary<string, GraphTensor> inputs, ParallelOptions po, Action<int, GraphTensor[]> opHook = null)
    {
        var values = (GraphTensor[])_constants.Clone();
        foreach (var (name, index) in Signature.Inputs)
        {
            if (!inputs.TryGetValue(name, out var v))
            {
                throw new ArgumentException($"Missing input '{name}'.");
            }
            values[index] = v;
        }
        // With a hook, intermediate tensors must stay intact for the caller: no recycling.
        var arena = opHook is null ? (_arenas.TryTake(out var a) ? a : new TensorArena()) : null;
        var previous = GraphTensor.CurrentArena;
        GraphTensor.CurrentArena = arena;
        try
        {
            for (int i = 0; i < _ops.Length; i++)
            {
                _ops[i](values, po);
                opHook?.Invoke(i, values);
                if (arena is not null)
                {
                    foreach (int t in _opOutputs[i])
                    {
                        arena.AddRef(values[t]?.F);
                    }
                    foreach (int t in _deadAfter[i])
                    {
                        arena.Release(values[t]?.F);
                        values[t] = null;
                    }
                }
            }
            var result = new Dictionary<string, GraphTensor>(StringComparer.Ordinal);
            foreach (var (name, index) in Signature.Outputs)
            {
                // Outputs leave the run: copy them out of recycled storage.
                result[name] = arena is null ? values[index] : values[index].Clone();
            }
            return result;
        }
        finally
        {
            GraphTensor.CurrentArena = previous;
            if (arena is not null)
            {
                arena.EndRun();
                _arenas.Add(arena);
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Op compilation
    // ---------------------------------------------------------------------------------------------

    private TfLiteQuantization Quant(int tensor) => _sg.Tensors[tensor].Quantization;

    private bool IsQuantized(int tensor) => _sg.Tensors[tensor].Type == TfLiteType.Int8 && Quant(tensor) is not null;

    /// <summary>TFLite <c>ActivationFunctionType</c>.</summary>
    private enum Activation { None = 0, Relu = 1, ReluN1To1 = 2, Relu6 = 3, Tanh = 4 }

    /// <summary>Reads an op's fused activation (slot 4 for ADD/SUB/MUL/DIV/FC options, 6 for CONCATENATION)
    /// and wraps the compiled kernel so the activation is applied to its float output.</summary>
    private static Action<GraphTensor[], ParallelOptions> WithActivation(TfLiteOperator op, int slot, Action<GraphTensor[], ParallelOptions> kernel)
    {
        var act = op.BuiltinOptions.IsNull ? Activation.None : (Activation)op.BuiltinOptions.GetSByte(slot);
        if (act == Activation.None)
        {
            return kernel;
        }
        int o = op.Outputs[0];
        return (v, po) =>
        {
            kernel(v, po);
            var t = v[o];
            if (t.F is null)
            {
                return;
            }
            var f = t.F.AsSpan(0, t.Length);
            switch (act)
            {
                case Activation.Relu: TensorPrimitives.Max(f, 0f, f); break;
                case Activation.Relu6: TensorPrimitives.Clamp(f, 0f, 6f, f); break;
                case Activation.ReluN1To1: TensorPrimitives.Clamp(f, -1f, 1f, f); break;
                case Activation.Tanh: TensorPrimitives.Tanh(f, f); break;
                default: throw new NotSupportedException($"Fused activation {act} is not supported.");
            }
        };
    }

    private Action<GraphTensor[], ParallelOptions> Compile(TfLiteOperator op)
    {
        var kernel = CompileKernel(op);
        var compiled = op.Opcode switch
        {
            TfLiteOp.Add or TfLiteOp.Sub or TfLiteOp.Mul or TfLiteOp.Div or TfLiteOp.FullyConnected => WithActivation(op, 4, kernel),
            TfLiteOp.Concatenation => WithActivation(op, 6, kernel),
            _ => kernel,
        };
        string name = $"op.{op.Opcode} [{(op.Outputs.Length > 0 && op.Outputs[0] >= 0 ? string.Join(",", _sg.Tensors[op.Outputs[0]].Shape) : "")}]";
        return (v, po) =>
        {
            using var _ = Profiler.Measure(name);
            compiled(v, po);
        };
    }

    private Action<GraphTensor[], ParallelOptions> CompileKernel(TfLiteOperator op)
    {
        int[] ins = op.Inputs, outs = op.Outputs;
        int o = outs.Length > 0 ? outs[0] : -1;
        int[] outShape = o >= 0 ? _sg.Tensors[o].Shape : null;
        switch (op.Opcode)
        {
            case TfLiteOp.Quantize:
            {
                var q = Quant(o);
                float s = q.Scale[0];
                int zp = q.ZeroPoint.Length > 0 ? (int)q.ZeroPoint[0] : 0;
                return (v, po) => v[o] = FakeQuantize(v[ins[0]], s, zp);
            }
            case TfLiteOp.Dequantize:
                return (v, po) => v[o] = v[ins[0]];
            case TfLiteOp.Reshape:
                return (v, po) => v[o] = v[ins[0]].WithShape(outShape);
            case TfLiteOp.Cast:
            {
                var to = _sg.Tensors[o].Type;
                return (v, po) => v[o] = Cast(v[ins[0]], to);
            }
            case TfLiteOp.Add: return Binary(ins, o, BinaryOp.Add, (a, b) => a + b);
            case TfLiteOp.Sub: return Binary(ins, o, BinaryOp.Sub, (a, b) => a - b);
            case TfLiteOp.Mul: return Binary(ins, o, BinaryOp.Mul, (a, b) => a * b);
            case TfLiteOp.Div: return Binary(ins, o, BinaryOp.Div, (a, b) => a / b);
            case TfLiteOp.Maximum: return Binary(ins, o, BinaryOp.Max, MathF.Max);
            case TfLiteOp.Minimum: return Binary(ins, o, BinaryOp.Min, MathF.Min);
            case TfLiteOp.Rsqrt: return Unary(ins, o, Xnn.ReciprocalSqrt);
            case TfLiteOp.Sqrt: return Unary(ins, o, MathF.Sqrt);
            case TfLiteOp.Logistic: return (v, po) => { var r = GraphTensor.Float(v[ins[0]].Shape); Xnn.Sigmoid(v[ins[0]].F.AsSpan(0, r.F.Length), r.F); v[o] = r; };
            case TfLiteOp.Tanh: return (v, po) => { var r = GraphTensor.Float(v[ins[0]].Shape); Xnn.Tanh(v[ins[0]].F.AsSpan(0, r.F.Length), r.F); v[o] = r; };
            case TfLiteOp.Exp: return (v, po) => { var r = GraphTensor.Float(v[ins[0]].Shape); TensorPrimitives.Exp(v[ins[0]].F.AsSpan(0, r.F.Length), r.F); v[o] = r; };
            case TfLiteOp.Abs: return Unary(ins, o, MathF.Abs);
            case TfLiteOp.NotEqual: return Compare(ins, o, (a, b) => a != b);
            case TfLiteOp.Less: return Compare(ins, o, (a, b) => a < b);
            case TfLiteOp.Greater: return Compare(ins, o, (a, b) => a > b);
            case TfLiteOp.GreaterEqual: return Compare(ins, o, (a, b) => a >= b);
            case TfLiteOp.LogicalNot: return (v, po) => { var a = v[ins[0]]; var r = GraphTensor.Bool(a.Shape); for (int i = 0; i < r.B.Length; i++) r.B[i] = !a.B[i]; v[o] = r; };
            case TfLiteOp.LogicalAnd: return (v, po) => v[o] = BroadcastBool(v[ins[0]], v[ins[1]], outShape, (a, b) => a && b);
            case TfLiteOp.SelectV2: return (v, po) => v[o] = Select(v[ins[0]], v[ins[1]], v[ins[2]], outShape);
            case TfLiteOp.Softmax:
            {
                float beta = op.BuiltinOptions.IsNull ? 1f : op.BuiltinOptions.GetFloat(4, 1f);
                return (v, po) => v[o] = Softmax(v[ins[0]], beta);
            }
            case TfLiteOp.Mean:
            case TfLiteOp.Sum:
            {
                bool mean = op.Opcode == TfLiteOp.Mean;
                // XNNPACK rewrites mul(x, x) into sqr(x) and reduce(sqr(x)) into a fused sum/mean of squares that
                // accumulates x·x with FMAs; reduce from x itself in that case.
                var producer = _sg.Operators.FirstOrDefault(p => p.Outputs.Length > 0 && p.Outputs[0] == ins[0]);
                bool squared = producer is not null && producer.Opcode == TfLiteOp.Mul && producer.Inputs.Length == 2 && producer.Inputs[0] == producer.Inputs[1];
                int src = squared ? producer.Inputs[0] : ins[0];
                return (v, po) => v[o] = Reduce(v[src], v[ins[1]].I, outShape, mean, squared);
            }
            case TfLiteOp.Pad: return (v, po) => v[o] = Pad(v[ins[0]], v[ins[1]].I, outShape);
            case TfLiteOp.Slice: return (v, po) => v[o] = Slice(v[ins[0]], v[ins[1]].I, outShape);
            case TfLiteOp.Transpose: return (v, po) => v[o] = Transpose(v[ins[0]], v[ins[1]].I);
            case TfLiteOp.Concatenation:
            {
                int axis = op.BuiltinOptions.IsNull ? 0 : op.BuiltinOptions.GetInt32(4);
                bool requant = IsQuantized(o);
                float outScale = requant ? Quant(o).Scale[0] : 0;
                var inScales = ins.Select(i => IsQuantized(i) ? Quant(i).Scale[0] : 0f).ToArray();
                return (v, po) => v[o] = Concat(ins.Select(i => v[i]).ToArray(), axis, outShape, requant ? outScale : 0, inScales);
            }
            case TfLiteOp.Tile: return (v, po) => v[o] = Tile(v[ins[0]], v[ins[1]].I);
            case TfLiteOp.BatchMatMul:
            {
                bool adjX = !op.BuiltinOptions.IsNull && op.BuiltinOptions.GetBool(4);
                bool adjY = !op.BuiltinOptions.IsNull && op.BuiltinOptions.GetBool(6);
                // XNNPACK's rewrite_dequant_bmm: when B is the only use of DEQUANTIZE(int8, zero point 0), the
                // product runs as a dynamically quantized (qd8) GEMM against the int8 values instead of in float.
                var bProducer = _sg.Operators.FirstOrDefault(p => p.Outputs.Length > 0 && p.Outputs[0] == ins[1]);
                if (bProducer is not null && bProducer.Opcode == TfLiteOp.Dequantize
                    && _sg.Tensors[bProducer.Inputs[0]].Type == TfLiteType.Int8 && Quant(bProducer.Inputs[0]) is { } bq
                    && bq.Scale.Length == 1 && (bq.ZeroPoint.Length == 0 || bq.ZeroPoint[0] == 0)
                    && _sg.Operators.Count(p => p.Inputs.Contains(ins[1])) == 1 && !Signature.Outputs.Values.Contains(ins[1]))
                {
                    float bScale = bq.Scale[0];
                    return (v, po) => v[o] = BatchMatMulQuantizedB(v[ins[0]], v[ins[1]], bScale, adjX, adjY, outShape, po);
                }
                return (v, po) => v[o] = BatchMatMul(v[ins[0]], v[ins[1]], adjX, adjY, outShape, po);
            }
            case TfLiteOp.FullyConnected: return CompileFullyConnected(op);
            case TfLiteOp.Conv2D: return CompileConv(op, depthwise: false);
            case TfLiteOp.DepthwiseConv2D: return CompileConv(op, depthwise: true);
            case TfLiteOp.StableHloComposite when op.CompositeName == "odml.rms_norm":
            {
                var w = _constants[ins[1]].F;
                float eps = RmsNormEpsilon(op);
                return (v, po) =>
                {
                    var x = v[ins[0]];
                    var r = GraphTensor.Float(x.Shape);
                    Ops.RmsNorm(x.F.AsSpan(0, r.F.Length), w, r.F, w.Length, eps);
                    v[o] = r;
                };
            }
            default:
                throw new NotSupportedException($"Operator {op} (opcode {op.Opcode}) is not supported by the graph executor.");
        }
    }

    private float RmsNormEpsilon(TfLiteOperator op) => _model.RmsNormEpsilon(op);

    private Action<GraphTensor[], ParallelOptions> CompileFullyConnected(TfLiteOperator op)
    {
        int x = op.Inputs[0], wIdx = op.Inputs[1], o = op.Outputs[0];
        int bIdx = op.Inputs.Length > 2 ? op.Inputs[2] : -1;
        var wt = _sg.Tensors[wIdx];
        int outF = wt.Shape[0], inF = wt.Shape[1];
        var outShape = _sg.Tensors[o].Shape;
        var bias = bIdx >= 0 ? _constants[bIdx]?.F : null;
        if (wt.Type == TfLiteType.Float32)
        {
            var wT = Transpose2D(_constants[wIdx].F, outF, inF);   // [in, out] for C = X·Wᵀ
            return (v, po) =>
            {
                var xin = v[x];
                int rows = xin.Length / inF;
                var r = GraphTensor.Float(outShape);
                // XNNPACK's f32 GEMM starts each output's FMA chain from the bias.
                if (bias is not null)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        bias.AsSpan(0, outF).CopyTo(r.F.AsSpan(i * outF, outF));
                    }
                }
                SGemm.Multiply(xin.F, inF, wT, outF, r.F, outF, rows, outF, inF, 1f, po, accumulate: bias is not null);
                v[o] = r;
            };
        }
        if (!IsQuantized(x) || !IsQuantized(o))
        {
            throw new NotSupportedException($"FC {op}: only float or fully int8-quantized (static) FCs are supported.");
        }
        var matrix = QuantizedMatrix.FromTensor(_model, wt);
        var qin = Quant(x);
        var qout = Quant(o);
        float sIn = qin.Scale[0];
        int zpIn = qin.ZeroPoint.Length > 0 ? (int)qin.ZeroPoint[0] : 0;
        var requant = new QGemm.Requantization(sIn, matrix.Scale, qout.Scale[0]);
        if (bias is not null || (qout.ZeroPoint.Length > 0 && qout.ZeroPoint[0] != 0))
        {
            throw new NotSupportedException($"FC {op}: bias / asymmetric output on int8 FC is not supported.");
        }
        return (v, po) =>
        {
            var xin = v[x];
            int rows = xin.Length / inF;
            var qa = new QuantizedActivations();
            qa.QuantizeStatic(xin.F, rows, inF, inF, sIn, zpIn, po);
            var r = GraphTensor.Float(outShape);
            QGemm.Multiply(qa, matrix, r.F, outF, po, requant);
            v[o] = r;
        };
    }

    private Action<GraphTensor[], ParallelOptions> CompileConv(TfLiteOperator op, bool depthwise)
    {
        int x = op.Inputs[0], wIdx = op.Inputs[1], o = op.Outputs[0];
        int bIdx = op.Inputs.Length > 2 ? op.Inputs[2] : -1;
        var opt = op.BuiltinOptions;
        // Conv2DOptions: padding(4), stride_w(6), stride_h(8), activation(10), dilation_w(12), dilation_h(14)
        // DepthwiseConv2DOptions: padding(4), stride_w(6), stride_h(8), depth_multiplier(10), activation(12), dilation_w(14), dilation_h(16)
        int padding = opt.GetByte(4);
        int sw = opt.GetInt32(6, 1), sh = opt.GetInt32(8, 1);
        int act = depthwise ? opt.GetByte(12) : opt.GetByte(10);
        int dw = depthwise ? opt.GetInt32(14, 1) : opt.GetInt32(12, 1);
        int dh = depthwise ? opt.GetInt32(16, 1) : opt.GetInt32(14, 1);
        if (act != 0)
        {
            throw new NotSupportedException($"Fused activation {act} on {op} is not supported.");
        }
        var w = _constants[wIdx].F;
        var wShape = _sg.Tensors[wIdx].Shape;   // conv: [O, kh, kw, I]; depthwise: [1, kh, kw, C·mult]
        var bias = bIdx >= 0 ? _constants[bIdx]?.F : null;
        var outShape = _sg.Tensors[o].Shape;
        // Regular convolution: weights [OC, kh, kw, C] regrouped once as [kh·kw·C, OC] so each FMA step updates all
        // output channels of one pixel (per-output order unchanged).
        float[] wt = null;
        if (!depthwise)
        {
            int oc = wShape[0], taps = wShape[1] * wShape[2] * wShape[3];
            wt = new float[taps * oc];
            for (int c = 0; c < oc; c++)
            {
                for (int t = 0; t < taps; t++)
                {
                    wt[t * oc + c] = w[c * taps + t];
                }
            }
        }
        return (v, po) => v[o] = Conv(v[x], w, wt, wShape, bias, outShape, sh, sw, dh, dw, padding == 0 /* SAME */, depthwise, po);
    }

    // ---------------------------------------------------------------------------------------------
    // Kernels
    // ---------------------------------------------------------------------------------------------

    private static float[] Transpose2D(float[] m, int rows, int cols)
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

    private static GraphTensor FakeQuantize(GraphTensor x, float scale, int zp)
    {
        // XNNPACK f32-qs8-vcvt: q = sat8(sat16(rint(x · (1/scale))) + zp), kept dequantized.
        var r = GraphTensor.Float(x.Shape);
        float inv = 1f / scale;
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            ref float xr = ref MemoryMarshal.GetReference(x.F.AsSpan());
            ref float rr = ref MemoryMarshal.GetReference(r.F.AsSpan());
            var vinv = Vector256.Create(inv);
            var vscale = Vector256.Create(scale);
            var vzp = Vector256.Create(zp);
            for (; i + 8 <= r.F.Length; i += 8)
            {
                var q = Vector256.Round(Vector256.LoadUnsafe(ref xr, (nuint)i) * vinv);
                q = Vector256.Min(Vector256.Max(q, Vector256.Create((float)short.MinValue)), Vector256.Create((float)short.MaxValue));
                var qi = Vector256.Min(Vector256.Max(Vector256.ConvertToInt32(q) + vzp, Vector256.Create(-128)), Vector256.Create(127)) - vzp;
                (Vector256.ConvertToSingle(qi) * vscale).StoreUnsafe(ref rr, (nuint)i);
            }
        }
        for (; i < r.F.Length; i++)
        {
            // Through an integer like the real int8 tensor, so a quantized zero dequantizes to +0 (never -0).
            int q = (int)Math.Clamp(MathF.Round(x.F[i] * inv, MidpointRounding.ToEven), short.MinValue, short.MaxValue) + zp;
            r.F[i] = (Math.Clamp(q, -128, 127) - zp) * scale;
        }
        return r;
    }

    private static GraphTensor Cast(GraphTensor x, TfLiteType to)
    {
        int n = x.Length;
        switch (to)
        {
            case TfLiteType.Float32:
            {
                var r = GraphTensor.Float(x.Shape);
                if (x.B is not null) for (int i = 0; i < n; i++) r.F[i] = x.B[i] ? 1f : 0f;
                else if (x.I is not null) for (int i = 0; i < n; i++) r.F[i] = x.I[i];
                else Array.Copy(x.F, r.F, n);
                return r;
            }
            case TfLiteType.Bool:
            {
                var r = GraphTensor.Bool(x.Shape);
                if (x.F is not null) for (int i = 0; i < n; i++) r.B[i] = x.F[i] != 0;
                else if (x.I is not null) for (int i = 0; i < n; i++) r.B[i] = x.I[i] != 0;
                else Array.Copy(x.B, r.B, n);
                return r;
            }
            case TfLiteType.Int32:
            case TfLiteType.Int64:
            {
                var r = GraphTensor.Int(x.Shape);
                if (x.B is not null) for (int i = 0; i < n; i++) r.I[i] = x.B[i] ? 1 : 0;
                else if (x.F is not null) for (int i = 0; i < n; i++) r.I[i] = (int)x.F[i];
                else Array.Copy(x.I, r.I, n);
                return r;
            }
            default:
                throw new NotSupportedException($"CAST to {to} is not supported.");
        }
    }

    /// <summary>Row-major strides of <paramref name="shape"/> broadcast to rank <paramref name="rank"/> (0 stride for size-1 dims).</summary>
    private static int[] BroadcastStrides(int[] shape, int[] outShape)
    {
        int rank = outShape.Length;
        var strides = new int[rank];
        int s = 1;
        for (int d = rank - 1, k = shape.Length - 1; d >= 0; d--, k--)
        {
            int dim = k >= 0 ? shape[k] : 1;
            strides[d] = dim == 1 ? 0 : s;
            s *= dim;
        }
        return strides;
    }

    private static int[] BroadcastShape(int[] a, int[] b)
    {
        int rank = Math.Max(a.Length, b.Length);
        var r = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            int da = d - (rank - a.Length) >= 0 ? a[d - (rank - a.Length)] : 1;
            int db = d - (rank - b.Length) >= 0 ? b[d - (rank - b.Length)] : 1;
            r[d] = Math.Max(da, db);
        }
        return r;
    }

    private static bool SameShape(int[] a, int[] b) => a.AsSpan().SequenceEqual(b);

    /// <summary>Visits every output element with the flat indices of two broadcast inputs.</summary>
    private static void ForEachBroadcast(int[] outShape, int[] sa, int[] sb, Action<int, int, int> body)
    {
        int rank = outShape.Length;
        int n = GraphTensor.Count(outShape);
        var idx = new int[rank];
        int ia = 0, ib = 0;
        for (int i = 0; i < n; i++)
        {
            body(i, ia, ib);
            for (int d = rank - 1; d >= 0; d--)
            {
                idx[d]++;
                ia += sa[d];
                ib += sb[d];
                if (idx[d] < outShape[d])
                {
                    break;
                }
                ia -= sa[d] * outShape[d];
                ib -= sb[d] * outShape[d];
                idx[d] = 0;
            }
        }
    }

    private enum BinaryOp { Add, Sub, Mul, Div, Max, Min }

    /// <summary>Element-wise <paramref name="op"/> of two rows, or of a row and a scalar (a null span means
    /// "use the scalar"); <see cref="TensorPrimitives"/> computes exactly the scalar IEEE operations.</summary>
    private static void ApplyRow(BinaryOp op, ReadOnlySpan<float> a, float sa, ReadOnlySpan<float> b, float sb, Span<float> r, bool aScalar, bool bScalar)
    {
        if (!aScalar && !bScalar)
        {
            switch (op)
            {
                case BinaryOp.Add: TensorPrimitives.Add(a, b, r); break;
                case BinaryOp.Sub: TensorPrimitives.Subtract(a, b, r); break;
                case BinaryOp.Mul: TensorPrimitives.Multiply(a, b, r); break;
                case BinaryOp.Div: TensorPrimitives.Divide(a, b, r); break;
                case BinaryOp.Max: TensorPrimitives.Max(a, b, r); break;
                case BinaryOp.Min: TensorPrimitives.Min(a, b, r); break;
            }
        }
        else if (bScalar && !aScalar)
        {
            switch (op)
            {
                case BinaryOp.Add: TensorPrimitives.Add(a, sb, r); break;
                case BinaryOp.Sub: TensorPrimitives.Subtract(a, sb, r); break;
                case BinaryOp.Mul: TensorPrimitives.Multiply(a, sb, r); break;
                case BinaryOp.Div: TensorPrimitives.Divide(a, sb, r); break;
                case BinaryOp.Max: TensorPrimitives.Max(a, sb, r); break;
                case BinaryOp.Min: TensorPrimitives.Min(a, sb, r); break;
            }
        }
        else if (aScalar && !bScalar)
        {
            switch (op)
            {
                case BinaryOp.Add: TensorPrimitives.Add(b, sa, r); break;
                case BinaryOp.Sub: TensorPrimitives.Subtract(sa, b, r); break;
                case BinaryOp.Mul: TensorPrimitives.Multiply(b, sa, r); break;
                case BinaryOp.Div: TensorPrimitives.Divide(sa, b, r); break;
                case BinaryOp.Max: TensorPrimitives.Max(b, sa, r); break;
                case BinaryOp.Min: TensorPrimitives.Min(b, sa, r); break;
            }
        }
        else
        {
            float v = op switch
            {
                BinaryOp.Add => sa + sb,
                BinaryOp.Sub => sa - sb,
                BinaryOp.Mul => sa * sb,
                BinaryOp.Div => sa / sb,
                BinaryOp.Max => MathF.Max(sa, sb),
                _ => MathF.Min(sa, sb),
            };
            r.Fill(v);
        }
    }

    private static Action<GraphTensor[], ParallelOptions> Binary(int[] ins, int o, BinaryOp op, Func<float, float, float> f)
        => (v, po) =>
        {
            var a = v[ins[0]];
            var b = v[ins[1]];
            if (a.F is null || b.F is null)
            {
                // Integer arithmetic (shape computations in a few graphs).
                var shapeI = BroadcastShape(a.Shape, b.Shape);
                var ri = GraphTensor.Int(shapeI);
                ForEachBroadcast(shapeI, BroadcastStrides(a.Shape, shapeI), BroadcastStrides(b.Shape, shapeI),
                    (i, ia, ib) => ri.I[i] = (int)f(a.I[ia], b.I[ib]));
                v[o] = ri;
                return;
            }
            var shape = BroadcastShape(a.Shape, b.Shape);
            var r = GraphTensor.Float(shape);
            int n = r.F.Length, na = a.Length, nb = b.Length;
            if (SameShape(a.Shape, b.Shape) || (na == n && nb == n))
            {
                ApplyRow(op, a.F.AsSpan(0, n), 0, b.F.AsSpan(0, n), 0, r.F, false, false);
            }
            else if (nb == 1)
            {
                ApplyRow(op, a.F.AsSpan(0, n), 0, default, b.F[0], r.F, false, true);
            }
            else if (na == 1)
            {
                ApplyRow(op, default, a.F[0], b.F.AsSpan(0, n), 0, r.F, true, false);
            }
            else
            {
                // Rows of the innermost output dimension: each input is either contiguous along it or broadcast.
                int rank = shape.Length;
                var sa = BroadcastStrides(a.Shape, shape);
                var sb = BroadcastStrides(b.Shape, shape);
                int inner = shape[rank - 1];
                bool aScalar = sa[rank - 1] == 0, bScalar = sb[rank - 1] == 0;
                var idx = new int[rank];
                int ia = 0, ib = 0;
                for (int i = 0; i < n; i += inner)
                {
                    ApplyRow(op, aScalar ? default : a.F.AsSpan(ia, inner), a.F[ia], bScalar ? default : b.F.AsSpan(ib, inner), b.F[ib],
                        r.F.AsSpan(i, inner), aScalar, bScalar);
                    for (int d = rank - 2; d >= 0; d--)
                    {
                        idx[d]++;
                        ia += sa[d];
                        ib += sb[d];
                        if (idx[d] < shape[d])
                        {
                            break;
                        }
                        ia -= sa[d] * shape[d];
                        ib -= sb[d] * shape[d];
                        idx[d] = 0;
                    }
                }
            }
            v[o] = r;
        };

    private static Action<GraphTensor[], ParallelOptions> Unary(int[] ins, int o, Func<float, float> f)
        => (v, po) =>
        {
            var a = v[ins[0]];
            var r = GraphTensor.Float(a.Shape);
            for (int i = 0; i < r.F.Length; i++) r.F[i] = f(a.F[i]);
            v[o] = r;
        };

    private static Action<GraphTensor[], ParallelOptions> Compare(int[] ins, int o, Func<float, float, bool> f)
        => (v, po) =>
        {
            var a = v[ins[0]];
            var b = v[ins[1]];
            var shape = BroadcastShape(a.Shape, b.Shape);
            var r = GraphTensor.Bool(shape);
            Func<GraphTensor, int, float> get = (t, i) => t.F is not null ? t.F[i] : t.I is not null ? t.I[i] : (t.B[i] ? 1f : 0f);
            ForEachBroadcast(shape, BroadcastStrides(a.Shape, shape), BroadcastStrides(b.Shape, shape),
                (i, ia, ib) => r.B[i] = f(get(a, ia), get(b, ib)));
            v[o] = r;
        };

    private static GraphTensor BroadcastBool(GraphTensor a, GraphTensor b, int[] outShape, Func<bool, bool, bool> f)
    {
        var shape = BroadcastShape(a.Shape, b.Shape);
        var r = GraphTensor.Bool(shape);
        ForEachBroadcast(shape, BroadcastStrides(a.Shape, shape), BroadcastStrides(b.Shape, shape), (i, ia, ib) => r.B[i] = f(a.B[ia], b.B[ib]));
        return r;
    }

    private static GraphTensor Select(GraphTensor cond, GraphTensor x, GraphTensor y, int[] outShape)
    {
        var shape = BroadcastShape(BroadcastShape(cond.Shape, x.Shape), y.Shape);
        var r = GraphTensor.Float(shape);
        var sc = BroadcastStrides(cond.Shape, shape);
        var sx = BroadcastStrides(x.Shape, shape);
        var sy = BroadcastStrides(y.Shape, shape);
        // Two passes of the pairwise walker: (cond, x) then (cond, y) would duplicate work; walk all three.
        int rank = shape.Length;
        var idx = new int[rank];
        int ic = 0, ix = 0, iy = 0;
        for (int i = 0; i < r.F.Length; i++)
        {
            r.F[i] = cond.B[ic] ? x.F[ix] : y.F[iy];
            for (int d = rank - 1; d >= 0; d--)
            {
                idx[d]++;
                ic += sc[d];
                ix += sx[d];
                iy += sy[d];
                if (idx[d] < shape[d])
                {
                    break;
                }
                ic -= sc[d] * shape[d];
                ix -= sx[d] * shape[d];
                iy -= sy[d] * shape[d];
                idx[d] = 0;
            }
        }
        return r;
    }

    private static GraphTensor Softmax(GraphTensor x, float beta)
    {
        int last = x.Shape[^1];
        var r = GraphTensor.Float(x.Shape);
        if (beta != 1f)
        {
            TensorPrimitives.Multiply(x.F.AsSpan(0, r.F.Length), beta, r.F);
        }
        else
        {
            Array.Copy(x.F, r.F, r.F.Length);
        }
        Ops.SoftmaxRows(r.F, r.F.Length / last, last, last);
        return r;
    }

    private static GraphTensor Reduce(GraphTensor x, int[] axes, int[] outShape, bool mean, bool squared = false)
    {
        int rank = x.Shape.Length;
        var reduce = new bool[rank];
        foreach (var a in axes)
        {
            reduce[a < 0 ? a + rank : a] = true;
        }
        // Fast path: reduce over the trailing axis only.
        if (rank > 0 && reduce[rank - 1] && reduce.Count(b => b) == 1)
        {
            int last = x.Shape[^1];
            int rows = x.Length / last;
            var r1 = GraphTensor.Float(outShape);
            // XNNPACK f32-rsum / f32-rsum2 (AVX-512 lane layout), scaled by 1/n for MEAN.
            float scale = mean ? 1f / last : 1f;
            for (int i = 0; i < rows; i++)
            {
                var row = x.F.AsSpan(i * last, last);
                r1.F[i] = squared ? Xnn.SumOfSquares(row, scale) : Xnn.Sum(row, scale);
            }
            return r1;
        }
        if (squared)
        {
            var sq = GraphTensor.Float(x.Shape);
            TensorPrimitives.Multiply(x.F.AsSpan(0, sq.F.Length), x.F.AsSpan(0, sq.F.Length), sq.F);
            x = sq;
        }
        var keptShape = new int[rank];
        int count = 1;
        for (int d = 0; d < rank; d++)
        {
            keptShape[d] = reduce[d] ? 1 : x.Shape[d];
            if (reduce[d])
            {
                count *= x.Shape[d];
            }
        }
        var r = GraphTensor.Float(outShape);
        var strides = BroadcastStrides(keptShape, x.Shape);
        var idx = new int[rank];
        int io = 0;
        for (int i = 0; i < x.Length; i++)
        {
            r.F[io] += x.F[i];
            for (int d = rank - 1; d >= 0; d--)
            {
                idx[d]++;
                io += strides[d];
                if (idx[d] < x.Shape[d])
                {
                    break;
                }
                io -= strides[d] * x.Shape[d];
                idx[d] = 0;
            }
        }
        if (mean)
        {
            TensorPrimitives.Divide(r.F, count, r.F);
        }
        return r;
    }

    private static GraphTensor NewLike(GraphTensor x, int[] shape)
        => x.F is not null ? GraphTensor.Float(shape) : x.I is not null ? GraphTensor.Int(shape) : GraphTensor.Bool(shape);

    /// <summary>Copies element <paramref name="from"/> of <paramref name="src"/> to element <paramref name="to"/> of <paramref name="dst"/>.</summary>
    private static void CopyElement(GraphTensor src, int from, GraphTensor dst, int to)
    {
        if (src.F is not null) dst.F[to] = src.F[from];
        else if (src.I is not null) dst.I[to] = src.I[from];
        else dst.B[to] = src.B[from];
    }

    private static int[] RowMajorStrides(int[] shape)
    {
        var s = new int[shape.Length];
        int acc = 1;
        for (int d = shape.Length - 1; d >= 0; d--)
        {
            s[d] = acc;
            acc *= shape[d];
        }
        return s;
    }

    private static GraphTensor Pad(GraphTensor x, int[] paddings, int[] outShape)
    {
        // Constant padding with 0 (the zero point of every quantized tensor in these graphs): copy each innermost
        // input row to its place in the zero-initialized output.
        var r = NewLike(x, outShape);
        int rank = x.Shape.Length;
        int n = x.Length;
        if (n == 0)
        {
            return r;
        }
        var so = RowMajorStrides(outShape);
        int inner = x.Shape[rank - 1];
        var idx = new int[rank];
        for (int i = 0; i < n; i += inner)
        {
            int to = paddings[2 * (rank - 1)];
            for (int d = 0; d < rank - 1; d++)
            {
                to += (idx[d] + paddings[2 * d]) * so[d];
            }
            CopyRow(x, i, r, to, inner);
            for (int d = rank - 2; d >= 0; d--)
            {
                if (++idx[d] < x.Shape[d])
                {
                    break;
                }
                idx[d] = 0;
            }
        }
        return r;
    }

    private static void CopyRow(GraphTensor src, int from, GraphTensor dst, int to, int count)
    {
        if (src.F is not null) Array.Copy(src.F, from, dst.F, to, count);
        else if (src.I is not null) Array.Copy(src.I, from, dst.I, to, count);
        else Array.Copy(src.B, from, dst.B, to, count);
    }

    private static GraphTensor Slice(GraphTensor x, int[] begin, int[] outShape)
    {
        var r = NewLike(x, outShape);
        int rank = outShape.Length;
        var sx = RowMajorStrides(x.Shape);
        var idx = new int[rank];
        int n = GraphTensor.Count(outShape);
        // Copy contiguous innermost runs.
        int inner = outShape[rank - 1];
        for (int i = 0; i < n; i += inner)
        {
            int from = 0;
            for (int d = 0; d < rank; d++)
            {
                from += (idx[d] + begin[d]) * sx[d];
            }
            if (x.F is not null) Array.Copy(x.F, from, r.F, i, inner);
            else if (x.I is not null) Array.Copy(x.I, from, r.I, i, inner);
            else Array.Copy(x.B, from, r.B, i, inner);
            for (int d = rank - 2; d >= 0; d--)
            {
                if (++idx[d] < outShape[d])
                {
                    break;
                }
                idx[d] = 0;
            }
        }
        return r;
    }

    private static GraphTensor Transpose(GraphTensor x, int[] perm)
    {
        int rank = perm.Length;
        var outShape = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            outShape[d] = x.Shape[perm[d]];
        }
        var r = NewLike(x, outShape);
        var sx = RowMajorStrides(x.Shape);
        var srcStride = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            srcStride[d] = sx[perm[d]];
        }
        if (r.F is not null) TransposeCore(x.F, r.F, outShape, srcStride);
        else if (r.I is not null) TransposeCore(x.I, r.I, outShape, srcStride);
        else TransposeCore(x.B, r.B, outShape, srcStride);
        return r;
    }

    private static void TransposeCore<T>(T[] src, T[] dst, int[] outShape, int[] srcStride)
    {
        int rank = outShape.Length;
        int n = GraphTensor.Count(outShape);
        if (n == 0)
        {
            return;
        }
        int inner = outShape[rank - 1];
        int innerStride = srcStride[rank - 1];
        var idx = new int[rank];
        int from = 0;
        for (int i = 0; i < n; i += inner)
        {
            if (innerStride == 1)
            {
                Array.Copy(src, from, dst, i, inner);
            }
            else
            {
                for (int j = 0, f = from; j < inner; j++, f += innerStride)
                {
                    dst[i + j] = src[f];
                }
            }
            for (int d = rank - 2; d >= 0; d--)
            {
                idx[d]++;
                from += srcStride[d];
                if (idx[d] < outShape[d])
                {
                    break;
                }
                from -= srcStride[d] * outShape[d];
                idx[d] = 0;
            }
        }
    }

    private static GraphTensor Concat(GraphTensor[] parts, int axis, int[] outShape, float outScale, float[] inScales)
    {
        int rank = outShape.Length;
        if (axis < 0)
        {
            axis += rank;
        }
        var r = NewLike(parts[0], outShape);
        int outer = 1;
        for (int d = 0; d < axis; d++)
        {
            outer *= outShape[d];
        }
        int innerOut = r.Length / outer;
        int offset = 0;
        for (int p = 0; p < parts.Length; p++)
        {
            var t = parts[p];
            int innerIn = t.Length / outer;
            bool requant = outScale > 0 && inScales[p] > 0 && inScales[p] != outScale;
            for (int o = 0; o < outer; o++)
            {
                if (t.F is not null)
                {
                    Array.Copy(t.F, o * innerIn, r.F, o * innerOut + offset, innerIn);
                    if (requant)
                    {
                        for (int i = 0; i < innerIn; i++)
                        {
                            ref float val = ref r.F[o * innerOut + offset + i];
                            val = Math.Clamp(MathF.Round(val / outScale, MidpointRounding.ToEven), -128f, 127f) * outScale;
                        }
                    }
                }
                else if (t.I is not null) Array.Copy(t.I, o * innerIn, r.I, o * innerOut + offset, innerIn);
                else Array.Copy(t.B, o * innerIn, r.B, o * innerOut + offset, innerIn);
            }
            offset += innerIn;
        }
        return r;
    }

    private static GraphTensor Tile(GraphTensor x, int[] multiples)
    {
        int rank = x.Shape.Length;
        var outShape = new int[rank];
        for (int d = 0; d < rank; d++)
        {
            outShape[d] = x.Shape[d] * multiples[d];
        }
        var r = NewLike(x, outShape);
        var sx = RowMajorStrides(x.Shape);
        var idx = new int[rank];
        for (int i = 0; i < r.Length; i++)
        {
            int from = 0;
            for (int d = 0; d < rank; d++)
            {
                from += (idx[d] % x.Shape[d]) * sx[d];
            }
            CopyElement(x, from, r, i);
            for (int d = rank - 1; d >= 0; d--)
            {
                if (++idx[d] < outShape[d])
                {
                    break;
                }
                idx[d] = 0;
            }
        }
        return r;
    }

    /// <summary>
    /// <c>BATCH_MATMUL</c> after XNNPACK's dequant rewrite (f32 × qc8w GEMM): A stays float, each output is one
    /// FMA chain over k against B's raw int8 values, and the sum is scaled by B's scale at the end.
    /// B arrives as its dequantized values (exact multiples of <paramref name="bScale"/>).
    /// </summary>
    private static GraphTensor BatchMatMulQuantizedB(GraphTensor a, GraphTensor b, float bScale, bool adjX, bool adjY, int[] outShape, ParallelOptions po)
    {
        int ra = a.Shape.Length, rb = b.Shape.Length;
        int m = adjX ? a.Shape[ra - 1] : a.Shape[ra - 2];
        int k = adjX ? a.Shape[ra - 2] : a.Shape[ra - 1];
        int n = adjY ? b.Shape[rb - 2] : b.Shape[rb - 1];
        var (offA, offB) = BatchOffsets(a.Shape[..(ra - 2)], b.Shape[..(rb - 2)]);
        var r = GraphTensor.Float(outShape);
        int matA = a.Shape[ra - 2] * a.Shape[ra - 1];
        int matB = b.Shape[rb - 2] * b.Shape[rb - 1];
        float invB = 1f / bScale;
        var af = a.F;
        var bf = b.F;
        var rf = r.F;
        void Batch(int bi)
        {
            var A = af.AsSpan(offA[bi] * matA, matA);
            var B = bf.AsSpan(offB[bi] * matB, matB);
            var qb = new float[k * n];   // B's int8 values as floats, [k, n]
            for (int p = 0; p < k; p++)
            {
                for (int j = 0; j < n; j++)
                {
                    qb[p * n + j] = MathF.Round((adjY ? B[j * k + p] : B[p * n + j]) * invB);
                }
            }
            for (int i = 0; i < m; i++)
            {
                var dst = rf.AsSpan((bi * m + i) * n, n);
                dst.Clear();
                for (int p = 0; p < k; p++)
                {
                    float av = adjX ? A[p * m + i] : A[i * k + p];
                    TensorPrimitives.FusedMultiplyAdd(qb.AsSpan(p * n, n), av, dst, dst);
                }
                TensorPrimitives.Multiply(dst, bScale, dst);
            }
        }
        RunBatches(offA.Length, (long)m * n * k, po, Batch);
        return r;
    }

    /// <summary>Flat batch indices of A and B for every (broadcast) batch element of a <c>BATCH_MATMUL</c>.</summary>
    private static (int[] A, int[] B) BatchOffsets(int[] batchA, int[] batchB)
    {
        var batch = BroadcastShape(batchA, batchB);
        int nb = GraphTensor.Count(batch);
        var sa = BroadcastStrides(batchA, batch);
        var sb = BroadcastStrides(batchB, batch);
        var offA = new int[nb];
        var offB = new int[nb];
        var idx = new int[batch.Length];
        int ia = 0, ib = 0;
        for (int bi = 0; bi < nb; bi++)
        {
            offA[bi] = ia;
            offB[bi] = ib;
            for (int d = batch.Length - 1; d >= 0; d--)
            {
                idx[d]++;
                ia += sa[d];
                ib += sb[d];
                if (idx[d] < batch[d])
                {
                    break;
                }
                ia -= sa[d] * batch[d];
                ib -= sb[d] * batch[d];
                idx[d] = 0;
            }
        }
        return (offA, offB);
    }

    /// <summary>Runs independent batch elements in parallel once there is enough work to pay for it.</summary>
    private static void RunBatches(int count, long macsPerBatch, ParallelOptions po, Action<int> batch)
    {
        if (count > 1 && macsPerBatch * count >= 1 << 16)
        {
            WorkerPool.For(count, po, batch);
        }
        else
        {
            for (int i = 0; i < count; i++)
            {
                batch(i);
            }
        }
    }

    private static GraphTensor BatchMatMul(GraphTensor a, GraphTensor b, bool adjX, bool adjY, int[] outShape, ParallelOptions po)
    {
        int ra = a.Shape.Length, rb = b.Shape.Length;
        int m = adjX ? a.Shape[ra - 1] : a.Shape[ra - 2];
        int k = adjX ? a.Shape[ra - 2] : a.Shape[ra - 1];
        int n = adjY ? b.Shape[rb - 2] : b.Shape[rb - 1];
        var (offA, offB) = BatchOffsets(a.Shape[..(ra - 2)], b.Shape[..(rb - 2)]);
        var r = GraphTensor.Float(outShape);
        int matA = a.Shape[ra - 2] * a.Shape[ra - 1];
        int matB = b.Shape[rb - 2] * b.Shape[rb - 1];
        var af = a.F;
        var bf = b.F;
        var rf = r.F;
        bool parallel = offA.Length > 1 && (long)m * n * k * offA.Length >= 1 << 16;
        void Batch(int bi)
        {
            ReadOnlyMemory<float> A = af.AsMemory(offA[bi] * matA, matA);
            ReadOnlyMemory<float> B = bf.AsMemory(offB[bi] * matB, matB);
            if (adjX)
            {
                var tmpA = new float[m * k];
                var src = A.Span;
                for (int i = 0; i < k; i++) for (int j = 0; j < m; j++) tmpA[j * k + i] = src[i * m + j];
                A = tmpA;
            }
            if (adjY)
            {
                var tmpB = new float[k * n];
                var src = B.Span;
                for (int i = 0; i < n; i++) for (int j = 0; j < k; j++) tmpB[j * n + i] = src[i * k + j];
                B = tmpB;
            }
            // A single batch element may still be large: let the GEMM parallelize it when batches don't.
            SGemm.Multiply(A, k, B, n, rf.AsMemory(bi * m * n, m * n), n, m, n, k, 1f, parallel ? null : po);
        }
        RunBatches(offA.Length, (long)m * n * k, po, Batch);
        return r;
    }

    /// <summary>
    /// NHWC convolution with XNNPACK's accumulation order: every output starts from its bias and accumulates one
    /// fused multiply-add per (tap, input channel). Regular and grouped/multiplier convolutions run as IGEMM (taps
    /// row-major, then input channels); true depthwise convolutions (one input channel per output) run as DWCONV,
    /// whose indirection walks the taps column-major. Out-of-image taps read XNNPACK's zero buffer, which leaves
    /// the sums unchanged, so they are skipped.
    /// </summary>
    private static GraphTensor Conv(GraphTensor x, float[] w, float[] wt, int[] wShape, float[] bias, int[] outShape, int sh, int sw, int dh, int dw, bool same, bool depthwise, ParallelOptions po)
    {
        int H = x.Shape[1], W = x.Shape[2], C = x.Shape[3];
        int OH = outShape[1], OW = outShape[2], OC = outShape[3];
        int kh = wShape[1], kw = wShape[2];
        int padT = 0, padL = 0;
        if (same)
        {
            int padH = Math.Max(0, (OH - 1) * sh + (kh - 1) * dh + 1 - H);
            int padW = Math.Max(0, (OW - 1) * sw + (kw - 1) * dw + 1 - W);
            padT = padH / 2;
            padL = padW / 2;
        }
        var r = GraphTensor.Float(outShape);
        int mult = depthwise ? OC / C : 0;
        bool dwconv = depthwise && mult == 1;
        var xf = x.F;
        var rf = r.F;
        void Row(int oy)
        {
            for (int ox = 0; ox < OW; ox++)
            {
                var acc = rf.AsSpan((oy * OW + ox) * OC, OC);
                if (bias is not null)
                {
                    bias.AsSpan(0, OC).CopyTo(acc);
                }
                if (!depthwise && Vector256.IsHardwareAccelerated && OC % 8 == 0)
                {
                    ConvPixel(xf, wt, acc, oy, ox, H, W, C, OC, kh, kw, sh, sw, dh, dw, padT, padL);
                    continue;
                }
                for (int a = 0; a < (dwconv ? kw : kh); a++)
                {
                    for (int b = 0; b < (dwconv ? kh : kw); b++)
                    {
                        int ky = dwconv ? b : a, kx = dwconv ? a : b;
                        int iy = oy * sh + ky * dh - padT;
                        int ix = ox * sw + kx * dw - padL;
                        if ((uint)iy >= (uint)H || (uint)ix >= (uint)W)
                        {
                            continue;
                        }
                        var px = xf.AsSpan((iy * W + ix) * C, C);
                        if (depthwise)
                        {
                            // filter [1, kh, kw, C·mult]: out channel c·mult + m reads input channel c.
                            var f = w.AsSpan((ky * kw + kx) * OC, OC);
                            if (dwconv)
                            {
                                TensorPrimitives.FusedMultiplyAdd(px, f, acc, acc);
                            }
                            else
                            {
                                for (int c = 0; c < C; c++)
                                {
                                    TensorPrimitives.FusedMultiplyAdd(f.Slice(c * mult, mult), px[c], acc.Slice(c * mult, mult), acc.Slice(c * mult, mult));
                                }
                            }
                        }
                        else
                        {
                            int t0 = (ky * kw + kx) * C;
                            for (int c = 0; c < C; c++)
                            {
                                TensorPrimitives.FusedMultiplyAdd(wt.AsSpan((t0 + c) * OC, OC), px[c], acc, acc);
                            }
                        }
                    }
                }
            }
        }
        if ((long)OH * OW * OC * kh * kw * (depthwise ? 1 : C) >= 1 << 20)
        {
            WorkerPool.For(OH, po, Row);
        }
        else
        {
            for (int oy = 0; oy < OH; oy++)
            {
                Row(oy);
            }
        }
        return r;
    }

    /// <summary>One output pixel of a regular convolution, 8 output channels at a time in a register: the same
    /// fused multiply-adds, in the same order (taps row-major, then input channels), as the span loop.</summary>
    private static void ConvPixel(float[] x, float[] wt, Span<float> acc, int oy, int ox, int H, int W, int C, int OC, int kh, int kw, int sh, int sw, int dh, int dw, int padT, int padL)
    {
        ref float wr = ref MemoryMarshal.GetArrayDataReference(wt);
        ref float xr = ref MemoryMarshal.GetArrayDataReference(x);
        ref float ar = ref MemoryMarshal.GetReference(acc);
        for (int o0 = 0; o0 < OC; o0 += 8)
        {
            var v = Vector256.LoadUnsafe(ref ar, (nuint)o0);
            for (int ky = 0; ky < kh; ky++)
            {
                int iy = oy * sh + ky * dh - padT;
                if ((uint)iy >= (uint)H)
                {
                    continue;
                }
                for (int kx = 0; kx < kw; kx++)
                {
                    int ix = ox * sw + kx * dw - padL;
                    if ((uint)ix >= (uint)W)
                    {
                        continue;
                    }
                    int px = (iy * W + ix) * C;
                    int t0 = (ky * kw + kx) * C;
                    for (int c = 0; c < C; c++)
                    {
                        v = Vector256.FusedMultiplyAdd(Vector256.LoadUnsafe(ref wr, (nuint)((t0 + c) * OC + o0)), Vector256.Create(Unsafe.Add(ref xr, px + c)), v);
                    }
                }
            }
            v.StoreUnsafe(ref ar, (nuint)o0);
        }
    }
}
