using System.Numerics.Tensors;
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

    public static GraphTensor Float(int[] shape, float[] data = null) => new() { Shape = shape, F = data ?? new float[Count(shape)] };
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
        for (int i = 0; i < _ops.Length; i++)
        {
            _ops[i](values, po);
            opHook?.Invoke(i, values);
        }
        var result = new Dictionary<string, GraphTensor>(StringComparer.Ordinal);
        foreach (var (name, index) in Signature.Outputs)
        {
            result[name] = values[index];
        }
        return result;
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
        return op.Opcode switch
        {
            TfLiteOp.Add or TfLiteOp.Sub or TfLiteOp.Mul or TfLiteOp.Div or TfLiteOp.FullyConnected => WithActivation(op, 4, kernel),
            TfLiteOp.Concatenation => WithActivation(op, 6, kernel),
            _ => kernel,
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
            case TfLiteOp.Add: return Binary(ins, o, (a, b) => a + b);
            case TfLiteOp.Sub: return Binary(ins, o, (a, b) => a - b);
            case TfLiteOp.Mul: return Binary(ins, o, (a, b) => a * b);
            case TfLiteOp.Div: return Binary(ins, o, (a, b) => a / b);
            case TfLiteOp.Maximum: return Binary(ins, o, MathF.Max);
            case TfLiteOp.Minimum: return Binary(ins, o, MathF.Min);
            case TfLiteOp.Rsqrt: return Unary(ins, o, x => 1f / MathF.Sqrt(x));
            case TfLiteOp.Sqrt: return Unary(ins, o, MathF.Sqrt);
            case TfLiteOp.Logistic: return (v, po) => { var r = GraphTensor.Float(v[ins[0]].Shape); TensorPrimitives.Sigmoid(v[ins[0]].F.AsSpan(0, r.F.Length), r.F); v[o] = r; };
            case TfLiteOp.Tanh: return (v, po) => { var r = GraphTensor.Float(v[ins[0]].Shape); TensorPrimitives.Tanh(v[ins[0]].F.AsSpan(0, r.F.Length), r.F); v[o] = r; };
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
                return (v, po) => v[o] = Reduce(v[ins[0]], v[ins[1]].I, outShape, mean);
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

    private float RmsNormEpsilon(TfLiteOperator op)
    {
        // The decomposition adds epsilon to mean(x²) before RSQRT; read it rather than assuming 1e-6.
        var dg = _model.Subgraph(op.DecompositionSubgraph);
        foreach (var d in dg.Operators)
        {
            if (d.Opcode == TfLiteOp.Add)
            {
                foreach (var i in d.Inputs)
                {
                    var t = dg.Tensors[i];
                    if (t.Type == TfLiteType.Float32 && t.ElementCount == 1 && _model.IsConstant(t))
                    {
                        return _model.ReadScalar(t);
                    }
                }
            }
        }
        return 1e-6f;
    }

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
                SGemm.Multiply(xin.F, inF, wT, outF, r.F, outF, rows, outF, inF, 1f, po);
                if (bias is not null)
                {
                    for (int i = 0; i < rows; i++)
                    {
                        TensorPrimitives.Add(r.F.AsSpan(i * outF, outF), bias, r.F.AsSpan(i * outF, outF));
                    }
                }
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
        return (v, po) => v[o] = Conv(v[x], w, wShape, bias, outShape, sh, sw, dh, dw, padding == 0 /* SAME */, depthwise);
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
        var r = GraphTensor.Float(x.Shape);
        float inv = 1f / scale;
        for (int i = 0; i < r.F.Length; i++)
        {
            float q = Math.Clamp(MathF.Round(x.F[i] * inv + zp, MidpointRounding.ToEven), -128f, 127f);
            r.F[i] = (q - zp) * scale;
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

    private static Action<GraphTensor[], ParallelOptions> Binary(int[] ins, int o, Func<float, float, float> f)
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
            int na = a.Length, nb = b.Length;
            if (SameShape(a.Shape, b.Shape))
            {
                for (int i = 0; i < r.F.Length; i++) r.F[i] = f(a.F[i], b.F[i]);
            }
            else if (nb == 1)
            {
                float s = b.F[0];
                for (int i = 0; i < r.F.Length; i++) r.F[i] = f(a.F[i], s);
            }
            else if (na == 1)
            {
                float s = a.F[0];
                for (int i = 0; i < r.F.Length; i++) r.F[i] = f(s, b.F[i]);
            }
            else
            {
                ForEachBroadcast(shape, BroadcastStrides(a.Shape, shape), BroadcastStrides(b.Shape, shape),
                    (i, ia, ib) => r.F[i] = f(a.F[ia], b.F[ib]));
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

    private static GraphTensor Reduce(GraphTensor x, int[] axes, int[] outShape, bool mean)
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
            for (int i = 0; i < rows; i++)
            {
                float s = TensorPrimitives.Sum(x.F.AsSpan(i * last, last));
                r1.F[i] = mean ? s / last : s;
            }
            return r1;
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
        // Constant padding with 0 (the zero point of every quantized tensor in these graphs).
        var r = NewLike(x, outShape);
        int rank = x.Shape.Length;
        var so = RowMajorStrides(outShape);
        var idx = new int[rank];
        for (int i = 0; i < x.Length; i++)
        {
            int to = 0;
            for (int d = 0; d < rank; d++)
            {
                to += (idx[d] + paddings[2 * d]) * so[d];
            }
            CopyElement(x, i, r, to);
            for (int d = rank - 1; d >= 0; d--)
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
        var idx = new int[rank];
        int from = 0;
        int n = r.Length;
        for (int i = 0; i < n; i++)
        {
            CopyElement(x, from, r, i);
            for (int d = rank - 1; d >= 0; d--)
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
        return r;
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

    private static GraphTensor BatchMatMul(GraphTensor a, GraphTensor b, bool adjX, bool adjY, int[] outShape, ParallelOptions po)
    {
        int ra = a.Shape.Length, rb = b.Shape.Length;
        int m = adjX ? a.Shape[ra - 1] : a.Shape[ra - 2];
        int k = adjX ? a.Shape[ra - 2] : a.Shape[ra - 1];
        int n = adjY ? b.Shape[rb - 2] : b.Shape[rb - 1];
        var batchA = a.Shape[..(ra - 2)];
        var batchB = b.Shape[..(rb - 2)];
        var batch = BroadcastShape(batchA, batchB);
        int nb = GraphTensor.Count(batch);
        var r = GraphTensor.Float(outShape);
        var sa = BroadcastStrides(batchA, batch);
        var sbb = BroadcastStrides(batchB, batch);
        int matA = a.Shape[ra - 2] * a.Shape[ra - 1];
        int matB = b.Shape[rb - 2] * b.Shape[rb - 1];
        var tmpA = adjX ? new float[m * k] : null;
        var tmpB = adjY ? new float[k * n] : null;
        var idx = new int[batch.Length];
        int ia = 0, ib = 0;
        for (int bi = 0; bi < nb; bi++)
        {
            ReadOnlySpan<float> A = a.F.AsSpan(ia * matA, matA);
            ReadOnlySpan<float> B = b.F.AsSpan(ib * matB, matB);
            if (adjX)
            {
                for (int i = 0; i < k; i++) for (int j = 0; j < m; j++) tmpA[j * k + i] = A[i * m + j];
                A = tmpA;
            }
            if (adjY)
            {
                for (int i = 0; i < n; i++) for (int j = 0; j < k; j++) tmpB[j * n + i] = B[i * k + j];
                B = tmpB;
            }
            SGemm.Multiply(A, k, B, n, r.F.AsSpan(bi * m * n, m * n), n, m, n, k);
            for (int d = batch.Length - 1; d >= 0; d--)
            {
                idx[d]++;
                ia += sa[d];
                ib += sbb[d];
                if (idx[d] < batch[d])
                {
                    break;
                }
                ia -= sa[d] * batch[d];
                ib -= sbb[d] * batch[d];
                idx[d] = 0;
            }
        }
        return r;
    }

    private static GraphTensor Conv(GraphTensor x, float[] w, int[] wShape, float[] bias, int[] outShape, int sh, int sw, int dh, int dw, bool same, bool depthwise)
    {
        // NHWC input, batch 1.
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
        for (int oy = 0; oy < OH; oy++)
        {
            for (int ox = 0; ox < OW; ox++)
            {
                var acc = r.F.AsSpan((oy * OW + ox) * OC, OC);
                if (bias is not null)
                {
                    bias.AsSpan(0, OC).CopyTo(acc);
                }
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
                        var px = x.F.AsSpan((iy * W + ix) * C, C);
                        if (depthwise)
                        {
                            // filter [1, kh, kw, C·mult]: out channel c·mult + m reads input channel c.
                            var f = w.AsSpan((ky * kw + kx) * OC, OC);
                            if (mult == 1)
                            {
                                TensorPrimitives.MultiplyAdd(px, f, acc, acc);
                            }
                            else
                            {
                                for (int c = 0; c < C; c++)
                                {
                                    TensorPrimitives.MultiplyAdd(f.Slice(c * mult, mult), px[c], acc.Slice(c * mult, mult), acc.Slice(c * mult, mult));
                                }
                            }
                        }
                        else
                        {
                            // filter [OC, kh, kw, C]
                            for (int oc = 0; oc < OC; oc++)
                            {
                                acc[oc] += TensorPrimitives.Dot(px, w.AsSpan(((oc * kh + ky) * kw + kx) * C, C));
                            }
                        }
                    }
                }
            }
        }
        return r;
    }
}
