using System.Buffers.Binary;
using System.Runtime.InteropServices;

namespace SentenceTransformers.EmbeddingGemma2.LiteRt;

/// <summary>TFLite tensor element types (subset of <c>tflite::TensorType</c>).</summary>
internal enum TfLiteType : sbyte
{
    Float32 = 0,
    Float16 = 1,
    Int32 = 2,
    UInt8 = 3,
    Int64 = 4,
    String = 5,
    Bool = 6,
    Int16 = 7,
    Int8 = 9,
    Int4 = 17,
    BFloat16 = 18,
    Int2 = 19,
}

/// <summary>TFLite builtin operator codes used by the EmbeddingGemma 2 graphs.</summary>
internal static class TfLiteOp
{
    public const int Add = 0;
    public const int AveragePool2D = 1;
    public const int Concatenation = 2;
    public const int Conv2D = 3;
    public const int DepthwiseConv2D = 4;
    public const int Dequantize = 6;
    public const int EmbeddingLookup = 7;
    public const int FullyConnected = 9;
    public const int Logistic = 14;
    public const int Mul = 18;
    public const int Reshape = 22;
    public const int Softmax = 25;
    public const int Tanh = 28;
    public const int Tile = 69;
    public const int Equal = 71;
    public const int NotEqual = 72;
    public const int LogicalOr = 84;
    public const int Quantize = 114;
    public const int Pad = 34;
    public const int Gather = 36;
    public const int Transpose = 39;
    public const int Mean = 40;
    public const int Sub = 41;
    public const int Div = 42;
    public const int StridedSlice = 45;
    public const int Exp = 47;
    public const int Cast = 53;
    public const int Maximum = 55;
    public const int Minimum = 57;
    public const int Less = 58;
    public const int Greater = 61;
    public const int GreaterEqual = 62;
    public const int Slice = 65;
    public const int Sin = 66;
    public const int Log = 73;
    public const int Sum = 74;
    public const int Sqrt = 75;
    public const int Rsqrt = 76;
    public const int Pack = 83;
    public const int LogicalAnd = 86;
    public const int LogicalNot = 87;
    public const int Unpack = 88;
    public const int Square = 92;
    public const int Abs = 101;
    public const int Cos = 108;
    public const int SelectV2 = 123;
    public const int BatchMatMul = 126;
    public const int Gelu = 150;
    public const int StableHloComposite = 206;
}

/// <summary>Per-tensor or per-channel affine quantization parameters.</summary>
internal sealed record TfLiteQuantization(float[] Scale, long[] ZeroPoint, int QuantizedDimension);

internal sealed class TfLiteTensor
{
    public int Index { get; init; }
    public string Name { get; init; }
    public TfLiteType Type { get; init; }
    public int[] Shape { get; init; }
    public int Buffer { get; init; }
    public uint ExternalBuffer { get; init; }
    public TfLiteQuantization Quantization { get; init; }

    public long ElementCount
    {
        get
        {
            long n = 1;
            foreach (var d in Shape)
            {
                n *= d;
            }
            return n;
        }
    }

    public override string ToString() => $"t{Index}:{Type}[{string.Join(",", Shape)}] {Name}";
}

internal sealed class TfLiteOperator
{
    public int Index { get; init; }
    public int Opcode { get; init; }
    public int[] Inputs { get; init; }
    public int[] Outputs { get; init; }
    public FbTable BuiltinOptions { get; init; }
    /// <summary>For <c>STABLEHLO_COMPOSITE</c>: the composite name (e.g. <c>odml.rms_norm</c>).</summary>
    public string CompositeName { get; init; }
    /// <summary>For <c>STABLEHLO_COMPOSITE</c>: the decomposition subgraph index.</summary>
    public int DecompositionSubgraph { get; init; } = -1;
    /// <summary>For <c>STABLEHLO_COMPOSITE</c>: raw flexbuffer attributes.</summary>
    public byte[] CompositeAttributes { get; init; }

    public bool IsComposite(string name) => Opcode == TfLiteOp.StableHloComposite && CompositeName == name;

    public override string ToString() => $"op{Index}:{(CompositeName ?? Opcode.ToString())}({string.Join(",", Inputs)})->({string.Join(",", Outputs)})";
}

internal sealed class TfLiteSubgraph
{
    public int Index { get; init; }
    public string Name { get; init; }
    public TfLiteTensor[] Tensors { get; init; }
    public TfLiteOperator[] Operators { get; init; }
    public int[] Inputs { get; init; }
    public int[] Outputs { get; init; }
}

internal sealed record TfLiteSignature(string Key, int SubgraphIndex, IReadOnlyDictionary<string, int> Inputs, IReadOnlyDictionary<string, int> Outputs);

/// <summary>
/// Read-only view of a TFLite flatbuffer model plus its external weight blob (LiteRT-LM stores the large
/// constant tensors in a separate <c>TFLiteWeights</c> section that tensors reference by
/// <c>external_buffer</c> id). Only the pieces the EmbeddingGemma 2 port needs are decoded: the op list
/// of each subgraph (to locate weights by graph structure), tensor metadata and constant data.
/// </summary>
internal sealed class TfLiteModel
{
    // Model table fields.
    private const int ModelOperatorCodes = 6, ModelSubgraphs = 8, ModelBuffers = 12, ModelSignatureDefs = 18, ModelExternalBuffers = 22;

    private readonly byte[] _graph;
    private readonly byte[] _weights;
    private readonly FbTable _root;
    private readonly int[] _opcodes;
    private readonly Dictionary<uint, (long Offset, long Length)> _external = new();
    private readonly TfLiteSubgraph[] _subgraphs;

    public IReadOnlyList<TfLiteSignature> Signatures { get; }

    public TfLiteModel(byte[] graph, byte[] weights)
    {
        _graph = graph ?? throw new ArgumentNullException(nameof(graph));
        _weights = weights;
        _root = FbTable.Root(graph);

        int nOpcodes = _root.VectorLength(ModelOperatorCodes);
        _opcodes = new int[nOpcodes];
        for (int i = 0; i < nOpcodes; i++)
        {
            var oc = _root.GetTableElement(ModelOperatorCodes, i);
            // builtin_code (int32, slot 10) supersedes deprecated_builtin_code (int8, slot 4).
            _opcodes[i] = Math.Max(oc.GetSByte(4), oc.GetInt32(10));
        }

        int nExt = _root.VectorLength(ModelExternalBuffers);
        for (int i = 0; i < nExt; i++)
        {
            var e = _root.GetTableElement(ModelExternalBuffers, i);
            _external[e.GetUInt32(4)] = ((long)e.GetUInt64(8), (long)e.GetUInt64(10));
        }

        _subgraphs = new TfLiteSubgraph[_root.VectorLength(ModelSubgraphs)];

        var sigs = new List<TfLiteSignature>();
        int nSig = _root.VectorLength(ModelSignatureDefs);
        for (int i = 0; i < nSig; i++)
        {
            var sd = _root.GetTableElement(ModelSignatureDefs, i);
            sigs.Add(new TfLiteSignature(sd.GetString(8), (int)sd.GetUInt32(12), ReadTensorMap(sd, 4), ReadTensorMap(sd, 6)));
        }
        Signatures = sigs;
    }

    private static Dictionary<string, int> ReadTensorMap(FbTable sd, int field)
    {
        var map = new Dictionary<string, int>(StringComparer.Ordinal);
        int n = sd.VectorLength(field);
        for (int i = 0; i < n; i++)
        {
            var tm = sd.GetTableElement(field, i);
            map[tm.GetString(4)] = (int)tm.GetUInt32(6);
        }
        return map;
    }

    public int SubgraphCount => _subgraphs.Length;

    public TfLiteSignature Signature(string key)
        => Signatures.FirstOrDefault(s => s.Key == key) ?? throw new InvalidDataException($"Signature '{key}' not found. Available: {string.Join(", ", Signatures.Select(s => s.Key))}");

    public TfLiteSubgraph Subgraph(int index)
    {
        var sg = _subgraphs[index];
        if (sg is not null)
        {
            return sg;
        }
        sg = ParseSubgraph(index);
        _subgraphs[index] = sg;
        return sg;
    }

    private TfLiteSubgraph ParseSubgraph(int index)
    {
        var t = _root.GetTableElement(ModelSubgraphs, index);
        int nt = t.VectorLength(4);
        var tensors = new TfLiteTensor[nt];
        for (int i = 0; i < nt; i++)
        {
            var tt = t.GetTableElement(4, i);
            TfLiteQuantization q = null;
            var qt = tt.GetTable(12);
            if (!qt.IsNull && qt.VectorLength(8) > 0)
            {
                q = new TfLiteQuantization(qt.GetFloatVector(8), qt.GetInt64Vector(10), qt.GetInt32(16));
            }
            tensors[i] = new TfLiteTensor
            {
                Index = i,
                Shape = tt.GetInt32Vector(4),
                Type = (TfLiteType)tt.GetSByte(6),
                Buffer = (int)tt.GetUInt32(8),
                Name = tt.GetString(10),
                Quantization = q,
                ExternalBuffer = tt.GetUInt32(24),
            };
        }

        int no = t.VectorLength(10);
        var ops = new TfLiteOperator[no];
        for (int i = 0; i < no; i++)
        {
            var o = t.GetTableElement(10, i);
            int code = _opcodes[o.GetUInt32(4)];
            string compName = null;
            int decomp = -1;
            byte[] attrs = null;
            if (code == TfLiteOp.StableHloComposite)
            {
                var opt = o.GetTable(28);
                compName = opt.GetString(4);
                decomp = opt.GetInt32(6);
                attrs = opt.GetBytes(8).ToArray();
            }
            ops[i] = new TfLiteOperator
            {
                Index = i,
                Opcode = code,
                Inputs = o.GetInt32Vector(6),
                Outputs = o.GetInt32Vector(8),
                BuiltinOptions = o.GetTable(12),
                CompositeName = compName,
                DecompositionSubgraph = decomp,
                CompositeAttributes = attrs,
            };
        }

        return new TfLiteSubgraph
        {
            Index = index,
            Name = t.GetString(12),
            Tensors = tensors,
            Operators = ops,
            Inputs = t.GetInt32Vector(6),
            Outputs = t.GetInt32Vector(8),
        };
    }

    /// <summary>Raw bytes of a constant tensor (inline buffer or external weight section), or empty for activations.</summary>
    public ReadOnlySpan<byte> RawData(TfLiteTensor tensor)
    {
        if (tensor.ExternalBuffer != 0)
        {
            if (_weights is null)
            {
                throw new InvalidDataException($"Tensor {tensor} references an external weight buffer but no weight section was loaded.");
            }
            if (!_external.TryGetValue(tensor.ExternalBuffer, out var e))
            {
                throw new InvalidDataException($"Tensor {tensor} references unknown external buffer {tensor.ExternalBuffer}.");
            }
            if (e.Offset < 0 || e.Offset + e.Length > _weights.Length)
            {
                throw new InvalidDataException($"External buffer of {tensor} is out of range (truncated weight section?).");
            }
            return _weights.AsSpan(checked((int)e.Offset), checked((int)e.Length));
        }
        if (tensor.Buffer <= 0 || tensor.Buffer >= _root.VectorLength(ModelBuffers))
        {
            return ReadOnlySpan<byte>.Empty;
        }
        var buf = _root.GetTableElement(ModelBuffers, tensor.Buffer);
        var (offset, length) = buf.GetBytesRange(4);
        if (length > 0)
        {
            return _graph.AsSpan(offset, length);
        }
        // Buffers beyond 2 GB use (offset, size) relative to the start of the model flatbuffer.
        long o = (long)buf.GetUInt64(6), s = (long)buf.GetUInt64(8);
        if (o > 1 && s > 0)
        {
            return _graph.AsSpan(checked((int)o), checked((int)s));
        }
        return ReadOnlySpan<byte>.Empty;
    }

    public bool IsConstant(TfLiteTensor tensor) => !RawData(tensor).IsEmpty;

    /// <summary>Reads a constant float tensor (FLOAT32, FLOAT16 or a dequantized INT8/INT4/INT2 tensor).</summary>
    public float[] ReadFloats(TfLiteTensor tensor)
    {
        var raw = RawData(tensor);
        if (raw.IsEmpty)
        {
            throw new InvalidDataException($"Tensor {tensor} is not a constant.");
        }
        long count = tensor.ElementCount;
        switch (tensor.Type)
        {
            case TfLiteType.Float32:
                return MemoryMarshal.Cast<byte, float>(raw).Slice(0, checked((int)count)).ToArray();
            case TfLiteType.Float16:
            {
                var halves = MemoryMarshal.Cast<byte, Half>(raw);
                var r = new float[count];
                for (int i = 0; i < r.Length; i++)
                {
                    r[i] = (float)halves[i];
                }
                return r;
            }
            case TfLiteType.Int8:
            case TfLiteType.Int4:
            case TfLiteType.Int2:
            {
                var q = ReadQuantizedValues(tensor);
                var r = new float[count];
                Dequantize(tensor, q, r);
                return r;
            }
            default:
                throw new NotSupportedException($"Cannot read {tensor} as float.");
        }
    }

    public int[] ReadInts(TfLiteTensor tensor)
    {
        var raw = RawData(tensor);
        if (tensor.Type == TfLiteType.Int32)
        {
            return MemoryMarshal.Cast<byte, int>(raw).Slice(0, checked((int)tensor.ElementCount)).ToArray();
        }
        if (tensor.Type == TfLiteType.Int64)
        {
            var l = MemoryMarshal.Cast<byte, long>(raw);
            var r = new int[tensor.ElementCount];
            for (int i = 0; i < r.Length; i++)
            {
                r[i] = checked((int)l[i]);
            }
            return r;
        }
        throw new NotSupportedException($"Cannot read {tensor} as int.");
    }

    /// <summary>Returns the raw signed integer values of an INT8/INT4/INT2 tensor, unpacked to one sbyte per element
    /// (TFLite packs sub-byte types little-end first: element 2i in the low nibble of byte i).</summary>
    public sbyte[] ReadQuantizedValues(TfLiteTensor tensor)
    {
        var raw = RawData(tensor);
        long count = tensor.ElementCount;
        var r = GC.AllocateUninitializedArray<sbyte>(checked((int)count));
        switch (tensor.Type)
        {
            case TfLiteType.Int8:
                MemoryMarshal.Cast<byte, sbyte>(raw).Slice(0, r.Length).CopyTo(r);
                break;
            case TfLiteType.Int4:
                UnpackInt4(raw, r);
                break;
            case TfLiteType.Int2:
                for (int i = 0; i < r.Length; i++)
                {
                    int v = (raw[i >> 2] >> ((i & 3) * 2)) & 3;
                    r[i] = (sbyte)(v >= 2 ? v - 4 : v);
                }
                break;
            default:
                throw new NotSupportedException($"{tensor} is not a quantized integer tensor.");
        }
        return r;
    }

    internal static void UnpackInt4(ReadOnlySpan<byte> packed, Span<sbyte> dst)
    {
        int n = dst.Length;
        int pairs = n >> 1;
        for (int i = 0; i < pairs; i++)
        {
            byte b = packed[i];
            dst[2 * i] = (sbyte)((sbyte)(b << 4) >> 4);
            dst[2 * i + 1] = (sbyte)((sbyte)b >> 4);
        }
        if ((n & 1) != 0)
        {
            byte b = packed[pairs];
            dst[n - 1] = (sbyte)((sbyte)(b << 4) >> 4);
        }
    }

    /// <summary>Dequantizes integer values with the tensor's (per-tensor or per-channel) scale and zero point.</summary>
    public static void Dequantize(TfLiteTensor tensor, ReadOnlySpan<sbyte> q, Span<float> dst)
    {
        var quant = tensor.Quantization ?? throw new InvalidDataException($"{tensor} has no quantization parameters.");
        var shape = tensor.Shape;
        int axis = quant.QuantizedDimension;
        int channels = quant.Scale.Length;
        // Elements of one "slice" along the quantized axis repeat with an inner stride.
        long inner = 1;
        for (int d = axis + 1; d < shape.Length; d++)
        {
            inner *= shape[d];
        }
        for (int i = 0; i < dst.Length; i++)
        {
            int c = channels == 1 ? 0 : (int)((i / inner) % channels);
            long zp = quant.ZeroPoint.Length > c ? quant.ZeroPoint[c] : 0;
            dst[i] = (q[i] - zp) * quant.Scale[c];
        }
    }

    /// <summary>Reads a scalar float constant (any rank, single element).</summary>
    public float ReadScalar(TfLiteTensor tensor)
    {
        var v = ReadFloats(tensor);
        if (v.Length != 1)
        {
            throw new InvalidDataException($"{tensor} is not a scalar.");
        }
        return v[0];
    }

    /// <summary>Reads a TFLite boolean builtin option (e.g. <c>GeluOptions.approximate</c>).</summary>
    public static bool ReadBoolOption(TfLiteOperator op, int field, bool def = false)
        => op.BuiltinOptions.IsNull ? def : op.BuiltinOptions.GetBool(field, def);
}
