using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>Weight storage / compute precision for the transformer's linear layers.</summary>
public enum Quantization
{
    /// <summary>Full float32 weights (largest, exact parity with the reference). ~400 MB for the
    /// transformer layers.</summary>
    None = 0,

    /// <summary>8-bit per-output-channel symmetric quantization (~4x smaller than fp32, ~100 MB).
    /// Quality is essentially indistinguishable from fp32.</summary>
    Int8 = 1,

    /// <summary>4-bit group-wise symmetric quantization (~8x smaller than fp32, ~50 MB) with a group
    /// size of 32 along the input dimension. Smallest footprint, with a small accuracy trade-off.</summary>
    Int4 = 2,
}

/// <summary>
/// A linear projection weight matrix (PyTorch <c>[outDim, inDim]</c> layout) that can multiply a batch
/// of activations: <c>y[s, o] = sum_i x[s, i] * W[o, i]</c>. Implementations differ only in how the
/// weights are stored; all dequantize to float and use the SIMD <see cref="TensorPrimitives.Dot"/> so
/// the activation path stays in float32 (weight-only quantization).
/// </summary>
internal interface IWeightMatrix
{
    int InDim { get; }
    int OutDim { get; }
    void Multiply(float[] x, float[] y, int seq);

    static IWeightMatrix Create(float[] weights, int outDim, int inDim, Quantization quantization) => quantization switch
    {
        Quantization.None => new FloatMatrix(weights, outDim, inDim),
        Quantization.Int8 => Int8Matrix.Quantize(weights, outDim, inDim),
        Quantization.Int4 => Int4Matrix.Quantize(weights, outDim, inDim),
        _ => throw new ArgumentOutOfRangeException(nameof(quantization)),
    };
}

/// <summary>Shared helpers for the VNNI int8 GEMM paths.</summary>
internal static class VnniActivations
{
    /// <summary>Dynamic per-row symmetric int8 activation quantization, offset by +128 into the uint8
    /// operand range that <c>vpdpbusd</c> expects. Returns pooled buffers the caller must return.</summary>
    public static (byte[] Ua, float[] Scale) Quantize(float[] x, int seq, int inDim)
    {
        byte[] ua = ArrayPool<byte>.Shared.Rent(seq * inDim);
        float[] scale = ArrayPool<float>.Shared.Rent(seq);
        Parallel.For(0, seq, s =>
        {
            int b = s * inDim;
            float amax = 0f;
            for (int i = 0; i < inDim; i++)
            {
                amax = MathF.Max(amax, MathF.Abs(x[b + i]));
            }
            float sc = amax > 0 ? amax / 127f : 1f;
            float inv = 1f / sc;
            scale[s] = sc;
            for (int i = 0; i < inDim; i++)
            {
                int q = Math.Clamp((int)MathF.Round(x[b + i] * inv), -127, 127);
                ua[b + i] = (byte)(q + 128);
            }
        });
        return (ua, scale);
    }

    public static void Return(byte[] ua, float[] scale)
    {
        ArrayPool<byte>.Shared.Return(ua);
        ArrayPool<float>.Shared.Return(scale);
    }
}

/// <summary>Plain float32 weights. Delegates to <see cref="Ops.Linear"/>.</summary>
internal sealed class FloatMatrix(float[] weights, int outDim, int inDim) : IWeightMatrix
{
    private readonly float[] _w = weights;
    public int InDim => inDim;
    public int OutDim => outDim;

    public void Multiply(float[] x, float[] y, int seq) => Numerics.Ops.Linear(x, _w, y, seq, inDim, outDim);
}

/// <summary>
/// 8-bit symmetric quantization with one scale per output channel:
/// <c>W[o, i] ≈ scale[o] * q[o, i]</c>, with <c>q ∈ [-127, 127]</c>.
///
/// When AVX-VNNI is available the matmul runs as a true int8 GEMM (W8A8): activations are quantized
/// per row to uint8 and the dot products use <c>vpdpbusd</c> (32 int8 MACs accumulated to int32 per
/// instruction), which is several times faster than the float path. On hardware without VNNI it falls
/// back to dequantizing each weight row to float and using the SIMD float dot product (weight-only,
/// activations kept in float32).
/// </summary>
internal sealed class Int8Matrix : IWeightMatrix
{
    private readonly sbyte[] _q;
    private readonly float[] _scale;
    private readonly int[] _rowSum; // sum of quantized weights per output channel (for the uint8 offset correction)
    public int InDim { get; }
    public int OutDim { get; }

    private static bool UseVnni(int inDim) => AvxVnni.IsSupported && (inDim % 32 == 0);

    private Int8Matrix(sbyte[] q, float[] scale, int[] rowSum, int outDim, int inDim)
    {
        _q = q;
        _scale = scale;
        _rowSum = rowSum;
        InDim = inDim;
        OutDim = outDim;
    }

    public static Int8Matrix Quantize(float[] w, int outDim, int inDim)
    {
        var q = new sbyte[(long)outDim * inDim <= int.MaxValue ? outDim * inDim : throw new OverflowException()];
        var scale = new float[outDim];
        var rowSum = new int[outDim];
        Parallel.For(0, outDim, o =>
        {
            int baseIdx = o * inDim;
            float amax = 0f;
            for (int i = 0; i < inDim; i++)
            {
                amax = MathF.Max(amax, MathF.Abs(w[baseIdx + i]));
            }
            float s = amax > 0 ? amax / 127f : 1f;
            float inv = 1f / s;
            scale[o] = s;
            int sum = 0;
            for (int i = 0; i < inDim; i++)
            {
                int v = Math.Clamp((int)MathF.Round(w[baseIdx + i] * inv), -127, 127);
                q[baseIdx + i] = (sbyte)v;
                sum += v;
            }
            rowSum[o] = sum;
        });
        return new Int8Matrix(q, scale, rowSum, outDim, inDim);
    }

    public void Multiply(float[] x, float[] y, int seq)
    {
        if (UseVnni(InDim))
        {
            MultiplyVnni(x, y, seq);
        }
        else
        {
            MultiplyFloat(x, y, seq);
        }
    }

    /// <summary>True int8 GEMM (W8A8) via VNNI <c>vpdpbusd</c>. Activations are quantized per row to
    /// signed int8, offset by +128 to feed the unsigned operand of <c>vpdpbusd</c>; the offset is
    /// corrected with the precomputed per-channel weight sum.</summary>
    private void MultiplyVnni(float[] x, float[] y, int seq)
    {
        int inDim = InDim, outDim = OutDim;
        var (ua, aScale) = VnniActivations.Quantize(x, seq, inDim);
        try
        {
            Parallel.For(0, outDim, o =>
            {
                int wBase = o * inDim;
                float wsc = _scale[o];
                int offset = 128 * _rowSum[o];
                int s = 0;
                // Block 4 sequence positions per weight-row load: each weight vector is reused across
                // 4 activation rows (4x less weight traffic) and the 4 independent accumulators hide
                // the vpdpbusd latency.
                for (; s + 4 <= seq; s += 4)
                {
                    int u0 = s * inDim, u1 = u0 + inDim, u2 = u1 + inDim, u3 = u2 + inDim;
                    var a0 = Vector256<int>.Zero;
                    var a1 = Vector256<int>.Zero;
                    var a2 = Vector256<int>.Zero;
                    var a3 = Vector256<int>.Zero;
                    for (int i = 0; i < inDim; i += 32)
                    {
                        var wv = Vector256.LoadUnsafe(ref _q[wBase + i]);
                        a0 = AvxVnni.MultiplyWideningAndAdd(a0, Vector256.LoadUnsafe(ref ua[u0 + i]), wv);
                        a1 = AvxVnni.MultiplyWideningAndAdd(a1, Vector256.LoadUnsafe(ref ua[u1 + i]), wv);
                        a2 = AvxVnni.MultiplyWideningAndAdd(a2, Vector256.LoadUnsafe(ref ua[u2 + i]), wv);
                        a3 = AvxVnni.MultiplyWideningAndAdd(a3, Vector256.LoadUnsafe(ref ua[u3 + i]), wv);
                    }
                    y[(s + 0) * outDim + o] = aScale[s + 0] * wsc * (Vector256.Sum(a0) - offset);
                    y[(s + 1) * outDim + o] = aScale[s + 1] * wsc * (Vector256.Sum(a1) - offset);
                    y[(s + 2) * outDim + o] = aScale[s + 2] * wsc * (Vector256.Sum(a2) - offset);
                    y[(s + 3) * outDim + o] = aScale[s + 3] * wsc * (Vector256.Sum(a3) - offset);
                }
                for (; s < seq; s++)
                {
                    int uBase = s * inDim;
                    var acc = Vector256<int>.Zero;
                    for (int i = 0; i < inDim; i += 32)
                    {
                        acc = AvxVnni.MultiplyWideningAndAdd(acc, Vector256.LoadUnsafe(ref ua[uBase + i]), Vector256.LoadUnsafe(ref _q[wBase + i]));
                    }
                    y[s * outDim + o] = aScale[s] * wsc * (Vector256.Sum(acc) - offset);
                }
            });
        }
        finally
        {
            VnniActivations.Return(ua, aScale);
        }
    }

    /// <summary>Portable fallback: dequantize each weight row to float and dot against float activations.</summary>
    private void MultiplyFloat(float[] x, float[] y, int seq)
    {
        int inDim = InDim, outDim = OutDim;
        Parallel.For(0, outDim, o =>
        {
            Span<float> buf = stackalloc float[inDim];
            int baseIdx = o * inDim;
            float s = _scale[o];
            for (int i = 0; i < inDim; i++)
            {
                buf[i] = _q[baseIdx + i] * s;
            }
            for (int sIdx = 0; sIdx < seq; sIdx++)
            {
                y[sIdx * outDim + o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(x, sIdx * inDim, inDim), buf);
            }
        });
    }
}

/// <summary>
/// 4-bit group-wise symmetric quantization. The input dimension is split into groups of
/// <see cref="GroupSize"/>; each (output channel, group) pair has its own scale, and weights are
/// packed two nibbles per byte.
///
/// With AVX-VNNI the group size (32) maps onto exactly one <c>vpdpbusd</c>, so the matmul runs as an
/// int8 GEMM: each output row's nibbles are unpacked once per matmul, then dotted (per group, with a
/// per-group scale) against the dynamically int8-quantized activations. Otherwise it dequantizes each
/// row to float and uses the SIMD float dot product.
/// </summary>
internal sealed class Int4Matrix : IWeightMatrix
{
    public const int GroupSize = 32;

    private readonly byte[] _packed;   // two 4-bit weights per byte
    private readonly float[] _scale;   // [outDim * numGroups]
    private readonly int[] _groupSum;  // [outDim * numGroups] sum of signed int4 weights per group
    private readonly int _numGroups;
    public int InDim { get; }
    public int OutDim { get; }

    private static bool UseVnni() => AvxVnni.IsSupported && GroupSize == 32;

    private Int4Matrix(byte[] packed, float[] scale, int[] groupSum, int numGroups, int outDim, int inDim)
    {
        _packed = packed;
        _scale = scale;
        _groupSum = groupSum;
        _numGroups = numGroups;
        InDim = inDim;
        OutDim = outDim;
    }

    public static Int4Matrix Quantize(float[] w, int outDim, int inDim)
    {
        if (inDim % GroupSize != 0)
        {
            throw new NotSupportedException($"Int4 requires inDim ({inDim}) to be a multiple of the group size {GroupSize}.");
        }
        int numGroups = inDim / GroupSize;
        var packed = new byte[outDim * inDim / 2];
        var scale = new float[outDim * numGroups];
        var groupSum = new int[outDim * numGroups];

        Parallel.For(0, outDim, o =>
        {
            int rowBase = o * inDim;
            for (int g = 0; g < numGroups; g++)
            {
                int gStart = g * GroupSize;
                float amax = 0f;
                for (int i = 0; i < GroupSize; i++)
                {
                    amax = MathF.Max(amax, MathF.Abs(w[rowBase + gStart + i]));
                }
                // Symmetric int4 in [-7, 7]; reserve nibble 0..15 = q + 8.
                float s = amax > 0 ? amax / 7f : 1f;
                float inv = 1f / s;
                scale[o * numGroups + g] = s;
                int sum = 0;
                for (int i = 0; i < GroupSize; i++)
                {
                    int idx = rowBase + gStart + i;
                    int q = Math.Clamp((int)MathF.Round(w[idx] * inv), -7, 7);
                    sum += q;
                    int nibble = q + 8; // 1..15
                    int packIdx = idx >> 1;
                    if ((idx & 1) == 0)
                    {
                        packed[packIdx] = (byte)((packed[packIdx] & 0xF0) | (nibble & 0x0F));
                    }
                    else
                    {
                        packed[packIdx] = (byte)((packed[packIdx] & 0x0F) | (nibble << 4));
                    }
                }
                groupSum[o * numGroups + g] = sum;
            }
        });
        return new Int4Matrix(packed, scale, groupSum, numGroups, outDim, inDim);
    }

    public void Multiply(float[] x, float[] y, int seq)
    {
        if (UseVnni())
        {
            MultiplyVnni(x, y, seq);
        }
        else
        {
            MultiplyFloat(x, y, seq);
        }
    }

    private void MultiplyVnni(float[] x, float[] y, int seq)
    {
        int inDim = InDim, outDim = OutDim, numGroups = _numGroups;
        var (ua, aScale) = VnniActivations.Quantize(x, seq, inDim);
        try
        {
            Parallel.For(0, outDim, o =>
            {
                // Unpack this output row's nibbles to signed int8 once, reused across all positions.
                Span<sbyte> wbuf = stackalloc sbyte[inDim];
                int rowBase = o * inDim;
                for (int i = 0; i < inDim; i++)
                {
                    int idx = rowBase + i;
                    byte b = _packed[idx >> 1];
                    int nibble = (idx & 1) == 0 ? (b & 0x0F) : (b >> 4);
                    wbuf[i] = (sbyte)(nibble - 8);
                }

                int scaleBase = o * numGroups;
                for (int s = 0; s < seq; s++)
                {
                    int sBase = s * inDim;
                    float facc = 0f;
                    for (int g = 0; g < numGroups; g++)
                    {
                        int gStart = g * GroupSize;
                        var a = Vector256.LoadUnsafe(ref ua[sBase + gStart]);
                        var wv = Vector256.LoadUnsafe(ref wbuf[gStart]);
                        int dot = Vector256.Sum(AvxVnni.MultiplyWideningAndAdd(Vector256<int>.Zero, a, wv)) - 128 * _groupSum[scaleBase + g];
                        facc += _scale[scaleBase + g] * dot;
                    }
                    y[s * outDim + o] = aScale[s] * facc;
                }
            });
        }
        finally
        {
            VnniActivations.Return(ua, aScale);
        }
    }

    private void MultiplyFloat(float[] x, float[] y, int seq)
    {
        int inDim = InDim, outDim = OutDim, numGroups = _numGroups;
        Parallel.For(0, outDim, o =>
        {
            Span<float> buf = stackalloc float[inDim];
            int rowBase = o * inDim;
            for (int g = 0; g < numGroups; g++)
            {
                float s = _scale[o * numGroups + g];
                int gStart = g * GroupSize;
                for (int i = 0; i < GroupSize; i++)
                {
                    int idx = rowBase + gStart + i;
                    byte b = _packed[idx >> 1];
                    int nibble = (idx & 1) == 0 ? (b & 0x0F) : (b >> 4);
                    buf[gStart + i] = (nibble - 8) * s;
                }
            }
            for (int sIdx = 0; sIdx < seq; sIdx++)
            {
                y[sIdx * outDim + o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(x, sIdx * inDim, inDim), buf);
            }
        });
    }
}
