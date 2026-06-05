using System.Numerics.Tensors;

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
/// <c>W[o, i] ≈ scale[o] * q[o, i]</c>, with <c>q ∈ [-127, 127]</c>. Each output row is dequantized
/// once per matmul into a small scratch buffer, then dotted against every activation row.
/// </summary>
internal sealed class Int8Matrix : IWeightMatrix
{
    private readonly sbyte[] _q;
    private readonly float[] _scale;
    public int InDim { get; }
    public int OutDim { get; }

    private Int8Matrix(sbyte[] q, float[] scale, int outDim, int inDim)
    {
        _q = q;
        _scale = scale;
        InDim = inDim;
        OutDim = outDim;
    }

    public static Int8Matrix Quantize(float[] w, int outDim, int inDim)
    {
        var q = new sbyte[(long)outDim * inDim <= int.MaxValue ? outDim * inDim : throw new OverflowException()];
        var scale = new float[outDim];
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
            for (int i = 0; i < inDim; i++)
            {
                int v = (int)MathF.Round(w[baseIdx + i] * inv);
                q[baseIdx + i] = (sbyte)Math.Clamp(v, -127, 127);
            }
        });
        return new Int8Matrix(q, scale, outDim, inDim);
    }

    public void Multiply(float[] x, float[] y, int seq)
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
/// packed two nibbles per byte. Dequantization widens each nibble to float at matmul time.
/// </summary>
internal sealed class Int4Matrix : IWeightMatrix
{
    public const int GroupSize = 32;

    private readonly byte[] _packed;   // two 4-bit weights per byte
    private readonly float[] _scale;   // [outDim * numGroups]
    private readonly int _numGroups;
    public int InDim { get; }
    public int OutDim { get; }

    private Int4Matrix(byte[] packed, float[] scale, int numGroups, int outDim, int inDim)
    {
        _packed = packed;
        _scale = scale;
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
                for (int i = 0; i < GroupSize; i++)
                {
                    int idx = rowBase + gStart + i;
                    int q = Math.Clamp((int)MathF.Round(w[idx] * inv), -7, 7);
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
            }
        });
        return new Int4Matrix(packed, scale, numGroups, outDim, inDim);
    }

    public void Multiply(float[] x, float[] y, int seq)
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
