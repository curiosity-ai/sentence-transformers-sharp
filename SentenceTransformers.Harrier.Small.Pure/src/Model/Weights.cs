using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SentenceTransformers;

namespace SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// int8 dot-accumulate used by the quantized GEMM kernels: uint8×int8 products summed into int32 lanes.
/// It picks the best instruction available at runtime, in throughput order:
/// <list type="number">
/// <item><c>AvxVnniInt8.V512</c> - <c>vpdpbsud</c> on 512-bit registers (64 int8 MACs / instruction).</item>
/// <item><c>AvxVnni</c> (<c>vpdpbusd</c>) / <c>AvxVnniInt8</c> (<c>vpdpbsud</c>) on 256-bit (32 MACs / instruction).</item>
/// <item><c>Avx512BW</c> / <c>Avx2</c> - widen to int16 + <c>vpmaddwd</c> (.NET does not expose AVX-512
/// VNNI as a standalone ISA, so this is the AVX-512 fallback when <c>AvxVnniInt8.V512</c> is absent).</item>
/// </list>
/// Every path consumes the same operands - a uint8 activation (the symmetric int8 value offset by +128)
/// and an int8 weight - so the activation buffer and the <c>128 * rowSum</c> offset correction are
/// shared. The <c>IsSupported</c> probes are JIT-time constants, so dead paths are eliminated and the
/// chosen instruction is inlined.
/// </summary>
internal static class Vnni
{
    /// <summary>Whether any int8 GEMM path is available (true on any AVX2-capable x64).</summary>
    public static bool IsSupported => AvxVnni.IsSupported || AvxVnniInt8.IsSupported || Avx2.IsSupported;

    /// <summary>Activation zero-point (uint8 offset for the unsigned operand). Always 8-bit; every code
    /// path accumulates into int32 without saturation.</summary>
    public const int ZeroPoint = 128;

    /// <summary>Max magnitude of a quantized activation (8-bit symmetric).</summary>
    public const int QMax = 127;

    /// <summary>Use the 512-bit kernel when a 512-bit int8 dot is available, or when AVX-512 is present
    /// without any 256-bit int8-VNNI (so the wider widen+madd beats the 256-bit one). A 256-bit
    /// <c>vpdpbusd</c>/<c>vpdpbsud</c> is faster per element than 512-bit widen+madd, so the 256-bit
    /// kernel is preferred when one of those exists.</summary>
    public static bool Use512 =>
        AvxVnniInt8.V512.IsSupported ||
        (Avx512BW.IsSupported && !AvxVnni.IsSupported && !AvxVnniInt8.IsSupported);

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector256<int> DotAccumulate(Vector256<int> acc, Vector256<byte> a, Vector256<sbyte> b)
    {
        if (AvxVnni.IsSupported)
        {
            return AvxVnni.MultiplyWideningAndAdd(acc, a, b);          // vpdpbusd: uint8 a * int8 b
        }
        if (AvxVnniInt8.IsSupported)
        {
            return AvxVnniInt8.MultiplyWideningAndAdd(acc, b, a);      // vpdpbsud: int8 b * uint8 a
        }
        // AVX2 fallback: widen uint8/int8 to int16 and use vpmaddwd (accumulates pairs into int32 with
        // no saturation, so the full 8-bit activation range is safe). vpmaddubsw is avoided because its
        // int16 intermediate would saturate for 8-bit activations.
        var aLo = Avx2.ConvertToVector256Int16(a.GetLower());
        var aHi = Avx2.ConvertToVector256Int16(a.GetUpper());
        var bLo = Avx2.ConvertToVector256Int16(b.GetLower());
        var bHi = Avx2.ConvertToVector256Int16(b.GetUpper());
        acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(aLo, bLo));
        acc = Avx2.Add(acc, Avx2.MultiplyAddAdjacent(aHi, bHi));
        return acc;
    }

    /// <summary>512-bit int8 dot-accumulate (64 uint8×int8 into 16 int32 lanes).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    public static Vector512<int> DotAccumulate512(Vector512<int> acc, Vector512<byte> a, Vector512<sbyte> b)
    {
        if (AvxVnniInt8.V512.IsSupported)
        {
            return AvxVnniInt8.V512.MultiplyWideningAndAdd(acc, b, a); // vpdpbsud (512): int8 b * uint8 a
        }
        // AVX-512 fallback: widen to int16 and vpmaddwd.
        var aLo = Avx512BW.ConvertToVector512Int16(a.GetLower());
        var aHi = Avx512BW.ConvertToVector512Int16(a.GetUpper());
        var bLo = Avx512BW.ConvertToVector512Int16(b.GetLower());
        var bHi = Avx512BW.ConvertToVector512Int16(b.GetUpper());
        acc += Avx512BW.MultiplyAddAdjacent(aLo, bLo);
        acc += Avx512BW.MultiplyAddAdjacent(aHi, bHi);
        return acc;
    }
}

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

    /// <summary>Multiplies a batch of activations, parallelizing over output channels on the
    /// <see cref="GlobalThreadPool"/> so the (CPU-heavy) work does not block the awaiting thread and
    /// honours <paramref name="ct"/>.</summary>
    ValueTask MultiplyAsync(float[] x, float[] y, int seq, CancellationToken ct);

    /// <summary>Builds the weight matrix for the chosen precision. For the quantized variants the
    /// (CPU-heavy) quantization runs on the <see cref="GlobalThreadPool"/> so it does not block a
    /// thread-pool thread when awaited from the async load path.</summary>
    static async Task<IWeightMatrix> CreateAsync(float[] weights, int outDim, int inDim, Quantization quantization, CancellationToken ct)
    {
        switch (quantization)
        {
            case Quantization.None:
                return new FloatMatrix(weights, outDim, inDim);
            case Quantization.Int8:
                return await Int8Matrix.QuantizeAsync(weights, outDim, inDim, ct).ConfigureAwait(false);
            case Quantization.Int4:
                return await Int4Matrix.QuantizeAsync(weights, outDim, inDim, ct).ConfigureAwait(false);
            default:
                throw new ArgumentOutOfRangeException(nameof(quantization));
        }
    }
}

/// <summary>Shared helpers for the VNNI int8 GEMM paths.</summary>
internal static class VnniActivations
{
    /// <summary>Dynamic per-row symmetric int8 activation quantization, offset by +128 into the uint8
    /// operand range that <c>vpdpbusd</c> expects. Returns pooled buffers the caller must return.</summary>
    public static async ValueTask<(byte[] Ua, float[] Scale)> QuantizeAsync(float[] x, int seq, int inDim, CancellationToken ct)
    {
        byte[] ua     = ArrayPool<byte>.Shared.Rent(seq * inDim);
        float[] scale = ArrayPool<float>.Shared.Rent(seq);
        
        int zp = Vnni.ZeroPoint;
        int qmax = Vnni.QMax;

        await GlobalThreadPool.ForAsync(0, seq, (x, ua, scale, inDim, zp, qmax), static (start, end, st) =>
        {
            for (int s = start; s < end; s++)
            {
                QuantizeRow(st.x, st.ua, st.scale, s, st.inDim, st.zp, st.qmax);
            }
        }, ct).ConfigureAwait(false);

        return (ua, scale);
    }

    private static void QuantizeRow(float[] x, byte[] ua, float[] scale, int s, int inDim, int zp, int qmax)
    {
        int b = s * inDim;
        float amax = 0f;
        for (int i = 0; i < inDim; i++)
        {
            amax = MathF.Max(amax, MathF.Abs(x[b + i]));
        }
        float sc = amax > 0 ? amax / qmax : 1f;
        float inv = 1f / sc;
        scale[s] = sc;
        for (int i = 0; i < inDim; i++)
        {
            int q = Math.Clamp((int)MathF.Round(x[b + i] * inv), -qmax, qmax);
            ua[b + i] = (byte)(q + zp);
        }
    }

    public static void Return(byte[] ua, float[] scale)
    {
        ArrayPool<byte>.Shared.Return(ua);
        ArrayPool<float>.Shared.Return(scale);
    }
}

/// <summary>Plain float32 weights. Delegates to <see cref="Ops.LinearAsync"/>.</summary>
internal sealed class FloatMatrix(float[] weights, int outDim, int inDim) : IWeightMatrix
{
    private readonly float[] _w = weights;
    public int InDim => inDim;
    public int OutDim => outDim;

    public ValueTask MultiplyAsync(float[] x, float[] y, int seq, CancellationToken ct)
        => new ValueTask(Numerics.Ops.LinearAsync(x, _w, y, seq, inDim, outDim, ct));
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

    private static bool UseVnni(int inDim) => Vnni.IsSupported && (inDim % 32 == 0);

    private Int8Matrix(sbyte[] q, float[] scale, int[] rowSum, int outDim, int inDim)
    {
        _q = q;
        _scale = scale;
        _rowSum = rowSum;
        InDim = inDim;
        OutDim = outDim;
    }

    public static async Task<Int8Matrix> QuantizeAsync(float[] w, int outDim, int inDim, CancellationToken ct)
    {
        var q = new sbyte[(long)outDim * inDim <= int.MaxValue ? outDim * inDim : throw new OverflowException()];
        var scale = new float[outDim];
        var rowSum = new int[outDim];
        await GlobalThreadPool.ForAsync(0, outDim, (w, q, scale, rowSum, inDim), static (start, end, st) =>
        {
            for (int o = start; o < end; o++)
            {
                int baseIdx = o * st.inDim;
                float amax = 0f;
                for (int i = 0; i < st.inDim; i++)
                {
                    amax = MathF.Max(amax, MathF.Abs(st.w[baseIdx + i]));
                }
                float s = amax > 0 ? amax / 127f : 1f;
                float inv = 1f / s;
                st.scale[o] = s;
                int sum = 0;
                for (int i = 0; i < st.inDim; i++)
                {
                    int v = Math.Clamp((int)MathF.Round(st.w[baseIdx + i] * inv), -127, 127);
                    st.q[baseIdx + i] = (sbyte)v;
                    sum += v;
                }
                st.rowSum[o] = sum;
            }
        }, ct).ConfigureAwait(false);
        return new Int8Matrix(q, scale, rowSum, outDim, inDim);
    }

    public ValueTask MultiplyAsync(float[] x, float[] y, int seq, CancellationToken ct)
        => UseVnni(InDim) ? MultiplyVnniAsync(x, y, seq, ct) : MultiplyFloatAsync(x, y, seq, ct);

    /// <summary>True int8 GEMM (W8A8) via VNNI <c>vpdpbusd</c>. Activations are quantized per row to
    /// signed int8, offset by +128 to feed the unsigned operand of <c>vpdpbusd</c>; the offset is
    /// corrected with the precomputed per-channel weight sum.
    ///
    /// The matmul is register-tiled 4 output channels x 2 sequence positions: each activation vector
    /// load feeds four dot products and each weight vector load feeds two, so the activation matrix is
    /// read ~4x fewer times. This matmul is memory-bound (the int8 dot itself is cheap), so cutting that
    /// traffic is the dominant speed-up. The eight accumulators fit in the 16 AVX YMM registers.</summary>
    private async ValueTask MultiplyVnniAsync(float[] x, float[] y, int seq, CancellationToken ct)
    {
        int inDim = InDim, outDim = OutDim;
        var (ua, aScale) = await VnniActivations.QuantizeAsync(x, seq, inDim, ct).ConfigureAwait(false);
        try
        {
            bool use512 = Vnni.Use512 && (inDim % 64 == 0);
            int oTiles = (outDim + 3) / 4;
            await GlobalThreadPool.ForAsync(0, oTiles, (self: this, ua, aScale, y, seq, inDim, outDim, use512), static (start, end, st) =>
            {
                for (int ot = start; ot < end; ot++)
                {
                    int o0 = ot * 4;
                    if (o0 + 4 <= st.outDim)
                    {
                        if (st.use512)
                        {
                            st.self.Tile4_512(st.ua, st.aScale, st.y, o0, st.seq, st.inDim, st.outDim);
                        }
                        else
                        {
                            st.self.Tile4(st.ua, st.aScale, st.y, o0, st.seq, st.inDim, st.outDim);
                        }
                    }
                    else
                    {
                        for (int o = o0; o < st.outDim; o++)
                        {
                            st.self.SingleChannel(st.ua, st.aScale, st.y, o, st.seq, st.inDim, st.outDim);
                        }
                    }
                }
            }, ct).ConfigureAwait(false);
        }
        finally
        {
            VnniActivations.Return(ua, aScale);
        }
    }

    /// <summary>512-bit variant of <see cref="Tile4"/> for AVX-512 hosts (no AVX-512 VNNI is exposed by
    /// the runtime, so this widens to int16 and uses vpmaddwd): the same 4-output x 2-position register
    /// tile, processing 64 int8 elements per step.</summary>
    private void Tile4_512(byte[] ua, float[] aScale, float[] y, int o0, int seq, int inDim, int outDim)
    {
        int wb0 = o0 * inDim, wb1 = wb0 + inDim, wb2 = wb1 + inDim, wb3 = wb2 + inDim;
        float sc0 = _scale[o0], sc1 = _scale[o0 + 1], sc2 = _scale[o0 + 2], sc3 = _scale[o0 + 3];
        int of0 = Vnni.ZeroPoint * _rowSum[o0], of1 = Vnni.ZeroPoint * _rowSum[o0 + 1], of2 = Vnni.ZeroPoint * _rowSum[o0 + 2], of3 = Vnni.ZeroPoint * _rowSum[o0 + 3];

        ref byte uaRef = ref MemoryMarshal.GetArrayDataReference(ua);
        ref sbyte qRef = ref MemoryMarshal.GetArrayDataReference(_q);
        ref float aScaleRef = ref MemoryMarshal.GetArrayDataReference(aScale);
        ref float yRef = ref MemoryMarshal.GetArrayDataReference(y);

        int s = 0;
        for (; s + 2 <= seq; s += 2)
        {
            int u0 = s * inDim, u1 = u0 + inDim;
            var c00 = Vector512<int>.Zero; var c01 = Vector512<int>.Zero; var c02 = Vector512<int>.Zero; var c03 = Vector512<int>.Zero;
            var c10 = Vector512<int>.Zero; var c11 = Vector512<int>.Zero; var c12 = Vector512<int>.Zero; var c13 = Vector512<int>.Zero;
            for (int i = 0; i < inDim; i += 64)
            {
                var av0 = Vector512.LoadUnsafe(ref Unsafe.Add(ref uaRef, u0 + i));
                var av1 = Vector512.LoadUnsafe(ref Unsafe.Add(ref uaRef, u1 + i));
                var w0  = Vector512.LoadUnsafe(ref Unsafe.Add(ref qRef , wb0 + i));
                var w1  = Vector512.LoadUnsafe(ref Unsafe.Add(ref qRef , wb1 + i));
                var w2  = Vector512.LoadUnsafe(ref Unsafe.Add(ref qRef , wb2 + i));
                var w3  = Vector512.LoadUnsafe(ref Unsafe.Add(ref qRef , wb3 + i));
                c00 = Vnni.DotAccumulate512(c00, av0, w0);
                c01 = Vnni.DotAccumulate512(c01, av0, w1);
                c02 = Vnni.DotAccumulate512(c02, av0, w2);
                c03 = Vnni.DotAccumulate512(c03, av0, w3);
                c10 = Vnni.DotAccumulate512(c10, av1, w0);
                c11 = Vnni.DotAccumulate512(c11, av1, w1);
                c12 = Vnni.DotAccumulate512(c12, av1, w2);
                c13 = Vnni.DotAccumulate512(c13, av1, w3);
            }
            
            float as0 = aScale[s], as1 = aScale[s + 1];
            
            int b0 = s * outDim + o0, b1 = b0 + outDim;

            Unsafe.Add(ref yRef, b0)      = as0 * sc0 * (Vector512.Sum(c00) - of0);
            Unsafe.Add(ref yRef, b0 + 1)  = as0 * sc1 * (Vector512.Sum(c01) - of1);
            Unsafe.Add(ref yRef, b0 + 2)  = as0 * sc2 * (Vector512.Sum(c02) - of2);
            Unsafe.Add(ref yRef, b0 + 3)  = as0 * sc3 * (Vector512.Sum(c03) - of3);
            Unsafe.Add(ref yRef, b1)      = as1 * sc0 * (Vector512.Sum(c10) - of0);
            Unsafe.Add(ref yRef, b1 + 1)  = as1 * sc1 * (Vector512.Sum(c11) - of1);
            Unsafe.Add(ref yRef, b1 + 2)  = as1 * sc2 * (Vector512.Sum(c12) - of2);
            Unsafe.Add(ref yRef, b1 + 3)  = as1 * sc3 * (Vector512.Sum(c13) - of3);
        }
        for (; s < seq; s++)
        {
            int u0 = s * inDim;
            var c0 = Vector512<int>.Zero; var c1 = Vector512<int>.Zero; var c2 = Vector512<int>.Zero; var c3 = Vector512<int>.Zero;
            for (int i = 0; i < inDim; i += 64)
            {
                var av = Vector512.LoadUnsafe(ref Unsafe.Add(ref uaRef, u0 + i));
                c0 = Vnni.DotAccumulate512(c0, av, Vector512.LoadUnsafe(ref Unsafe.Add(ref qRef, wb0 + i)));
                c1 = Vnni.DotAccumulate512(c1, av, Vector512.LoadUnsafe(ref Unsafe.Add(ref qRef, wb1 + i)));
                c2 = Vnni.DotAccumulate512(c2, av, Vector512.LoadUnsafe(ref Unsafe.Add(ref qRef, wb2 + i)));
                c3 = Vnni.DotAccumulate512(c3, av, Vector512.LoadUnsafe(ref Unsafe.Add(ref qRef, wb3 + i)));
            }
            float as0 = aScale[s];
            int b0 = s * outDim + o0;
            Unsafe.Add(ref yRef, b0    ) = as0 * sc0 * (Vector512.Sum(c0) - of0);
            Unsafe.Add(ref yRef, b0 + 1) = as0 * sc1 * (Vector512.Sum(c1) - of1);
            Unsafe.Add(ref yRef, b0 + 2) = as0 * sc2 * (Vector512.Sum(c2) - of2);
            Unsafe.Add(ref yRef, b0 + 3) = as0 * sc3 * (Vector512.Sum(c3) - of3);
        }
    }

    /// <summary>Computes a tile of 4 output channels (o0..o0+3) for all sequence positions, blocking
    /// 2 positions at a time so each activation load is reused across the 4 channels.</summary>
    private void Tile4(byte[] ua, float[] aScale, float[] y, int o0, int seq, int inDim, int outDim)
    {
        int wb0 = o0 * inDim;
        int wb1 = wb0 + inDim;
        int wb2 = wb1 + inDim;
        int wb3 = wb2 + inDim;

        float sc0 = _scale[o0];
        float sc1 = _scale[o0 + 1];
        float sc2 = _scale[o0 + 2];
        float sc3 = _scale[o0 + 3];

        int of0 = Vnni.ZeroPoint * _rowSum[o0];
        int of1 = Vnni.ZeroPoint * _rowSum[o0 + 1];
        int of2 = Vnni.ZeroPoint * _rowSum[o0 + 2];
        int of3 = Vnni.ZeroPoint * _rowSum[o0 + 3];

        ref byte uaRef = ref MemoryMarshal.GetArrayDataReference(ua);
        ref sbyte qRef = ref MemoryMarshal.GetArrayDataReference(_q);
        ref float aScaleRef = ref MemoryMarshal.GetArrayDataReference(aScale);
        ref float yRef = ref MemoryMarshal.GetArrayDataReference(y);

        int s = 0;

        for (; s + 4 <= seq; s += 4)
        {
            var c00 = Vector256<int>.Zero; var c01 = Vector256<int>.Zero; var c02 = Vector256<int>.Zero; var c03 = Vector256<int>.Zero;
            var c10 = Vector256<int>.Zero; var c11 = Vector256<int>.Zero; var c12 = Vector256<int>.Zero; var c13 = Vector256<int>.Zero;
            var c20 = Vector256<int>.Zero; var c21 = Vector256<int>.Zero; var c22 = Vector256<int>.Zero; var c23 = Vector256<int>.Zero;
            var c30 = Vector256<int>.Zero; var c31 = Vector256<int>.Zero; var c32 = Vector256<int>.Zero; var c33 = Vector256<int>.Zero;

            int u0 = s * inDim;
            int u1 = u0 + inDim;
            int u2 = u1 + inDim;
            int u3 = u2 + inDim;

            for (int i = 0; i < inDim; i += 32)
            {
                var av0 = Vector256.LoadUnsafe(ref Unsafe.Add(ref uaRef, u0 + i));
                var av1 = Vector256.LoadUnsafe(ref Unsafe.Add(ref uaRef, u1 + i));
                var av2 = Vector256.LoadUnsafe(ref Unsafe.Add(ref uaRef, u2 + i));
                var av3 = Vector256.LoadUnsafe(ref Unsafe.Add(ref uaRef, u3 + i));

                var w0 = Vector256.LoadUnsafe(ref Unsafe.Add(ref qRef, wb0 + i));

                c00 = Vnni.DotAccumulate(c00, av0, w0);
                c10 = Vnni.DotAccumulate(c10, av1, w0);
                c20 = Vnni.DotAccumulate(c20, av2, w0);
                c30 = Vnni.DotAccumulate(c30, av3, w0);

                var w1 = Vector256.LoadUnsafe(ref Unsafe.Add(ref qRef, wb1 + i));

                c01 = Vnni.DotAccumulate(c01, av0, w1);
                c11 = Vnni.DotAccumulate(c11, av1, w1);
                c21 = Vnni.DotAccumulate(c21, av2, w1);
                c31 = Vnni.DotAccumulate(c31, av3, w1);

                var w2 = Vector256.LoadUnsafe(ref Unsafe.Add(ref qRef, wb2 + i));

                c02 = Vnni.DotAccumulate(c02, av0, w2);
                c12 = Vnni.DotAccumulate(c12, av1, w2);
                c22 = Vnni.DotAccumulate(c22, av2, w2);
                c32 = Vnni.DotAccumulate(c32, av3, w2);

                var w3 = Vector256.LoadUnsafe(ref Unsafe.Add(ref qRef, wb3 + i));

                c03 = Vnni.DotAccumulate(c03, av0, w3);
                c13 = Vnni.DotAccumulate(c13, av1, w3);
                c23 = Vnni.DotAccumulate(c23, av2, w3);
                c33 = Vnni.DotAccumulate(c33, av3, w3);
            }

            float as0 = Unsafe.Add(ref aScaleRef, s);
            float as1 = Unsafe.Add(ref aScaleRef, s + 1);
            float as2 = Unsafe.Add(ref aScaleRef, s + 2);
            float as3 = Unsafe.Add(ref aScaleRef, s + 3);

            int b0 = s * outDim + o0;
            int b1 = b0 + outDim;
            int b2 = b1 + outDim;
            int b3 = b2 + outDim;

            Unsafe.Add(ref yRef, b0)     = as0 * sc0 * (Vector256.Sum(c00) - of0);
            Unsafe.Add(ref yRef, b0 + 1) = as0 * sc1 * (Vector256.Sum(c01) - of1);
            Unsafe.Add(ref yRef, b0 + 2) = as0 * sc2 * (Vector256.Sum(c02) - of2);
            Unsafe.Add(ref yRef, b0 + 3) = as0 * sc3 * (Vector256.Sum(c03) - of3);

            Unsafe.Add(ref yRef, b1)     = as1 * sc0 * (Vector256.Sum(c10) - of0);
            Unsafe.Add(ref yRef, b1 + 1) = as1 * sc1 * (Vector256.Sum(c11) - of1);
            Unsafe.Add(ref yRef, b1 + 2) = as1 * sc2 * (Vector256.Sum(c12) - of2);
            Unsafe.Add(ref yRef, b1 + 3) = as1 * sc3 * (Vector256.Sum(c13) - of3);

            Unsafe.Add(ref yRef, b2)     = as2 * sc0 * (Vector256.Sum(c20) - of0);
            Unsafe.Add(ref yRef, b2 + 1) = as2 * sc1 * (Vector256.Sum(c21) - of1);
            Unsafe.Add(ref yRef, b2 + 2) = as2 * sc2 * (Vector256.Sum(c22) - of2);
            Unsafe.Add(ref yRef, b2 + 3) = as2 * sc3 * (Vector256.Sum(c23) - of3);

            Unsafe.Add(ref yRef, b3)     = as3 * sc0 * (Vector256.Sum(c30) - of0);
            Unsafe.Add(ref yRef, b3 + 1) = as3 * sc1 * (Vector256.Sum(c31) - of1);
            Unsafe.Add(ref yRef, b3 + 2) = as3 * sc2 * (Vector256.Sum(c32) - of2);
            Unsafe.Add(ref yRef, b3 + 3) = as3 * sc3 * (Vector256.Sum(c33) - of3);
        }

        for (; s < seq; s++)
        {
            var c0 = Vector256<int>.Zero; var c1 = Vector256<int>.Zero; var c2 = Vector256<int>.Zero; var c3 = Vector256<int>.Zero;

            int u0 = s * inDim;

            for (int i = 0; i < inDim; i += 32)
            {
                var av = Vector256.LoadUnsafe(ref Unsafe.Add(ref uaRef, u0 + i));

                var w0 = Vector256.LoadUnsafe(ref Unsafe.Add(ref qRef, wb0 + i));
                c0 = Vnni.DotAccumulate(c0, av, w0);

                var w1 = Vector256.LoadUnsafe(ref Unsafe.Add(ref qRef, wb1 + i));
                c1 = Vnni.DotAccumulate(c1, av, w1);

                var w2 = Vector256.LoadUnsafe(ref Unsafe.Add(ref qRef, wb2 + i));
                c2 = Vnni.DotAccumulate(c2, av, w2);

                var w3 = Vector256.LoadUnsafe(ref Unsafe.Add(ref qRef, wb3 + i));
                c3 = Vnni.DotAccumulate(c3, av, w3);
            }

            float as0 = Unsafe.Add(ref aScaleRef, s);
            int b0    = s * outDim + o0;

            Unsafe.Add(ref yRef, b0)     = as0 * sc0 * (Vector256.Sum(c0) - of0);
            Unsafe.Add(ref yRef, b0 + 1) = as0 * sc1 * (Vector256.Sum(c1) - of1);
            Unsafe.Add(ref yRef, b0 + 2) = as0 * sc2 * (Vector256.Sum(c2) - of2);
            Unsafe.Add(ref yRef, b0 + 3) = as0 * sc3 * (Vector256.Sum(c3) - of3);
        }
    }

    /// <summary>Tail path for the (rare) output channels that do not fill a 4-wide tile.</summary>
    private void SingleChannel(byte[] ua, float[] aScale, float[] y, int o, int seq, int inDim, int outDim)
    {
        int wBase = o * inDim;
        float wsc = _scale[o];
        int offset = Vnni.ZeroPoint * _rowSum[o];
        for (int s = 0; s < seq; s++)
        {
            int uBase = s * inDim;
            var acc = Vector256<int>.Zero;
            for (int i = 0; i < inDim; i += 32)
            {
                acc = Vnni.DotAccumulate(acc, Vector256.LoadUnsafe(ref ua[uBase + i]), Vector256.LoadUnsafe(ref _q[wBase + i]));
            }
            y[s * outDim + o] = aScale[s] * wsc * (Vector256.Sum(acc) - offset);
        }
    }

    /// <summary>Portable fallback: dequantize each weight row to float and dot against float activations.</summary>
    private ValueTask MultiplyFloatAsync(float[] x, float[] y, int seq, CancellationToken ct)
    {
        int inDim = InDim, outDim = OutDim;
        return new ValueTask(GlobalThreadPool.ForAsync(0, outDim, (self: this, x, y, seq, inDim, outDim), static (start, end, st) =>
        {
            for (int o = start; o < end; o++)
            {
                st.self.FloatColumn(st.x, st.y, st.seq, st.inDim, st.outDim, o);
            }
        }, ct));
    }

    // Separate sync method so the stackalloc/Span does not live inside a (forbidden) async body.
    private void FloatColumn(float[] x, float[] y, int seq, int inDim, int outDim, int o)
    {
        Span<float> buf = stackalloc float[inDim];
        int baseIdx = o * inDim;
        float s = _scale[o];
        for (int i = 0; i < inDim; i++)
        {
            buf[i] = _q[baseIdx + i] * s;
        }
        var outIndex = o;
        var inIndex = 0;
        var inSpan = x.AsSpan();
        for (int sIdx = 0; sIdx < seq; sIdx++)
        {
            y[outIndex] = TensorPrimitives.Dot(inSpan.Slice(inIndex, inDim), buf);
            outIndex += outDim;
            inIndex += inDim;
        }
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

    private static bool UseVnni() => Vnni.IsSupported && GroupSize == 32;

    private Int4Matrix(byte[] packed, float[] scale, int[] groupSum, int numGroups, int outDim, int inDim)
    {
        _packed = packed;
        _scale = scale;
        _groupSum = groupSum;
        _numGroups = numGroups;
        InDim = inDim;
        OutDim = outDim;
    }

    public static async Task<Int4Matrix> QuantizeAsync(float[] w, int outDim, int inDim, CancellationToken ct)
    {
        if (inDim % GroupSize != 0)
        {
            throw new NotSupportedException($"Int4 requires inDim ({inDim}) to be a multiple of the group size {GroupSize}.");
        }
        int numGroups = inDim / GroupSize;
        var packed = new byte[outDim * inDim / 2];
        var scale = new float[outDim * numGroups];
        var groupSum = new int[outDim * numGroups];

        await GlobalThreadPool.ForAsync(0, outDim, (w, packed, scale, groupSum, numGroups, inDim), static (start, end, st) =>
        {
            for (int o = start; o < end; o++)
            {
                int rowBase = o * st.inDim;
                for (int g = 0; g < st.numGroups; g++)
                {
                    int gStart = g * GroupSize;
                    float amax = 0f;
                    for (int i = 0; i < GroupSize; i++)
                    {
                        amax = MathF.Max(amax, MathF.Abs(st.w[rowBase + gStart + i]));
                    }
                    // Symmetric int4 in [-7, 7]; reserve nibble 0..15 = q + 8.
                    float s = amax > 0 ? amax / 7f : 1f;
                    float inv = 1f / s;
                    st.scale[o * st.numGroups + g] = s;
                    int sum = 0;
                    for (int i = 0; i < GroupSize; i++)
                    {
                        int idx = rowBase + gStart + i;
                        int q = Math.Clamp((int)MathF.Round(st.w[idx] * inv), -7, 7);
                        sum += q;
                        int nibble = q + 8; // 1..15
                        int packIdx = idx >> 1;
                        if ((idx & 1) == 0)
                        {
                            st.packed[packIdx] = (byte)((st.packed[packIdx] & 0xF0) | (nibble & 0x0F));
                        }
                        else
                        {
                            st.packed[packIdx] = (byte)((st.packed[packIdx] & 0x0F) | (nibble << 4));
                        }
                    }
                    st.groupSum[o * st.numGroups + g] = sum;
                }
            }
        }, ct).ConfigureAwait(false);
        return new Int4Matrix(packed, scale, groupSum, numGroups, outDim, inDim);
    }

    public ValueTask MultiplyAsync(float[] x, float[] y, int seq, CancellationToken ct)
        => UseVnni() ? MultiplyVnniAsync(x, y, seq, ct) : MultiplyFloatAsync(x, y, seq, ct);

    private async ValueTask MultiplyVnniAsync(float[] x, float[] y, int seq, CancellationToken ct)
    {
        int inDim = InDim, outDim = OutDim;
        var (ua, aScale) = await VnniActivations.QuantizeAsync(x, seq, inDim, ct).ConfigureAwait(false);
        try
        {
            await GlobalThreadPool.ForAsync(0, outDim, (self: this, ua, aScale, y, seq, inDim, outDim), static (start, end, st) =>
            {
                for (int o = start; o < end; o++)
                {
                    st.self.VnniColumn(st.ua, st.aScale, st.y, o, st.seq, st.inDim, st.outDim);
                }
            }, ct).ConfigureAwait(false);
        }
        finally
        {
            VnniActivations.Return(ua, aScale);
        }
    }

    // Sync (stackalloc) per-channel kernel: unpack this output row's nibbles to signed int8 once,
    // reused across all positions, then int8-dot each group with its own scale.
    private void VnniColumn(byte[] ua, float[] aScale, float[] y, int o, int seq, int inDim, int outDim)
    {
        int numGroups = _numGroups;
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
                int dot = Vector256.Sum(Vnni.DotAccumulate(Vector256<int>.Zero, a, wv)) - Vnni.ZeroPoint * _groupSum[scaleBase + g];
                facc += _scale[scaleBase + g] * dot;
            }
            y[s * outDim + o] = aScale[s] * facc;
        }
    }

    private ValueTask MultiplyFloatAsync(float[] x, float[] y, int seq, CancellationToken ct)
    {
        int inDim = InDim, outDim = OutDim;
        return new ValueTask(GlobalThreadPool.ForAsync(0, outDim, (self: this, x, y, seq, inDim, outDim), static (start, end, st) =>
        {
            for (int o = start; o < end; o++)
            {
                st.self.FloatColumn(st.x, st.y, st.seq, st.inDim, st.outDim, o);
            }
        }, ct));
    }

    private void FloatColumn(float[] x, float[] y, int seq, int inDim, int outDim, int o)
    {
        int numGroups = _numGroups;
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
    }
}
