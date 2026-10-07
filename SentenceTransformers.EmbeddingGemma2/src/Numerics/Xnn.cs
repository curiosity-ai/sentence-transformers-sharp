using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>
/// Bit-exact ports of the XNNPACK micro-kernels LiteRT runs on x86-64 CPUs with AVX-512 (the CPU path of the
/// LiteRT-LM embedding engine), so the managed encoder reproduces the engine's float rounding instead of
/// merely approximating it. Every int8 activation quantization downstream turns a one-ulp difference into a
/// whole quantization step with some probability, so matching the elementwise kernels bit for bit is what
/// makes the port agree with the reference to ~1e-7 instead of ~3e-3.
/// <para>
/// Each function mirrors one kernel of google/XNNPACK as bundled with LiteRT-LM 0.18: the arithmetic order, the fused multiply-adds, the SIMD lane layout of reductions and the
/// polynomial / rational approximations are reproduced exactly. The 16-lane accumulators of the AVX-512 kernels
/// are emulated with 128-bit vectors and the AVX-512 <c>vrsqrt14ps</c> estimate is emulated in software where the
/// instruction is unavailable (<see cref="Rsqrt14"/>), so results are identical on every CPU (x86 with or
/// without AVX-512, ARM64).
/// </para>
/// </summary>
internal static class Xnn
{
    private const int Lanes = 16;   // AVX-512 f32 lanes

    /// <summary>A float from its IEEE-754 bit pattern (the kernels' hexadecimal constants).</summary>
    private static float F(uint bits) => BitConverter.UInt32BitsToSingle(bits);

    // ---------------------------------------------------------------------------------------------
    // Lane helpers
    // ---------------------------------------------------------------------------------------------

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Fmadd(Vector128<float> a, Vector128<float> b, Vector128<float> c)
    {
        if (Fma.IsSupported)
        {
            return Fma.MultiplyAdd(a, b, c);
        }
        if (AdvSimd.IsSupported)
        {
            return AdvSimd.FusedMultiplyAdd(c, a, b);
        }
        return Vector128.Create(
            MathF.FusedMultiplyAdd(a[0], b[0], c[0]), MathF.FusedMultiplyAdd(a[1], b[1], c[1]),
            MathF.FusedMultiplyAdd(a[2], b[2], c[2]), MathF.FusedMultiplyAdd(a[3], b[3], c[3]));
    }

    /// <summary><c>xnn_reduce_add_f32</c> (AVX-512): 512 → 256 → 128-bit halving adds, then
    /// <c>movehl</c> and <c>movehdup</c>; the 16 lanes are given as four 4-lane quarters.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float ReduceAdd(Vector128<float> q0, Vector128<float> q1, Vector128<float> q2, Vector128<float> q3)
    {
        var a = (q0 + q2) + (q1 + q3);   // lanes i + (i + 8), then i + (i + 4)
        float d0 = a[0] + a[2], d1 = a[1] + a[3];
        return d0 + d1;
    }

    // ---------------------------------------------------------------------------------------------
    // RMS norm: mul(x, x) + mean -> reduce_mean_squared (f32-rsum2 avx512f u64 acc4), add eps,
    // rsqrt (f32-vrsqrt avx512f: rsqrt14 + one Newton-Raphson step), mul(x, r), mul(., w)
    // ---------------------------------------------------------------------------------------------

    /// <summary><c>xnn_f32_rsum2_ukernel__avx512f_u64_acc4</c>: <c>Σ x²</c> with four 16-lane FMA accumulators,
    /// times <paramref name="scale"/>.</summary>
    public static float SumOfSquares(ReadOnlySpan<float> x, float scale)
    {
        // acc[k] is a 16-lane accumulator held as four 4-lane quarters.
        Span<Vector128<float>> acc = stackalloc Vector128<float>[16];
        acc.Clear();
        ref float xr = ref MemoryMarshal.GetReference(x);
        int n = x.Length, p = 0;
        for (; n - p >= 64; p += 64)
        {
            for (int k = 0; k < 4; k++)
            {
                for (int q = 0; q < 4; q++)
                {
                    var v = Vector128.LoadUnsafe(ref xr, (nuint)(p + 16 * k + 4 * q));
                    acc[4 * k + q] = Fmadd(v, v, acc[4 * k + q]);
                }
            }
        }
        for (int k = 0; k < 3 && n - p >= Lanes; k++, p += Lanes)
        {
            for (int q = 0; q < 4; q++)
            {
                var v = Vector128.LoadUnsafe(ref xr, (nuint)(p + 4 * q));
                acc[4 * k + q] = Fmadd(v, v, acc[4 * k + q]);
            }
        }
        for (int q = 0; q < 4; q++)
        {
            acc[q] += acc[8 + q];
            acc[4 + q] += acc[12 + q];
            acc[q] += acc[4 + q];
        }
        int rem = n - p;
        if (rem > 0)
        {
            Span<float> tail = stackalloc float[Lanes];
            tail.Clear();
            x.Slice(p, rem).CopyTo(tail);
            ref float tr = ref MemoryMarshal.GetReference(tail);
            for (int q = 0; q < 4; q++)
            {
                var v = Vector128.LoadUnsafe(ref tr, (nuint)(4 * q));
                acc[q] = Fmadd(v, v, acc[q]);
            }
        }
        return 0f + ReduceAdd(acc[0], acc[1], acc[2], acc[3]) * scale;
    }

    /// <summary><c>xnn_f32_vrsqrt_ukernel__avx512f_rsqrt</c>: the 14-bit hardware estimate refined by one
    /// Newton-Raphson step evaluated without fused multiply-adds.</summary>
    public static float ReciprocalSqrt(float a)
    {
        float y = Avx512F.VL.IsSupported
            ? Avx512F.VL.ReciprocalSqrt14(Vector128.CreateScalarUnsafe(a)).ToScalar()
            : Rsqrt14.Estimate(a);
        float t1 = y * y;
        float t2 = a * t1;
        float t3 = t2 - 1f;
        float t4 = 0.5f * y;
        float t5 = t3 * t4;
        y -= t5;
        return float.IsPositiveInfinity(a) ? 0f : y;
    }

    /// <summary>The inlined <c>odml.rms_norm</c> decomposition as XNNPACK executes it:
    /// <c>(x · rsqrt(mean(x²) + eps)) · w</c> per <paramref name="dim"/>-sized row (in place allowed).</summary>
    public static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, Span<float> dst, int dim, float eps)
    {
        int rows = x.Length / dim;
        float invDim = 1f / dim;
        for (int r = 0; r < rows; r++)
        {
            var row = x.Slice(r * dim, dim);
            var o = dst.Slice(r * dim, dim);
            float inv = ReciprocalSqrt(SumOfSquares(row, invDim) + eps);
            int i = 0;
            if (Vector256.IsHardwareAccelerated)
            {
                var vinv = Vector256.Create(inv);
                for (; i + 8 <= dim; i += 8)
                {
                    var v = Vector256.Create(row.Slice(i, 8)) * vinv;
                    if (!weight.IsEmpty)
                    {
                        v *= Vector256.Create(weight.Slice(i, 8));
                    }
                    v.CopyTo(o.Slice(i, 8));
                }
            }
            for (; i < dim; i++)
            {
                float v = row[i] * inv;
                o[i] = weight.IsEmpty ? v : v * weight[i];
            }
        }
    }

    /// <summary><c>xnn_f32_rsum_ukernel__avx512f_u32_acc2</c>: <c>Σ x</c> with two 16-lane accumulators, times
    /// <paramref name="scale"/> (1 for SUM, 1/n for MEAN).</summary>
    public static float Sum(ReadOnlySpan<float> x, float scale)
    {
        Span<Vector128<float>> acc = stackalloc Vector128<float>[8];
        acc.Clear();
        ref float xr = ref MemoryMarshal.GetReference(x);
        int n = x.Length, p = 0;
        for (; n - p >= 32; p += 32)
        {
            for (int q = 0; q < 8; q++)
            {
                acc[q] += Vector128.LoadUnsafe(ref xr, (nuint)(p + 4 * q));
            }
        }
        if (n - p >= Lanes)
        {
            for (int q = 0; q < 4; q++)
            {
                acc[q] += Vector128.LoadUnsafe(ref xr, (nuint)(p + 4 * q));
            }
            p += Lanes;
        }
        for (int q = 0; q < 4; q++)
        {
            acc[q] += acc[4 + q];
        }
        int rem = n - p;
        if (rem > 0)
        {
            Span<float> tail = stackalloc float[Lanes];
            tail.Clear();
            x.Slice(p, rem).CopyTo(tail);
            ref float tr = ref MemoryMarshal.GetReference(tail);
            for (int q = 0; q < 4; q++)
            {
                acc[q] += Vector128.LoadUnsafe(ref tr, (nuint)(4 * q));
            }
        }
        return 0f + ReduceAdd(acc[0], acc[1], acc[2], acc[3]) * scale;
    }

    // ---------------------------------------------------------------------------------------------
    // GELU (approximate=true): xnn_f32_vapproxgelu_ukernel__avx512f_rational_12_10_div
    // ---------------------------------------------------------------------------------------------

    private const float GeluMaxX = 4.84974098e+00f, GeluMinX = -4.84974098e+00f;
    private const float GeluA1 = 7.9788458347e-01f, GeluA3 = 6.0803253204e-02f, GeluA5 = 7.2898347862e-03f;
    private const float GeluA7 = 2.6887017884e-04f, GeluA9 = 1.4302649106e-05f, GeluA11 = 4.9544411240e-08f;
    private const float GeluB2 = 2.4369759858e-01f, GeluB4 = 2.4381054565e-02f, GeluB6 = 1.3060354395e-03f;
    private const float GeluB8 = 7.6477612311e-05f, GeluB10 = 1.3433452750e-06f;

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Gelu(Vector128<float> xo)
    {
        var x = Vector128.Max(Vector128.Create(GeluMinX), Vector128.Min(Vector128.Create(GeluMaxX), xo));
        var x2 = x * x;
        var p = Fmadd(x2, Vector128.Create(GeluA11), Vector128.Create(GeluA9));
        p = Fmadd(x2, p, Vector128.Create(GeluA7));
        p = Fmadd(x2, p, Vector128.Create(GeluA5));
        p = Fmadd(x2, p, Vector128.Create(GeluA3));
        p = Fmadd(x2, p, Vector128.Create(GeluA1));
        p = x * p;
        var q = Fmadd(x2, Vector128.Create(GeluB10), Vector128.Create(GeluB8));
        q = Fmadd(x2, q, Vector128.Create(GeluB6));
        q = Fmadd(x2, q, Vector128.Create(GeluB4));
        q = Fmadd(x2, q, Vector128.Create(GeluB2));
        q = Fmadd(x2, q, Vector128<float>.One);
        return (xo * Vector128.Create(0.5f)) * (p / q + Vector128<float>.One);
    }

    /// <summary>Tanh-approximated GELU, in place, exactly as XNNPACK's rational 12/10 kernel computes it.</summary>
    public static void Gelu(Span<float> x)
    {
        ref float r = ref MemoryMarshal.GetReference(x);
        int i = 0;
        for (; i + 4 <= x.Length; i += 4)
        {
            Gelu(Vector128.LoadUnsafe(ref r, (nuint)i)).StoreUnsafe(ref r, (nuint)i);
        }
        if (i < x.Length)
        {
            Span<float> tail = stackalloc float[4];
            tail.Clear();
            x.Slice(i).CopyTo(tail);
            var y = Gelu(Vector128.Create((ReadOnlySpan<float>)tail));
            for (int j = 0; i < x.Length; i++, j++)
            {
                x[i] = y[j];
            }
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Sigmoid: xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div
    // ---------------------------------------------------------------------------------------------

    private static readonly float[] SigmoidTable =
    {
        F(0x3F800000), F(0x3F82CD87), F(0x3F85AAC3), F(0x3F88980F), F(0x3F8B95C2), F(0x3F8EA43A), F(0x3F91C3D3), F(0x3F94F4F0),
        F(0x3F9837F0), F(0x3F9B8D3A), F(0x3F9EF532), F(0x3FA27043), F(0x3FA5FED7), F(0x3FA9A15B), F(0x3FAD583F), F(0x3FB123F6),
        F(0x3FB504F3), F(0x3FB8FBAF), F(0x3FBD08A4), F(0x3FC12C4D), F(0x3FC5672A), F(0x3FC9B9BE), F(0x3FCE248C), F(0x3FD2A81E),
        F(0x3FD744FD), F(0x3FDBFBB8), F(0x3FE0CCDF), F(0x3FE5B907), F(0x3FEAC0C7), F(0x3FEFE4BA), F(0x3FF5257D), F(0x3FFA83B3),
    };

    /// <summary>Logistic sigmoid as XNNPACK's AVX-512 kernel computes it: <c>e = exp(−|x|)</c> from a 32-entry table,
    /// a degree-2 polynomial and <c>scalef</c>; <c>f = e / (e + 1)</c>, mirrored for <c>x ≥ 0</c>.</summary>
    public static void Sigmoid(ReadOnlySpan<float> x, Span<float> y)
    {
        float MagicBias = F(0x48C00000), Log2e = F(0x3FB8AA3B);
        float MinusLn2Hi = F(0xBF317218), MinusLn2Lo = F(0x3102E308), C2 = F(0x3F000000), C1 = F(0x3F80007B);
        int i = 0;
        if (Avx2.IsSupported && x.Length >= 8)
        {
            i = SigmoidAvx2(x, y);
        }
        for (; i < x.Length; i++)
        {
            int xb = BitConverter.SingleToInt32Bits(x[i]);
            float z = BitConverter.Int32BitsToSingle(xb | int.MinValue);
            float n = MathF.FusedMultiplyAdd(z, Log2e, MagicBias);
            float l = SigmoidTable[BitConverter.SingleToInt32Bits(n) & 31];
            n -= MagicBias;
            float t = MathF.FusedMultiplyAdd(n, MinusLn2Hi, z);
            t = MathF.FusedMultiplyAdd(n, MinusLn2Lo, t);
            float p = MathF.FusedMultiplyAdd(t, C2, C1);
            t *= l;
            p = MathF.FusedMultiplyAdd(t, p, l);
            float e = MathF.ScaleB(p, (int)MathF.Floor(n));   // vscalefps
            if (MathF.Abs(e) < 1.17549435E-38f)   // smallest normal float
            {
                e = 0f;   // the engine runs with flush-to-zero: denormal results become +0
            }
            float f = e / (e + 1f);
            y[i] = xb >= 0 ? 1f - f : f;
        }
    }

    /// <summary>8 lanes at a time of <see cref="Sigmoid"/>, lane for lane the same operations: the table lookup is a
    /// gather, and <c>scalef(p, ⌊n⌋)</c> is the exact product <c>p · 2^⌊n⌋</c> (n ≤ 0 and p &lt; 2, so whenever
    /// <c>⌊n⌋ &lt; −126</c> the result is below the smallest normal and flushed to +0 anyway).
    /// Returns the number of elements done.</summary>
    private static unsafe int SigmoidAvx2(ReadOnlySpan<float> x, Span<float> y)
    {
        var magic = Vector256.Create(F(0x48C00000));
        var log2e = Vector256.Create(F(0x3FB8AA3B));
        var ln2Hi = Vector256.Create(F(0xBF317218));
        var ln2Lo = Vector256.Create(F(0x3102E308));
        var c2 = Vector256.Create(F(0x3F000000));
        var c1 = Vector256.Create(F(0x3F80007B));
        var minNormal = Vector256.Create(1.17549435E-38f);
        var one = Vector256.Create(1f);
        var signBit = Vector256.Create(int.MinValue);
        int i = 0;
        fixed (float* table = SigmoidTable)
        {
            ref float xr = ref MemoryMarshal.GetReference(x);
            ref float yr = ref MemoryMarshal.GetReference(y);
            for (; i + 8 <= x.Length; i += 8)
            {
                var xv = Vector256.LoadUnsafe(ref xr, (nuint)i);
                var xb = xv.AsInt32();
                var z = (xb | signBit).AsSingle();
                var n = Vector256.FusedMultiplyAdd(z, log2e, magic);
                var l = Avx2.GatherVector256(table, n.AsInt32() & Vector256.Create(31), 4);
                n -= magic;
                var t = Vector256.FusedMultiplyAdd(n, ln2Hi, z);
                t = Vector256.FusedMultiplyAdd(n, ln2Lo, t);
                var p = Vector256.FusedMultiplyAdd(t, c2, c1);
                t *= l;
                p = Vector256.FusedMultiplyAdd(t, p, l);
                var fn = Vector256.Floor(n);
                var pow2 = Vector256.ShiftLeft(Vector256.ConvertToInt32(fn) + Vector256.Create(127), 23).AsSingle();
                var e = Vector256.ConditionalSelect(Vector256.LessThan(fn, Vector256.Create(-126f)), Vector256<float>.Zero, p * pow2);
                e = Vector256.ConditionalSelect(Vector256.LessThan(Vector256.Abs(e), minNormal), Vector256<float>.Zero, e);
                var f = e / (e + one);
                var r = Vector256.ConditionalSelect(Vector256.GreaterThanOrEqual(xb, Vector256<int>.Zero).AsSingle(), one - f, f);
                r.StoreUnsafe(ref yr, (nuint)i);
            }
        }
        return i;
    }

    // ---------------------------------------------------------------------------------------------
    // Tanh: xnn_f32_vtanh_ukernel__avx512f_rational_9_8_div
    // ---------------------------------------------------------------------------------------------

    /// <summary><c>tanh(x)</c> as XNNPACK's rational 9/8 kernel computes it (FMA variant).</summary>
    public static void Tanh(ReadOnlySpan<float> x, Span<float> y)
    {
        const float MaxX = 7.9807181358e+00f, MinX = -7.9807181358e+00f;
        const float A3 = 1.3412411511e-01f, A5 = 3.5330520477e-03f, A7 = 2.1235626264e-05f, A9 = 1.4248920266e-08f;
        const float B2 = 4.6745735407e-01f, B4 = 2.6018999517e-02f, B6 = 3.3472978976e-04f, B8 = 8.1365948290e-07f;
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            ref float xr = ref MemoryMarshal.GetReference(x);
            ref float yr = ref MemoryMarshal.GetReference(y);
            for (; i + 8 <= x.Length; i += 8)
            {
                var v = Vector256.Max(Vector256.Create(MinX), Vector256.Min(Vector256.Create(MaxX), Vector256.LoadUnsafe(ref xr, (nuint)i)));
                var v2 = v * v;
                var p = Vector256.FusedMultiplyAdd(v2, Vector256.Create(A9), Vector256.Create(A7));
                p = Vector256.FusedMultiplyAdd(v2, p, Vector256.Create(A5));
                p = Vector256.FusedMultiplyAdd(v2, p, Vector256.Create(A3));
                p = Vector256.FusedMultiplyAdd(v2, p, Vector256.Create(1f));
                p = v * p;
                var q = Vector256.FusedMultiplyAdd(v2, Vector256.Create(B8), Vector256.Create(B6));
                q = Vector256.FusedMultiplyAdd(v2, q, Vector256.Create(B4));
                q = Vector256.FusedMultiplyAdd(v2, q, Vector256.Create(B2));
                q = Vector256.FusedMultiplyAdd(v2, q, Vector256.Create(1f));
                (p / q).StoreUnsafe(ref yr, (nuint)i);
            }
        }
        for (; i < x.Length; i++)
        {
            float v = MathF.Max(MinX, MathF.Min(MaxX, x[i]));
            float v2 = v * v;
            float p = MathF.FusedMultiplyAdd(v2, A9, A7);
            p = MathF.FusedMultiplyAdd(v2, p, A5);
            p = MathF.FusedMultiplyAdd(v2, p, A3);
            p = MathF.FusedMultiplyAdd(v2, p, 1f);
            p = v * p;
            float q = MathF.FusedMultiplyAdd(v2, B8, B6);
            q = MathF.FusedMultiplyAdd(v2, q, B4);
            q = MathF.FusedMultiplyAdd(v2, q, B2);
            q = MathF.FusedMultiplyAdd(v2, q, 1f);
            y[i] = p / q;
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Softmax: rmax, xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2, 1/sum, vmulc
    // ---------------------------------------------------------------------------------------------

    private static readonly Vector128<float> ExpLog2e = Vector128.Create(BitConverter.Int32BitsToSingle(0x3FB8AA3B));
    private static readonly Vector128<float> ExpMagicBias = Vector128.Create(BitConverter.Int32BitsToSingle(0x4B40007F));
    private static readonly Vector128<float> ExpMinusLn2Hi = Vector128.Create(BitConverter.Int32BitsToSingle(unchecked((int)0xBF317200)));
    private static readonly Vector128<float> ExpMinusLn2Lo = Vector128.Create(BitConverter.Int32BitsToSingle(unchecked((int)0xB5BFBE8E)));
    private static readonly Vector128<float> ExpC5 = Vector128.Create(BitConverter.Int32BitsToSingle(0x3C07CFCE));
    private static readonly Vector128<float> ExpC4 = Vector128.Create(BitConverter.Int32BitsToSingle(0x3D2B9D0D));
    private static readonly Vector128<float> ExpC3 = Vector128.Create(BitConverter.Int32BitsToSingle(0x3E2AAD40));
    private static readonly Vector128<float> ExpC2 = Vector128.Create(BitConverter.Int32BitsToSingle(0x3EFFFEE3));
    private static readonly Vector128<float> ExpC1 = Vector128.Create(BitConverter.Int32BitsToSingle(0x3F7FFFFB));
    private static readonly Vector128<float> ExpDenormCutoff = Vector128.Create(BitConverter.Int32BitsToSingle(unchecked((int)0xC2AEAC4F)));

    /// <summary><c>exp(x)</c> for <c>x ≤ 0</c> (rr2_p5: Cody-Waite reduction, degree-5 polynomial, flush below the
    /// denormal cutoff).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ExpNonPositive(Vector128<float> x)
    {
        var n = Fmadd(x, ExpLog2e, ExpMagicBias);
        var s = Vector128.ShiftLeft(n.AsInt32(), 23).AsSingle();
        n -= ExpMagicBias;
        var t = Fmadd(n, ExpMinusLn2Hi, x);
        t = Fmadd(n, ExpMinusLn2Lo, t);
        var p = Fmadd(ExpC5, t, ExpC4);
        p = Fmadd(p, t, ExpC3);
        p = Fmadd(p, t, ExpC2);
        p = Fmadd(p, t, ExpC1);
        t *= s;
        var f = Fmadd(t, p, s);
        return Vector128.ConditionalSelect(Vector128.LessThan(x, ExpDenormCutoff), Vector128<float>.Zero, f);
    }

    /// <summary>
    /// XNNPACK f32 softmax over one row, in place. The kernel sums <c>exp</c> values into two 16-lane accumulators
    /// (alternating per 16 elements) over whole 64-element blocks, folds them, then adds the remaining 16-element
    /// blocks and the masked tail into the folded accumulator, so the rounding depends on the row length the kernel
    /// sees. <paramref name="maskedPrefix"/> = true means <paramref name="row"/> is the valid prefix of a longer row
    /// padded to a multiple of 64 with masked logits (the text encoder's signature padding): those produce exact
    /// zeros and every valid element falls in the 64-element blocks.
    /// </summary>
    public static void Softmax(Span<float> row, bool maskedPrefix = false)
    {
        float max = TensorPrimitives.Max<float>(row);
        var vmax = Vector128.Create(max);
        int n = row.Length;
        // Elements handled by the 64-wide loop (alternating accumulators); the rest go to the folded accumulator.
        int blockEnd = maskedPrefix ? n : n / 64 * 64;
        Span<Vector128<float>> acc = stackalloc Vector128<float>[8];
        acc.Clear();
        int j = 0;
        if (Vector256.IsHardwareAccelerated && blockEnd >= 32)
        {
            // Same lanes, two 256-bit halves per 16-lane accumulator: every lane sees the same operations in the
            // same order, so the result is identical to the 128-bit loop below.
            ref float r0 = ref MemoryMarshal.GetReference(row);
            var vmax8 = Vector256.Create(max);
            Vector256<float> aLo = default, aHi = default, bLo = default, bHi = default;
            for (; j + 32 <= blockEnd; j += 32)
            {
                aLo += ExpStore(ref r0, j, vmax8);
                aHi += ExpStore(ref r0, j + 8, vmax8);
                bLo += ExpStore(ref r0, j + 16, vmax8);
                bHi += ExpStore(ref r0, j + 24, vmax8);
            }
            acc[0] = aLo.GetLower();
            acc[1] = aLo.GetUpper();
            acc[2] = aHi.GetLower();
            acc[3] = aHi.GetUpper();
            acc[4] = bLo.GetLower();
            acc[5] = bLo.GetUpper();
            acc[6] = bHi.GetLower();
            acc[7] = bHi.GetUpper();
        }
        for (; j < blockEnd; j += 4)
        {
            acc[((j / Lanes) & 1) * 4 + (j % Lanes) / 4] += ExpGroup(row, j, vmax);
        }
        for (int q = 0; q < 4; q++)
        {
            acc[q] += acc[4 + q];
        }
        for (; j < n; j += 4)
        {
            acc[(j % Lanes) / 4] += ExpGroup(row, j, vmax);
        }
        float scale = 1f / ReduceAdd(acc[0], acc[1], acc[2], acc[3]);
        TensorPrimitives.Multiply(row, scale, row);
    }

    /// <summary>The 256-bit form of <see cref="ExpGroup"/> for 8 in-range elements.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> ExpStore(ref float row, int j, Vector256<float> vmax)
    {
        var x = Vector256.LoadUnsafe(ref row, (nuint)j) - vmax;
        var n = Vector256.FusedMultiplyAdd(x, Vector256.Create(ExpLog2e.ToScalar()), Vector256.Create(ExpMagicBias.ToScalar()));
        var sc = Vector256.ShiftLeft(n.AsInt32(), 23).AsSingle();
        n -= Vector256.Create(ExpMagicBias.ToScalar());
        var t = Vector256.FusedMultiplyAdd(n, Vector256.Create(ExpMinusLn2Hi.ToScalar()), x);
        t = Vector256.FusedMultiplyAdd(n, Vector256.Create(ExpMinusLn2Lo.ToScalar()), t);
        var p = Vector256.FusedMultiplyAdd(Vector256.Create(ExpC5.ToScalar()), t, Vector256.Create(ExpC4.ToScalar()));
        p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(ExpC3.ToScalar()));
        p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(ExpC2.ToScalar()));
        p = Vector256.FusedMultiplyAdd(p, t, Vector256.Create(ExpC1.ToScalar()));
        t *= sc;
        var f = Vector256.FusedMultiplyAdd(t, p, sc);
        f = Vector256.ConditionalSelect(Vector256.LessThan(x, Vector256.Create(ExpDenormCutoff.ToScalar())), Vector256<float>.Zero, f);
        f.StoreUnsafe(ref row, (nuint)j);
        return f;
    }

    /// <summary><c>exp(row[j..j+4] − max)</c> stored back in place; lanes past the end of the row count as masked
    /// (exact zeros).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> ExpGroup(Span<float> row, int j, Vector128<float> vmax)
    {
        ref float r = ref MemoryMarshal.GetReference(row);
        if (j + 4 <= row.Length)
        {
            var e = ExpNonPositive(Vector128.LoadUnsafe(ref r, (nuint)j) - vmax);
            e.StoreUnsafe(ref r, (nuint)j);
            return e;
        }
        Span<float> tail = stackalloc float[4];
        tail.Fill(float.NegativeInfinity);
        row.Slice(j).CopyTo(tail);
        var et = ExpNonPositive(Vector128.Create((ReadOnlySpan<float>)tail) - vmax);
        for (int k = 0; j + k < row.Length; k++)
        {
            row[j + k] = et[k];
        }
        return et;
    }

    // ---------------------------------------------------------------------------------------------
    // Sine / cosine: xnn_f32_vsin / vcos_ukernel__avx512f_rational_5_4_div
    // ---------------------------------------------------------------------------------------------

    /// <summary><c>sin(x)</c> as XNNPACK's rational 5/4 kernel computes it.</summary>
    public static float Sin(float x) => SinCos(x, cosine: false);

    /// <summary><c>cos(x)</c> as XNNPACK's rational 5/4 kernel computes it (<c>sin(π/2 − x)</c> with π/2 split into
    /// hi + lo parts, as in the XNNPACK bundled with LiteRT-LM 0.18; the older standalone LiteRT 2.2 interpreter
    /// subtracts a single rounded π/2, which differs by an ulp for some angles).</summary>
    public static float Cos(float x) => SinCos(x, cosine: true);

    private static float SinCos(float x, bool cosine)
    {
        const float Pi = 3.1415927f, TwoPiInv = 0.15915494f;
        const float PiHalfHi = 1.5707963705e+00f, PiHalfLo = -4.3711388e-08f;   // π/2 split for an accurate π/2 − x
        const float TwoPiHi = 6.28125f, TwoPiLo = 1.9353072e-3f;
        const float A3 = -1.3314664364e-01f, A5 = 3.2340581529e-03f, B2 = 3.3519912511e-02f, B4 = 4.8770775902e-04f;
        float k = MathF.Round(x * TwoPiInv, MidpointRounding.ToEven);
        x = MathF.FusedMultiplyAdd(-k, TwoPiHi, x);
        x = MathF.FusedMultiplyAdd(-k, TwoPiLo, x);
        if (cosine)
        {
            x = (PiHalfHi - x) + PiHalfLo;
        }
        x = MathF.Min(x, Pi - x);
        x = MathF.Max(x, -Pi - x);
        x = MathF.Min(x, Pi - x);
        float x2 = x * x;
        float p = MathF.FusedMultiplyAdd(x2, A5, A3);
        p = MathF.FusedMultiplyAdd(x2, p, 1f);
        p = x * p;
        float q = MathF.FusedMultiplyAdd(x2, B4, B2);
        q = MathF.FusedMultiplyAdd(x2, q, 1f);
        return p / q;
    }

    // ---------------------------------------------------------------------------------------------
    // Engine post-processing
    // ---------------------------------------------------------------------------------------------

    /// <summary>LiteRT-LM's <c>L2Norm</c>: a sequential float sum of squares, <c>sqrt</c>, element-wise division.</summary>
    public static void L2Normalize(Span<float> v)
    {
        float sum = 0f;
        foreach (var x in v)
        {
            sum += x * x;
        }
        if (sum <= 0f)
        {
            return;
        }
        float norm = MathF.Sqrt(sum);
        for (int i = 0; i < v.Length; i++)
        {
            v[i] /= norm;
        }
    }
}
