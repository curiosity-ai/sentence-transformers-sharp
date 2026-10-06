using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Runtime.Intrinsics.X86;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>
/// Row-wise dynamically quantized activations: every row of a float matrix is mapped to 8 bits with its
/// own scale and zero point, exactly like XNNPACK's <c>f32 -&gt; qd8</c> conversion that the LiteRT CPU
/// runtime applies in front of every int4/int8 <c>FULLY_CONNECTED</c>. Values are stored offset by +128
/// (unsigned) which is the operand form the x86 <c>pmaddubsw</c>/<c>vpdpbusd</c> instructions expect.
/// </summary>
internal sealed class QuantizedActivations
{
    public int Rows { get; private set; }
    public int Cols { get; private set; }
    public int Stride { get; private set; }
    public byte[] Data { get; private set; } = Array.Empty<byte>();
    public float[] Scale { get; private set; } = Array.Empty<float>();
    public int[] ZeroPoint { get; private set; } = Array.Empty<int>();

    public void Resize(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        Stride = (cols + QuantizedMatrix.Alignment - 1) / QuantizedMatrix.Alignment * QuantizedMatrix.Alignment;
        if (Data.Length < rows * Stride)
        {
            // Padding bytes multiply zero weights, so their content never matters.
            Data = new byte[rows * Stride];
        }
        if (Scale.Length < rows)
        {
            Scale = new float[rows];
            ZeroPoint = new int[rows];
        }
    }

    /// <summary>Row-parallel overload of <see cref="Quantize(ReadOnlySpan{float}, int, int, int)"/>.</summary>
    public void Quantize(float[] x, int rows, int cols, int ldx, ParallelOptions po)
    {
        using var _ = Profiler.Measure("quantize");
        Resize(rows, cols);
        ParallelRows.For(rows, cols, po, (r0, r1) =>
        {
            for (int r = r0; r < r1; r++)
            {
                QuantizeRow(x.AsSpan(r * ldx, cols), Data.AsSpan(r * Stride, cols), out Scale[r], out ZeroPoint[r]);
            }
        });
    }

    /// <summary>Quantizes <paramref name="rows"/> rows of <paramref name="x"/> (row stride <paramref name="ldx"/>).</summary>
    public void Quantize(ReadOnlySpan<float> x, int rows, int cols, int ldx)
    {
        using var _ = Profiler.Measure("quantize");
        Resize(rows, cols);
        for (int r = 0; r < rows; r++)
        {
            QuantizeRow(x.Slice(r * ldx, cols), Data.AsSpan(r * Stride, cols), out Scale[r], out ZeroPoint[r]);
        }
    }

    /// <summary>
    /// Static (calibrated) quantization, as the TFLite <c>QUANTIZE</c> op in front of the vision tower's
    /// int8 <c>FULLY_CONNECTED</c> layers: every row uses the same <paramref name="scale"/> and zero point,
    /// values outside the int8 range saturate.
    /// </summary>
    public void QuantizeStatic(float[] x, int rows, int cols, int ldx, float scale, int zeroPoint, ParallelOptions po)
    {
        using var _ = Profiler.Measure("quantize");
        Resize(rows, cols);
        ParallelRows.For(rows, cols, po, (r0, r1) =>
        {
            for (int r = r0; r < r1; r++)
            {
                QuantizeRow(x.AsSpan(r * ldx, cols), Data.AsSpan(r * Stride, cols), scale, zeroPoint);
                Scale[r] = scale;
                ZeroPoint[r] = zeroPoint;
            }
        });
    }

    /// <summary>XNNPACK <c>xnn_f32_qd8_asymmetric_quantization_params</c> + <c>f32-qs8-vcvt</c>.</summary>
    internal static void QuantizeRow(ReadOnlySpan<float> x, Span<byte> dst, out float scale, out int zeroPoint)
    {
        float min = MathF.Min(0f, TensorPrimitives.Min(x));
        float max = MathF.Max(0f, TensorPrimitives.Max(x));
        const float QMin = -128f, QMax = 127f;
        scale = min == max ? 1f : (max - min) / (QMax - QMin);
        float minScaled = min / scale;
        float maxScaled = max / scale;
        float zp = (QMin + minScaled) + (QMax + maxScaled) > 0 ? QMin - minScaled : QMax - maxScaled;
        zp = Math.Clamp(zp, QMin, QMax);
        zeroPoint = (int)MathF.Round(zp, MidpointRounding.ToEven);
        QuantizeRow(x, dst, scale, zeroPoint);
    }

    /// <summary><c>f32-qs8-vcvt</c>: <c>q = clamp(rint(x · (1/scale)) + zp)</c>, stored offset by +128.</summary>
    internal static void QuantizeRow(ReadOnlySpan<float> x, Span<byte> dst, float scale, int zeroPoint)
    {
        const float QMin = -128f, QMax = 127f;
        float inv = 1f / scale;
        float zpf = zeroPoint;

        int i = 0;
        if (Avx2.IsSupported && x.Length >= 32)
        {
            unsafe
            {
                fixed (float* xp = x)
                fixed (byte* dp = dst)
                {
                    var vinv = Vector256.Create(inv);
                    var vzp = Vector256.Create(zpf);
                    var off = Vector256.Create(128);
                    // packs/packus interleave 128-bit lanes; this permutation restores element order.
                    var order = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
                    for (; i + 32 <= x.Length; i += 32)
                    {
                        // Separate multiply and add (no FMA) to match the reference rounding; vroundps
                        // rounds half to even like lrintf. The unsigned saturating pack is the clamp.
                        var a = Avx2.Add(Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Add(Avx.Multiply(Avx.LoadVector256(xp + i), vinv), vzp))), off);
                        var b = Avx2.Add(Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Add(Avx.Multiply(Avx.LoadVector256(xp + i + 8), vinv), vzp))), off);
                        var c = Avx2.Add(Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Add(Avx.Multiply(Avx.LoadVector256(xp + i + 16), vinv), vzp))), off);
                        var d = Avx2.Add(Avx.ConvertToVector256Int32(Avx.RoundToNearestInteger(Avx.Add(Avx.Multiply(Avx.LoadVector256(xp + i + 24), vinv), vzp))), off);
                        var ab = Avx2.PackSignedSaturate(a, b);
                        var cd = Avx2.PackSignedSaturate(c, d);
                        var bytes = Avx2.PackUnsignedSaturate(ab, cd);
                        Avx.Store(dp + i, Avx2.PermuteVar8x32(bytes.AsInt32(), order).AsByte());
                    }
                }
            }
        }
        else if (Vector256.IsHardwareAccelerated && x.Length >= 32)
        {
            var vinv = Vector256.Create(inv);
            var vzp = Vector256.Create(zpf);
            var lo = Vector256.Create(QMin);
            var hi = Vector256.Create(QMax);
            var off = Vector256.Create(128);
            ref float xr = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(x);
            ref byte dr = ref System.Runtime.InteropServices.MemoryMarshal.GetReference(dst);
            for (; i + 32 <= x.Length; i += 32)
            {
                // Separate multiply and add (no FMA) to match the reference rounding.
                var a = Vector256.ConvertToInt32(Vector256.Clamp(Vector256.Round(Vector256.LoadUnsafe(ref xr, (nuint)i) * vinv + vzp), lo, hi)) + off;
                var b = Vector256.ConvertToInt32(Vector256.Clamp(Vector256.Round(Vector256.LoadUnsafe(ref xr, (nuint)(i + 8)) * vinv + vzp), lo, hi)) + off;
                var c = Vector256.ConvertToInt32(Vector256.Clamp(Vector256.Round(Vector256.LoadUnsafe(ref xr, (nuint)(i + 16)) * vinv + vzp), lo, hi)) + off;
                var d = Vector256.ConvertToInt32(Vector256.Clamp(Vector256.Round(Vector256.LoadUnsafe(ref xr, (nuint)(i + 24)) * vinv + vzp), lo, hi)) + off;
                var ab = Vector256.Narrow(a.AsUInt32(), b.AsUInt32());
                var cd = Vector256.Narrow(c.AsUInt32(), d.AsUInt32());
                Vector256.Narrow(ab, cd).StoreUnsafe(ref dr, (nuint)i);
            }
        }
        for (; i < x.Length; i++)
        {
            float v = MathF.Round(x[i] * inv + zpf, MidpointRounding.ToEven);
            dst[i] = (byte)((int)Math.Clamp(v, QMin, QMax) + 128);
        }
    }
}

/// <summary>
/// <c>Y[N, M] = dequant(X_q8[N, K]) · W[M, K]^T (+ bias)</c> with integer accumulation: dynamically
/// quantized activations against int8 (or int4-range) per-channel weights, the computation XNNPACK's
/// <c>qd8-f32-qc4w/qc8w</c> GEMMs perform for LiteRT. For each output
/// <c>y = sx·sw·(Σ_k xu·w − (128 + zx)·Σ_k w)</c> where <c>xu = q + 128</c> is the unsigned activation,
/// so the integer part is exact and only the final scaling is floating point.
/// Kernels: AVX-VNNI (<c>vpdpbusd</c>), AVX2 (<c>pmaddubsw</c> for int4-range weights, widening
/// <c>pmaddwd</c> for full int8), ARM <c>sdot</c>, and a portable fallback.
/// </summary>
internal static class QGemm
{
    private const int TileN = 2;   // activation rows per micro-tile
    private const int TileM = 4;   // weight rows per micro-tile
    private const int BlockN = 32;
    private const int BlockM = 64;

    private static readonly bool UseVnni = AvxVnni.IsSupported;
    private static readonly bool UseAvx512 = Avx512BW.IsSupported;
    private static readonly bool UseAvx2 = Avx2.IsSupported;
    private static readonly bool UseDp = Dp.IsSupported;

    /// <summary>
    /// Output requantization of a fully quantized int8 FC (XNNPACK <c>qs8-qc8w</c> "fp32" requantization):
    /// <c>q = clamp(rint(acc · s_in·s_w[c]/s_out), -128, 127)</c>; the result is written dequantized
    /// (<c>q · s_out</c>), i.e. fused with the <c>DEQUANTIZE</c> that follows every such FC in the graph.
    /// </summary>
    internal sealed class Requantization
    {
        public float OutputScale { get; }
        public float[] Scale { get; }

        public Requantization(float inputScale, float[] weightScale, float outputScale)
        {
            OutputScale = outputScale;
            Scale = new float[weightScale.Length];
            for (int c = 0; c < Scale.Length; c++)
            {
                Scale[c] = inputScale * weightScale[c] / outputScale;
            }
        }
    }

    public static void Multiply(QuantizedActivations x, QuantizedMatrix w, Span<float> y, int ldy, ParallelOptions po = null, Requantization requant = null)
    {
        using var _ = Profiler.Measure("qgemm");
        if (x.Cols != w.Cols)
        {
            throw new ArgumentException($"Inner dimensions differ: {x.Cols} vs {w.Cols}.");
        }
        int n = x.Rows, m = w.Rows;
        int nBlocks = (n + BlockN - 1) / BlockN;
        int mBlocks = (m + BlockM - 1) / BlockM;
        int blocks = nBlocks * mBlocks;
        int dop = po?.MaxDegreeOfParallelism ?? 1;
        if (dop == -1)
        {
            dop = Environment.ProcessorCount;
        }

        unsafe
        {
            fixed (float* yp = y)
            {
                var yPtr = (nint)yp;
                if (dop <= 1 || blocks == 1)
                {
                    for (int b = 0; b < blocks; b++)
                    {
                        RunBlock(x, w, (float*)yPtr, ldy, b / mBlocks, b % mBlocks, requant);
                    }
                }
                else
                {
                    Parallel.For(0, blocks, po, b => RunBlock(x, w, (float*)yPtr, ldy, b / mBlocks, b % mBlocks, requant));
                }
            }
        }
    }

    private static unsafe void RunBlock(QuantizedActivations x, QuantizedMatrix w, float* y, int ldy, int nb, int mb, Requantization rq)
    {
        int n0 = nb * BlockN, n1 = Math.Min(x.Rows, n0 + BlockN);
        int m0 = mb * BlockM, m1 = Math.Min(w.Rows, m0 + BlockM);
        int k = w.Stride;
        bool small = w.SmallRange;
        int* acc = stackalloc int[TileN * TileM];
        fixed (byte* xBase = x.Data)
        fixed (sbyte* wBase = w.Data)
        {
            int i = n0;
            if (UseAvx512)
            {
                // 4-row tiles; a ragged last tile repeats its final row and only stores the valid ones.
                int* acc16 = stackalloc int[16];
                for (; i < n1; i += 4)
                {
                    int valid = Math.Min(4, n1 - i);
                    byte* x0 = xBase + (long)i * x.Stride;
                    byte* x1 = xBase + (long)(i + Math.Min(1, valid - 1)) * x.Stride;
                    byte* x2 = xBase + (long)(i + Math.Min(2, valid - 1)) * x.Stride;
                    byte* x3 = xBase + (long)(i + Math.Min(3, valid - 1)) * x.Stride;
                    int j = m0;
                    for (; j + 4 <= m1; j += 4)
                    {
                        sbyte* w0 = wBase + (long)j * k;
                        Dot4x4Avx512(x0, x1, x2, x3, w0, k, small, acc16);
                        if (valid == 4)
                        {
                            Store4x4(x, w, y, ldy, i, j, acc16, rq);
                        }
                        else
                        {
                            for (int a = 0; a < valid; a++)
                            {
                                for (int b = 0; b < 4; b++)
                                {
                                    Store(x, w, y, ldy, i + a, j + b, acc16[a * 4 + b], rq);
                                }
                            }
                        }
                    }
                    for (; j < m1; j++)
                    {
                        sbyte* wr = wBase + (long)j * k;
                        for (int a = 0; a < valid; a++)
                        {
                            Store(x, w, y, ldy, i + a, j, Dot1x1(xBase + (long)(i + a) * x.Stride, wr, k, small), rq);
                        }
                    }
                }
                return;
            }
            for (; i + TileN <= n1; i += TileN)
            {
                byte* x0 = xBase + (long)i * x.Stride;
                byte* x1 = x0 + x.Stride;
                int j = m0;
                for (; j + TileM <= m1; j += TileM)
                {
                    sbyte* w0 = wBase + (long)j * k;
                    Dot2x4(x0, x1, w0, w0 + k, w0 + 2 * k, w0 + 3 * k, k, small, acc);
                    for (int a = 0; a < TileN; a++)
                    {
                        for (int b = 0; b < TileM; b++)
                        {
                            Store(x, w, y, ldy, i + a, j + b, acc[a * TileM + b], rq);
                        }
                    }
                }
                for (; j < m1; j++)
                {
                    sbyte* wr = wBase + (long)j * k;
                    Store(x, w, y, ldy, i, j, Dot1x1(x0, wr, k, small), rq);
                    Store(x, w, y, ldy, i + 1, j, Dot1x1(x1, wr, k, small), rq);
                }
            }
            for (; i < n1; i++)
            {
                byte* xr = xBase + (long)i * x.Stride;
                int j = m0;
                for (; j + TileM <= m1; j += TileM)
                {
                    sbyte* w0 = wBase + (long)j * k;
                    // Reuse the 2x4 kernel with the same row twice: still 4 outputs per pass.
                    Dot2x4(xr, xr, w0, w0 + k, w0 + 2 * k, w0 + 3 * k, k, small, acc);
                    for (int b = 0; b < TileM; b++)
                    {
                        Store(x, w, y, ldy, i, j + b, acc[b], rq);
                    }
                }
                for (; j < m1; j++)
                {
                    Store(x, w, y, ldy, i, j, Dot1x1(xr, wBase + (long)j * k, k, small), rq);
                }
            }
        }
    }

    /// <summary>Vectorized epilogue for a 4×4 tile (same arithmetic order as <see cref="Store"/>).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store4x4(QuantizedActivations x, QuantizedMatrix w, float* y, int ldy, int row, int col, int* acc, Requantization rq)
    {
        if (rq is not null)
        {
            for (int a = 0; a < 4; a++)
            {
                for (int b = 0; b < 4; b++)
                {
                    Store(x, w, y, ldy, row + a, col + b, acc[a * 4 + b], rq);
                }
            }
            return;
        }
        fixed (int* rsp = w.RowSum)
        fixed (float* swp = w.Scale)
        {
            var rs = Vector128.Load(rsp + col);
            var sw = Vector128.Load(swp + col);
            var bias = w.Bias is null ? Vector128<float>.Zero : Vector128.Create(w.Bias[col], w.Bias[col + 1], w.Bias[col + 2], w.Bias[col + 3]);
            for (int a = 0; a < 4; a++)
            {
                var corr = Vector128.Load(acc + 4 * a) - Vector128.Create(128 + x.ZeroPoint[row + a]) * rs;
                var v = Vector128.ConvertToSingle(corr) * Vector128.Create(x.Scale[row + a]) * sw;
                if (w.Bias is not null)
                {
                    v += bias;
                }
                v.Store(y + (long)(row + a) * ldy + col);
            }
        }
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Store(QuantizedActivations x, QuantizedMatrix w, float* y, int ldy, int row, int col, int acc, Requantization rq)
    {
        // Σ (q - zx)·w = Σ xu·w − (128 + zx)·Σ w, exact in int32.
        int corrected = acc - (128 + x.ZeroPoint[row]) * w.RowSum[col];
        if (rq is not null)
        {
            float r = Math.Clamp(MathF.Round((float)corrected * rq.Scale[col], MidpointRounding.ToEven), -128f, 127f);
            y[(long)row * ldy + col] = r * rq.OutputScale;
            return;
        }
        float v = (float)corrected * x.Scale[row] * w.Scale[col];
        if (w.Bias is not null)
        {
            v += w.Bias[col];
        }
        y[(long)row * ldy + col] = v;
    }

    /// <summary>2 activation rows × 4 weight rows of unsigned·signed byte dot products (k multiple of 64).</summary>
    private static unsafe void Dot2x4(byte* x0, byte* x1, sbyte* w0, sbyte* w1, sbyte* w2, sbyte* w3, int k, bool small, int* acc)
    {
        if (UseVnni)
        {
            Vector256<int> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector256<int> a10 = default, a11 = default, a12 = default, a13 = default;
            for (int p = 0; p < k; p += 32)
            {
                var u0 = Avx.LoadVector256(x0 + p);
                var u1 = Avx.LoadVector256(x1 + p);
                var b = Avx.LoadVector256(w0 + p);
                a00 = AvxVnni.MultiplyWideningAndAdd(a00, u0, b);
                a10 = AvxVnni.MultiplyWideningAndAdd(a10, u1, b);
                b = Avx.LoadVector256(w1 + p);
                a01 = AvxVnni.MultiplyWideningAndAdd(a01, u0, b);
                a11 = AvxVnni.MultiplyWideningAndAdd(a11, u1, b);
                b = Avx.LoadVector256(w2 + p);
                a02 = AvxVnni.MultiplyWideningAndAdd(a02, u0, b);
                a12 = AvxVnni.MultiplyWideningAndAdd(a12, u1, b);
                b = Avx.LoadVector256(w3 + p);
                a03 = AvxVnni.MultiplyWideningAndAdd(a03, u0, b);
                a13 = AvxVnni.MultiplyWideningAndAdd(a13, u1, b);
            }
            Reduce8(a00, a01, a02, a03, a10, a11, a12, a13, acc);
            return;
        }
        if (UseAvx2 && small)
        {
            // pmaddubsw is exact here: |w| <= 8 so u8·s8 pair sums stay within int16.
            var ones = Vector256.Create((short)1);
            Vector256<int> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector256<int> a10 = default, a11 = default, a12 = default, a13 = default;
            for (int p = 0; p < k; p += 32)
            {
                var u0 = Avx.LoadVector256(x0 + p);
                var u1 = Avx.LoadVector256(x1 + p);
                var b = Avx.LoadVector256(w0 + p);
                a00 = Avx2.Add(a00, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u0, b), ones));
                a10 = Avx2.Add(a10, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u1, b), ones));
                b = Avx.LoadVector256(w1 + p);
                a01 = Avx2.Add(a01, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u0, b), ones));
                a11 = Avx2.Add(a11, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u1, b), ones));
                b = Avx.LoadVector256(w2 + p);
                a02 = Avx2.Add(a02, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u0, b), ones));
                a12 = Avx2.Add(a12, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u1, b), ones));
                b = Avx.LoadVector256(w3 + p);
                a03 = Avx2.Add(a03, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u0, b), ones));
                a13 = Avx2.Add(a13, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u1, b), ones));
            }
            Reduce8(a00, a01, a02, a03, a10, a11, a12, a13, acc);
            return;
        }
        if (UseAvx2)
        {
            // Full int8 weights: widen both operands to int16 and use pmaddwd (exact, no saturation).
            Vector256<int> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector256<int> a10 = default, a11 = default, a12 = default, a13 = default;
            for (int p = 0; p < k; p += 16)
            {
                var u0 = Avx2.ConvertToVector256Int16(Sse2.LoadVector128(x0 + p));
                var u1 = Avx2.ConvertToVector256Int16(Sse2.LoadVector128(x1 + p));
                var b = Avx2.ConvertToVector256Int16(Sse2.LoadVector128(w0 + p));
                a00 = Avx2.Add(a00, Avx2.MultiplyAddAdjacent(u0, b));
                a10 = Avx2.Add(a10, Avx2.MultiplyAddAdjacent(u1, b));
                b = Avx2.ConvertToVector256Int16(Sse2.LoadVector128(w1 + p));
                a01 = Avx2.Add(a01, Avx2.MultiplyAddAdjacent(u0, b));
                a11 = Avx2.Add(a11, Avx2.MultiplyAddAdjacent(u1, b));
                b = Avx2.ConvertToVector256Int16(Sse2.LoadVector128(w2 + p));
                a02 = Avx2.Add(a02, Avx2.MultiplyAddAdjacent(u0, b));
                a12 = Avx2.Add(a12, Avx2.MultiplyAddAdjacent(u1, b));
                b = Avx2.ConvertToVector256Int16(Sse2.LoadVector128(w3 + p));
                a03 = Avx2.Add(a03, Avx2.MultiplyAddAdjacent(u0, b));
                a13 = Avx2.Add(a13, Avx2.MultiplyAddAdjacent(u1, b));
            }
            Reduce8(a00, a01, a02, a03, a10, a11, a12, a13, acc);
            return;
        }
        acc[0] = Dot1x1(x0, w0, k, small);
        acc[1] = Dot1x1(x0, w1, k, small);
        acc[2] = Dot1x1(x0, w2, k, small);
        acc[3] = Dot1x1(x0, w3, k, small);
        acc[4] = Dot1x1(x1, w0, k, small);
        acc[5] = Dot1x1(x1, w1, k, small);
        acc[6] = Dot1x1(x1, w2, k, small);
        acc[7] = Dot1x1(x1, w3, k, small);
    }

    /// <summary>4 activation rows × 4 weight rows with 512-bit AVX-512BW (k multiple of 64).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Dot4x4Avx512(byte* x0, byte* x1, byte* x2, byte* x3, sbyte* w0, int k, bool small, int* acc)
    {
        sbyte* w1 = w0 + k, w2 = w1 + k, w3 = w2 + k;
        Vector512<int> a00 = default, a01 = default, a02 = default, a03 = default;
        Vector512<int> a10 = default, a11 = default, a12 = default, a13 = default;
        Vector512<int> a20 = default, a21 = default, a22 = default, a23 = default;
        Vector512<int> a30 = default, a31 = default, a32 = default, a33 = default;
        if (small)
        {
            Dot4x4Avx512Small(x0, x1, x2, x3, w0, w1, w2, w3, k, acc);
            return;
        }
        {
            for (int p = 0; p < k; p += 32)
            {
                var u0 = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256(x0 + p));
                var u1 = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256(x1 + p));
                var u2 = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256(x2 + p));
                var u3 = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256(x3 + p));
                var b = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256(w0 + p));
                a00 = Avx512F.Add(a00, Avx512BW.MultiplyAddAdjacent(u0, b));
                a10 = Avx512F.Add(a10, Avx512BW.MultiplyAddAdjacent(u1, b));
                a20 = Avx512F.Add(a20, Avx512BW.MultiplyAddAdjacent(u2, b));
                a30 = Avx512F.Add(a30, Avx512BW.MultiplyAddAdjacent(u3, b));
                b = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256(w1 + p));
                a01 = Avx512F.Add(a01, Avx512BW.MultiplyAddAdjacent(u0, b));
                a11 = Avx512F.Add(a11, Avx512BW.MultiplyAddAdjacent(u1, b));
                a21 = Avx512F.Add(a21, Avx512BW.MultiplyAddAdjacent(u2, b));
                a31 = Avx512F.Add(a31, Avx512BW.MultiplyAddAdjacent(u3, b));
                b = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256(w2 + p));
                a02 = Avx512F.Add(a02, Avx512BW.MultiplyAddAdjacent(u0, b));
                a12 = Avx512F.Add(a12, Avx512BW.MultiplyAddAdjacent(u1, b));
                a22 = Avx512F.Add(a22, Avx512BW.MultiplyAddAdjacent(u2, b));
                a32 = Avx512F.Add(a32, Avx512BW.MultiplyAddAdjacent(u3, b));
                b = Avx512BW.ConvertToVector512Int16(Avx.LoadVector256(w3 + p));
                a03 = Avx512F.Add(a03, Avx512BW.MultiplyAddAdjacent(u0, b));
                a13 = Avx512F.Add(a13, Avx512BW.MultiplyAddAdjacent(u1, b));
                a23 = Avx512F.Add(a23, Avx512BW.MultiplyAddAdjacent(u2, b));
                a33 = Avx512F.Add(a33, Avx512BW.MultiplyAddAdjacent(u3, b));
            }
        }
        Sse2.Store(acc, HSum4(Fold(a00), Fold(a01), Fold(a02), Fold(a03)));
        Sse2.Store(acc + 4, HSum4(Fold(a10), Fold(a11), Fold(a12), Fold(a13)));
        Sse2.Store(acc + 8, HSum4(Fold(a20), Fold(a21), Fold(a22), Fold(a23)));
        Sse2.Store(acc + 12, HSum4(Fold(a30), Fold(a31), Fold(a32), Fold(a33)));
    }

    /// <summary>
    /// Int4-range weights: <c>pmaddubsw</c> pair sums are bounded by 255·8·2 = 4080, so eight 64-byte steps
    /// are accumulated in int16 before widening (2 instead of 3 ALU ops per 64 MACs). Only the 16 int16
    /// tiles stay in registers; int32 totals live in a stack buffer touched once per 512-byte block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Dot4x4Avx512Small(byte* x0, byte* x1, byte* x2, byte* x3, sbyte* w0, sbyte* w1, sbyte* w2, sbyte* w3, int k, int* acc)
    {
        var ones = Vector512.Create((short)1);
        if (k <= 512)
        {
            Dot4x4Avx512SmallBlock(x0, x1, x2, x3, w0, w1, w2, w3, 0, k, ones, acc);
            return;
        }
        var tot = stackalloc Vector512<int>[16];
        for (int t = 0; t < 16; t++)
        {
            tot[t] = default;
        }
        for (int p0 = 0; p0 < k; p0 += 512)
        {
            int pEnd = Math.Min(k, p0 + 512);
            Vector512<short> s00 = default, s01 = default, s02 = default, s03 = default;
            Vector512<short> s10 = default, s11 = default, s12 = default, s13 = default;
            Vector512<short> s20 = default, s21 = default, s22 = default, s23 = default;
            Vector512<short> s30 = default, s31 = default, s32 = default, s33 = default;
            for (int p = p0; p < pEnd; p += 64)
            {
                var u0 = Avx512BW.LoadVector512(x0 + p);
                var u1 = Avx512BW.LoadVector512(x1 + p);
                var u2 = Avx512BW.LoadVector512(x2 + p);
                var u3 = Avx512BW.LoadVector512(x3 + p);
                var b = Avx512BW.LoadVector512(w0 + p);
                s00 = Avx512BW.Add(s00, Avx512BW.MultiplyAddAdjacent(u0, b));
                s10 = Avx512BW.Add(s10, Avx512BW.MultiplyAddAdjacent(u1, b));
                s20 = Avx512BW.Add(s20, Avx512BW.MultiplyAddAdjacent(u2, b));
                s30 = Avx512BW.Add(s30, Avx512BW.MultiplyAddAdjacent(u3, b));
                b = Avx512BW.LoadVector512(w1 + p);
                s01 = Avx512BW.Add(s01, Avx512BW.MultiplyAddAdjacent(u0, b));
                s11 = Avx512BW.Add(s11, Avx512BW.MultiplyAddAdjacent(u1, b));
                s21 = Avx512BW.Add(s21, Avx512BW.MultiplyAddAdjacent(u2, b));
                s31 = Avx512BW.Add(s31, Avx512BW.MultiplyAddAdjacent(u3, b));
                b = Avx512BW.LoadVector512(w2 + p);
                s02 = Avx512BW.Add(s02, Avx512BW.MultiplyAddAdjacent(u0, b));
                s12 = Avx512BW.Add(s12, Avx512BW.MultiplyAddAdjacent(u1, b));
                s22 = Avx512BW.Add(s22, Avx512BW.MultiplyAddAdjacent(u2, b));
                s32 = Avx512BW.Add(s32, Avx512BW.MultiplyAddAdjacent(u3, b));
                b = Avx512BW.LoadVector512(w3 + p);
                s03 = Avx512BW.Add(s03, Avx512BW.MultiplyAddAdjacent(u0, b));
                s13 = Avx512BW.Add(s13, Avx512BW.MultiplyAddAdjacent(u1, b));
                s23 = Avx512BW.Add(s23, Avx512BW.MultiplyAddAdjacent(u2, b));
                s33 = Avx512BW.Add(s33, Avx512BW.MultiplyAddAdjacent(u3, b));
            }
            tot[0] = Avx512F.Add(tot[0], Avx512BW.MultiplyAddAdjacent(s00, ones));
            tot[1] = Avx512F.Add(tot[1], Avx512BW.MultiplyAddAdjacent(s01, ones));
            tot[2] = Avx512F.Add(tot[2], Avx512BW.MultiplyAddAdjacent(s02, ones));
            tot[3] = Avx512F.Add(tot[3], Avx512BW.MultiplyAddAdjacent(s03, ones));
            tot[4] = Avx512F.Add(tot[4], Avx512BW.MultiplyAddAdjacent(s10, ones));
            tot[5] = Avx512F.Add(tot[5], Avx512BW.MultiplyAddAdjacent(s11, ones));
            tot[6] = Avx512F.Add(tot[6], Avx512BW.MultiplyAddAdjacent(s12, ones));
            tot[7] = Avx512F.Add(tot[7], Avx512BW.MultiplyAddAdjacent(s13, ones));
            tot[8] = Avx512F.Add(tot[8], Avx512BW.MultiplyAddAdjacent(s20, ones));
            tot[9] = Avx512F.Add(tot[9], Avx512BW.MultiplyAddAdjacent(s21, ones));
            tot[10] = Avx512F.Add(tot[10], Avx512BW.MultiplyAddAdjacent(s22, ones));
            tot[11] = Avx512F.Add(tot[11], Avx512BW.MultiplyAddAdjacent(s23, ones));
            tot[12] = Avx512F.Add(tot[12], Avx512BW.MultiplyAddAdjacent(s30, ones));
            tot[13] = Avx512F.Add(tot[13], Avx512BW.MultiplyAddAdjacent(s31, ones));
            tot[14] = Avx512F.Add(tot[14], Avx512BW.MultiplyAddAdjacent(s32, ones));
            tot[15] = Avx512F.Add(tot[15], Avx512BW.MultiplyAddAdjacent(s33, ones));
        }
        for (int r = 0; r < 4; r++)
        {
            Sse2.Store(acc + 4 * r, HSum4(Fold(tot[4 * r]), Fold(tot[4 * r + 1]), Fold(tot[4 * r + 2]), Fold(tot[4 * r + 3])));
        }
    }

    /// <summary>One ≤512-byte block of <see cref="Dot4x4Avx512Small"/>, reduced straight to 16 int sums.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Dot4x4Avx512SmallBlock(byte* x0, byte* x1, byte* x2, byte* x3, sbyte* w0, sbyte* w1, sbyte* w2, sbyte* w3, int p0, int pEnd, Vector512<short> ones, int* acc)
    {
        Vector512<short> s00 = default, s01 = default, s02 = default, s03 = default;
        Vector512<short> s10 = default, s11 = default, s12 = default, s13 = default;
        Vector512<short> s20 = default, s21 = default, s22 = default, s23 = default;
        Vector512<short> s30 = default, s31 = default, s32 = default, s33 = default;
        for (int p = p0; p < pEnd; p += 64)
        {
            var u0 = Avx512BW.LoadVector512(x0 + p);
            var u1 = Avx512BW.LoadVector512(x1 + p);
            var u2 = Avx512BW.LoadVector512(x2 + p);
            var u3 = Avx512BW.LoadVector512(x3 + p);
            var b = Avx512BW.LoadVector512(w0 + p);
            s00 = Avx512BW.Add(s00, Avx512BW.MultiplyAddAdjacent(u0, b));
            s10 = Avx512BW.Add(s10, Avx512BW.MultiplyAddAdjacent(u1, b));
            s20 = Avx512BW.Add(s20, Avx512BW.MultiplyAddAdjacent(u2, b));
            s30 = Avx512BW.Add(s30, Avx512BW.MultiplyAddAdjacent(u3, b));
            b = Avx512BW.LoadVector512(w1 + p);
            s01 = Avx512BW.Add(s01, Avx512BW.MultiplyAddAdjacent(u0, b));
            s11 = Avx512BW.Add(s11, Avx512BW.MultiplyAddAdjacent(u1, b));
            s21 = Avx512BW.Add(s21, Avx512BW.MultiplyAddAdjacent(u2, b));
            s31 = Avx512BW.Add(s31, Avx512BW.MultiplyAddAdjacent(u3, b));
            b = Avx512BW.LoadVector512(w2 + p);
            s02 = Avx512BW.Add(s02, Avx512BW.MultiplyAddAdjacent(u0, b));
            s12 = Avx512BW.Add(s12, Avx512BW.MultiplyAddAdjacent(u1, b));
            s22 = Avx512BW.Add(s22, Avx512BW.MultiplyAddAdjacent(u2, b));
            s32 = Avx512BW.Add(s32, Avx512BW.MultiplyAddAdjacent(u3, b));
            b = Avx512BW.LoadVector512(w3 + p);
            s03 = Avx512BW.Add(s03, Avx512BW.MultiplyAddAdjacent(u0, b));
            s13 = Avx512BW.Add(s13, Avx512BW.MultiplyAddAdjacent(u1, b));
            s23 = Avx512BW.Add(s23, Avx512BW.MultiplyAddAdjacent(u2, b));
            s33 = Avx512BW.Add(s33, Avx512BW.MultiplyAddAdjacent(u3, b));
        }
        Sse2.Store(acc, HSum4(W(s00, ones), W(s01, ones), W(s02, ones), W(s03, ones)));
        Sse2.Store(acc + 4, HSum4(W(s10, ones), W(s11, ones), W(s12, ones), W(s13, ones)));
        Sse2.Store(acc + 8, HSum4(W(s20, ones), W(s21, ones), W(s22, ones), W(s23, ones)));
        Sse2.Store(acc + 12, HSum4(W(s30, ones), W(s31, ones), W(s32, ones), W(s33, ones)));
    }

    /// <summary>Widens an int16 accumulator to int32 pair sums and folds 512 -> 256 bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> W(Vector512<short> s, Vector512<short> ones) => Fold(Avx512BW.MultiplyAddAdjacent(s, ones));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Fold(Vector512<int> v) => Avx2.Add(v.GetLower(), v.GetUpper());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Reduce8(Vector256<int> a00, Vector256<int> a01, Vector256<int> a02, Vector256<int> a03,
                                       Vector256<int> a10, Vector256<int> a11, Vector256<int> a12, Vector256<int> a13, int* acc)
    {
        // Pairwise horizontal adds reduce four 8-lane accumulators to one vector of four sums per row.
        var r0 = HSum4(a00, a01, a02, a03);
        var r1 = HSum4(a10, a11, a12, a13);
        Sse2.Store(acc, r0);
        Sse2.Store(acc + 4, r1);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> HSum4(Vector256<int> a, Vector256<int> b, Vector256<int> c, Vector256<int> d)
    {
        var ab = Avx2.HorizontalAdd(a, b);       // a01 a23 b01 b23 | a45 a67 b45 b67
        var cd = Avx2.HorizontalAdd(c, d);
        var abcd = Avx2.HorizontalAdd(ab, cd);   // a0-3 b0-3 c0-3 d0-3 | a4-7 b4-7 c4-7 d4-7
        return Sse2.Add(abcd.GetLower(), abcd.GetUpper());
    }

    private static unsafe int Dot1x1(byte* x, sbyte* w, int k, bool small)
    {
        if (UseVnni)
        {
            Vector256<int> a = default;
            for (int p = 0; p < k; p += 32)
            {
                a = AvxVnni.MultiplyWideningAndAdd(a, Avx.LoadVector256(x + p), Avx.LoadVector256(w + p));
            }
            return Vector256.Sum(a);
        }
        if (UseAvx2 && small)
        {
            var ones = Vector256.Create((short)1);
            Vector256<int> a = default;
            for (int p = 0; p < k; p += 32)
            {
                a = Avx2.Add(a, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Avx.LoadVector256(x + p), Avx.LoadVector256(w + p)), ones));
            }
            return Vector256.Sum(a);
        }
        if (UseAvx2)
        {
            Vector256<int> a = default;
            for (int p = 0; p < k; p += 16)
            {
                a = Avx2.Add(a, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(Sse2.LoadVector128(x + p)), Avx2.ConvertToVector256Int16(Sse2.LoadVector128(w + p))));
            }
            return Vector256.Sum(a);
        }
        if (UseDp)
        {
            // sdot works on signed bytes: flip xu back to q = xu - 128 and re-add 128·Σw afterwards.
            var flip = Vector128.Create((byte)0x80);
            Vector128<int> a = default;
            Vector128<int> wsum = default;
            var ones = Vector128.Create((sbyte)1);
            for (int p = 0; p < k; p += 16)
            {
                var q = AdvSimd.Xor(AdvSimd.LoadVector128(x + p), flip).AsSByte();
                var b = AdvSimd.LoadVector128(w + p);
                a = Dp.DotProduct(a, q, b);
                wsum = Dp.DotProduct(wsum, ones, b);
            }
            return Vector128.Sum(a) + 128 * Vector128.Sum(wsum);
        }
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<int> a = default;
            for (int p = 0; p < k; p += 16)
            {
                var (xl, xh) = Vector128.Widen(Vector128.Load(x + p));
                var (wl, wh) = Vector128.Widen(Vector128.Load(w + p));
                // 255·128 overflows int16, so widen both operands to int32 before multiplying.
                var (xll, xlh) = Vector128.Widen(xl.AsInt16());
                var (xhl, xhh) = Vector128.Widen(xh.AsInt16());
                var (wll, wlh) = Vector128.Widen(wl);
                var (whl, whh) = Vector128.Widen(wh);
                a += xll * wll + xlh * wlh + xhl * whl + xhh * whh;
            }
            return Vector128.Sum(a);
        }
        int s = 0;
        for (int p = 0; p < k; p++)
        {
            s += x[p] * w[p];
        }
        return s;
    }
}
