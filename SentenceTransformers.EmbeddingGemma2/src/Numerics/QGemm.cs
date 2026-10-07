using System.Runtime.InteropServices;
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

    /// <summary>XNNPACK <c>xnn_f32_qd8_asymmetric_quantization_params</c> + <c>f32-qs8-vcvt</c>: the quantization
    /// multiplier is <c>255 / (max − min)</c>, the zero point is derived from <c>min · multiplier</c>, and the
    /// dequantization scale used by the GEMM is <c>1 / multiplier</c> (all in float, exactly as XNNPACK rounds them).</summary>
    internal static void QuantizeRow(ReadOnlySpan<float> x, Span<byte> dst, out float scale, out int zeroPoint)
    {
        RowMinMax(x, out float rowMin, out float rowMax);
        float min = MathF.Min(0f, rowMin);
        float max = MathF.Max(0f, rowMax);
        const float QMin = -128f, QMax = 127f;
        float multiplier = min == max ? 1f : (QMax - QMin) / (max - min);
        float descaledMin = min * multiplier;
        float descaledMax = max * multiplier;
        float zp = (QMin + descaledMin) + (QMax + descaledMax) > 0 ? QMin - descaledMin : QMax - descaledMax;
        zp = Math.Clamp(zp, QMin, QMax);
        zeroPoint = (int)MathF.Round(zp, MidpointRounding.ToEven);
        scale = 1f / multiplier;
        QuantizeRowWithMultiplier(x, dst, multiplier, zeroPoint);
    }

    /// <summary>Row minimum and maximum with native <c>minps</c>/<c>maxps</c> (falling back to
    /// <see cref="TensorPrimitives"/> when the row holds a NaN). Only the sign of a zero extreme can differ, and
    /// the quantization parameters do not depend on it.</summary>
    private static void RowMinMax(ReadOnlySpan<float> x, out float min, out float max)
    {
        int n = x.Length, i = 0;
        min = float.PositiveInfinity;
        max = float.NegativeInfinity;
        ref float r = ref MemoryMarshal.GetReference(x);
        bool nan = false;
        if (Simd.Use512 && n >= 16)
        {
            var lo = Vector512.Create(float.PositiveInfinity);
            var hi = Vector512.Create(float.NegativeInfinity);
            var bad = Vector512<float>.Zero;
            for (; i + 16 <= n; i += 16)
            {
                var v = Vector512.LoadUnsafe(ref r, (nuint)i);
                lo = Vector512.MinNative(lo, v);
                hi = Vector512.MaxNative(hi, v);
                bad |= ~Vector512.Equals(v, v);
            }
            nan = bad != Vector512<float>.Zero;
            for (int k = 0; k < 16; k++)
            {
                min = MathF.Min(min, lo.GetElement(k));
                max = MathF.Max(max, hi.GetElement(k));
            }
        }
        for (; i < n; i++)
        {
            float v = Unsafe.Add(ref r, i);
            nan |= float.IsNaN(v);
            min = MathF.Min(min, v);
            max = MathF.Max(max, v);
        }
        if (nan)
        {
            min = TensorPrimitives.Min(x);
            max = TensorPrimitives.Max(x);
        }
    }

    /// <summary>Static quantization (<c>QUANTIZE</c> op): <c>q = clamp(rint(x · (1/scale)) + zp)</c>, stored offset by +128.</summary>
    internal static void QuantizeRow(ReadOnlySpan<float> x, Span<byte> dst, float scale, int zeroPoint)
        => QuantizeRowWithMultiplier(x, dst, 1f / scale, zeroPoint);

    /// <summary><c>f32-qs8-vcvt</c> (avx512skx / avx2): <c>q = sat8(sat16(rint(x · multiplier)) + zp)</c> - the
    /// product is rounded half-to-even <i>before</i> the zero point is added - stored offset by +128.</summary>
    private static void QuantizeRowWithMultiplier(ReadOnlySpan<float> x, Span<byte> dst, float multiplier, int zeroPoint)
    {
        int i = 0;
        dst = dst.Slice(0, x.Length);
        if (Avx2.IsSupported && x.Length >= 32)
        {
            ref float xr = ref MemoryMarshal.GetReference(x);
            ref byte dr = ref MemoryMarshal.GetReference(dst);
            var vmul = Vector256.Create(multiplier);
            var vzp = Vector256.Create((short)zeroPoint);
            var flip = Vector256.Create((byte)0x80);
            // packs interleave 128-bit lanes; this permutation restores element order.
            var order = Vector256.Create(0, 4, 1, 5, 2, 6, 3, 7);
            for (; i + 32 <= x.Length; i += 32)
            {
                // vcvtps2dq rounds half to even; the saturating packs/adds are XNNPACK's clamps.
                var a = Avx.ConvertToVector256Int32(Avx.Multiply(Vector256.LoadUnsafe(ref xr, (nuint)i), vmul));
                var b = Avx.ConvertToVector256Int32(Avx.Multiply(Vector256.LoadUnsafe(ref xr, (nuint)i + 8), vmul));
                var c = Avx.ConvertToVector256Int32(Avx.Multiply(Vector256.LoadUnsafe(ref xr, (nuint)i + 16), vmul));
                var d = Avx.ConvertToVector256Int32(Avx.Multiply(Vector256.LoadUnsafe(ref xr, (nuint)i + 24), vmul));
                var ab = Avx2.AddSaturate(Avx2.PackSignedSaturate(a, b), vzp);
                var cd = Avx2.AddSaturate(Avx2.PackSignedSaturate(c, d), vzp);
                var bytes = Avx2.Xor(Avx2.PackSignedSaturate(ab, cd).AsByte(), flip);
                Avx2.PermuteVar8x32(bytes.AsInt32(), order).AsByte().StoreUnsafe(ref dr, (nuint)i);
            }
        }
        for (; i < x.Length; i++)
        {
            float v = Math.Clamp(MathF.Round(x[i] * multiplier, MidpointRounding.ToEven), short.MinValue, short.MaxValue);
            dst[i] = (byte)(Math.Clamp((int)v + zeroPoint, -128, 127) + 128);
        }
    }
}

/// <summary>
/// <c>Y[N, M] = dequant(X_q8[N, K]) · W[M, K]^T (+ bias)</c> with integer accumulation: dynamically
/// quantized activations against int8 (or int4-range) per-channel weights, the computation XNNPACK's
/// <c>qd8-f32-qc4w/qc8w</c> GEMMs perform for LiteRT. For each output
/// <c>y = sx·sw·(Σ_k xu·w − (128 + zx)·Σ_k w)</c> where <c>xu = q + 128</c> is the unsigned activation,
/// so the integer part is exact and only the final scaling is floating point.
/// Kernels: packed INT4/INT2-range weights run as XNNPACK-style broadcast GEMMs over 32-channel panels
/// (AVX-512BW / AVX2 <c>pmaddubsw</c>, no horizontal reductions); 8-bit weights as row dot products with
/// AVX-VNNI (<c>vpdpbusd</c>), AVX-512BW/AVX2 (<c>pmaddubsw</c> / widening <c>pmaddwd</c>), ARM <c>sdot</c>, or a
/// portable fallback.
/// </summary>
internal static class QGemm
{
    private const int TileN = 2;   // activation rows per micro-tile
    private const int TileM = 4;   // weight rows per micro-tile
    private const int BlockN = 32;
    private const int BlockM = 64;
    private const int UnpackRows = 16;   // packed weights: rows expanded per pass (16 × K bytes, L1-resident)
    private const int UnpackWholeMinRows = 64;   // from this many activation rows, expand packed weights once up front

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

    /// <summary>
    /// GeGLU fused into the up projection: <c>gate[i] = gelu(gate[i]) · (X·Wᵀ)[i]</c>, i.e. <see cref="Multiply"/>
    /// followed by <see cref="Ops.GeluMul(float[], float[], int, ParallelOptions)"/> with the same per-element
    /// arithmetic, but computed in the GEMM epilogue so the product never round-trips through memory.
    /// </summary>
    public static void MultiplyGeluGated(QuantizedActivations x, QuantizedMatrix w, float[] gate, int ldy, ParallelOptions po = null)
    {
        int dop = po?.MaxDegreeOfParallelism ?? 1;
        if (dop == -1)
        {
            dop = Environment.ProcessorCount;
        }
        if (w.HasPanels && UseAvx2 && w.Bias is null && ldy == w.Rows)
        {
            using var _ = Profiler.Measure("qgemm");
            MultiplyPanels(x, w, gate, ldy, dop, po, null, geluGate: true);
            return;
        }
        var up = System.Buffers.ArrayPool<float>.Shared.Rent(x.Rows * ldy);
        try
        {
            Multiply(x, w, up, ldy, po);
            Ops.GeluMul(gate, up, x.Rows * ldy, po);
        }
        finally
        {
            System.Buffers.ArrayPool<float>.Shared.Return(up);
        }
    }

    public static void Multiply(QuantizedActivations x, QuantizedMatrix w, float[] y, int ldy, ParallelOptions po = null, Requantization requant = null)
    {
        using var _ = Profiler.Measure("qgemm");
        using var __ = Profiler.Enabled ? Profiler.Measure($"qg {x.Rows}x{x.Cols}x{w.Rows}{(requant is null ? "" : " rq")}") : default;
        if (x.Cols != w.Cols)
        {
            throw new ArgumentException($"Inner dimensions differ: {x.Cols} vs {w.Cols}.");
        }
        int n = x.Rows, m = w.Rows;
        CheckOutput(y, n, m, ldy);
        int nBlocks = (n + BlockN - 1) / BlockN;
        int mBlocks = (m + BlockM - 1) / BlockM;
        int blocks = nBlocks * mBlocks;
        int dop = po?.MaxDegreeOfParallelism ?? 1;
        if (dop == -1)
        {
            dop = Environment.ProcessorCount;
        }

        if (w.HasPanels && UseAvx2)
        {
            MultiplyPanels(x, w, y, ldy, dop, po, requant);
            return;
        }
        if (w.Bits == 8 || n >= UnpackWholeMinRows)
        {
            // Byte weights, or packed weights against enough activation rows that expanding the whole matrix
            // once (in parallel) costs little: the blocked kernel then reads them as bytes.
            sbyte[] unpacked = null;
            try
            {
                if (w.Bits != 8)
                {
                    unpacked = System.Buffers.ArrayPool<sbyte>.Shared.Rent(m * w.Stride);
                    WorkerPool.For(mBlocks, dop <= 1 ? 1 : dop, mb =>
                    {
                        int r0 = mb * BlockM, r1 = Math.Min(m, r0 + BlockM);
                        w.UnpackRows(r0, r1, unpacked.AsSpan(r0 * w.Stride, (r1 - r0) * w.Stride));
                    });
                }
                var wData = w.Bits == 8 ? w.Data : unpacked;
                if (dop <= 1 || blocks == 1)
                {
                    for (int b = 0; b < blocks; b++)
                    {
                        RunBlock(x, w, wData, y, ldy, b / mBlocks, b % mBlocks, requant);
                    }
                }
                else
                {
                    WorkerPool.For(blocks, po, b => RunBlock(x, w, wData, y, ldy, b / mBlocks, b % mBlocks, requant));
                }
            }
            finally
            {
                if (unpacked is not null)
                {
                    System.Buffers.ArrayPool<sbyte>.Shared.Return(unpacked);
                }
            }
            return;
        }
        // Packed (4/2-bit) weights against few activation rows (memory-bound): each work item streams one
        // block of weight rows, expanding it into a cache-resident scratch buffer a few rows at a time. N is
        // split into groups only when there are too few weight blocks to keep every thread busy.
        int groups = dop <= 1 ? 1 : Math.Max(1, Math.Min(nBlocks, (2 * dop + mBlocks - 1) / mBlocks));
        int items = mBlocks * groups;
        void Item(int it)
        {
            int mb = it / groups, g = it % groups;
            int nb0 = g * nBlocks / groups, nb1 = (g + 1) * nBlocks / groups;
            if (nb0 >= nb1)
            {
                return;
            }
            int m0 = mb * BlockM, m1 = Math.Min(m, m0 + BlockM);
            int n0 = nb0 * BlockN, n1 = Math.Min(n, nb1 * BlockN);
            // Expand UnpackRows weight rows at a time so they stay L1-resident while every activation row of
            // this item streams past them.
            var s = AlignedScratch<sbyte>.Get(UnpackRows * w.Stride);
            for (int r0 = m0; r0 < m1; r0 += UnpackRows)
            {
                int r1 = Math.Min(m1, r0 + UnpackRows);
                w.UnpackRows(r0, r1, s);
                RunRange(x, w, s, y, ldy, n0, n1, r0, r1, requant);
            }
        }
        if (dop <= 1 || items == 1)
        {
            for (int it = 0; it < items; it++)
            {
                Item(it);
            }
        }
        else
        {
            WorkerPool.For(items, po, Item);
        }
    }

    /// <summary>The kernels index <paramref name="y"/> without bounds checks, so its extent is validated once here.</summary>
    private static void CheckOutput(float[] y, int n, int m, int ldy)
    {
        ArgumentNullException.ThrowIfNull(y);
        if (n > 0 && (ldy < m || y.Length < (long)(n - 1) * ldy + m))
        {
            throw new ArgumentException($"Output of {y.Length} floats is too small for {n} x {m} (row stride {ldy}).", nameof(y));
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Broadcast ("panel") kernels for packed INT4/INT2-range weights
    // ---------------------------------------------------------------------------------------------

    /// <summary>
    /// <c>Y = X · Wᵀ</c> for packed low-bit weights, XNNPACK-style: weights are expanded one 32-channel panel at a
    /// time (<c>[K/4][32][4]</c>), four activation bytes are broadcast to every lane, and <c>pmaddubsw</c> leaves each
    /// output channel in its own lane - no horizontal reductions. Pair sums are at most 255·8·2 = 4080, so eight
    /// 4-byte steps accumulate in int16 before widening to int32. The integer results equal the row kernels'
    /// exactly, and the epilogue performs the same per-element operations.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void MultiplyPanels(QuantizedActivations x, QuantizedMatrix w, float[] y, int ldy, int dop, ParallelOptions po, Requantization rq, bool geluGate = false)
    {
        int n = x.Rows, m = w.Rows, panels = w.Panels;
        CheckOutput(y, n, m, ldy);
        int mr = UseAvx512 ? 6 : 3;
        int tiles = (n + mr - 1) / mr;
        int groups = dop <= 1 ? 1 : Math.Max(1, Math.Min(tiles, (2 * dop + panels - 1) / panels));
        int items = panels * groups;
        bool wide = w.PanelData is not null;   // 8-bit: clamped dense panels + sparse residuals
        var xData = x.Data;
        int xStride = x.Stride;
        var epi = new Epilogue(w, x, rq, y, ldy, geluGate);
        void Item(int it)
        {
            int p = it / groups, g = it % groups;
            int t0 = (int)((long)g * tiles / groups), t1 = (int)((long)(g + 1) * tiles / groups);
            if (t0 >= t1)
            {
                return;
            }
            Span<sbyte> panelSpan;
            if (wide)
            {
                panelSpan = w.PanelData.AsSpan(p * w.PanelBytes, w.PanelBytes);
            }
            else
            {
                panelSpan = AlignedScratch<sbyte>.Get(w.PanelBytes);
                w.UnpackPanels(p, p + 1, panelSpan);
            }
            ref sbyte panel = ref MemoryMarshal.GetReference(panelSpan);
            Span<int> accSpan = stackalloc int[6 * QuantizedMatrix.PanelWidth];
            ref int acc = ref MemoryMarshal.GetReference(accSpan);
            ref byte xb = ref MemoryMarshal.GetArrayDataReference(xData);
            int col0 = p * QuantizedMatrix.PanelWidth;
            for (int t = t0; t < t1; t++)
            {
                // A ragged last tile repeats its final row (the extra results are not stored).
                int r0 = t * mr, valid = Math.Min(mr, n - r0), last = r0 + valid - 1;
                ref byte x0 = ref Unsafe.Add(ref xb, (nint)r0 * xStride);
                ref byte x1 = ref Unsafe.Add(ref xb, (nint)Math.Min(r0 + 1, last) * xStride);
                ref byte x2 = ref Unsafe.Add(ref xb, (nint)Math.Min(r0 + 2, last) * xStride);
                if (UseAvx512)
                {
                    ref byte x3 = ref Unsafe.Add(ref xb, (nint)Math.Min(r0 + 3, last) * xStride);
                    ref byte x4 = ref Unsafe.Add(ref xb, (nint)Math.Min(r0 + 4, last) * xStride);
                    ref byte x5 = ref Unsafe.Add(ref xb, (nint)Math.Min(r0 + 5, last) * xStride);
                    if (wide)
                    {
                        PanelKernel6x32Wide(ref x0, ref x1, ref x2, ref x3, ref x4, ref x5, ref panel, w.Stride, ref acc);
                        AddOutliers(w, xData, xStride, r0, valid, col0, 32, accSpan);
                    }
                    else
                    {
                        PanelKernel6x32(ref x0, ref x1, ref x2, ref x3, ref x4, ref x5, ref panel, w.Stride, ref acc);
                    }
                    for (int i = 0; i < valid; i++)
                    {
                        epi.Row(r0 + i, col0, ref Unsafe.Add(ref acc, i * 32), 32);
                    }
                }
                else
                {
                    for (int half = 0; half < 2; half++)
                    {
                        ref sbyte wh = ref Unsafe.Add(ref panel, half * 64);
                        if (wide)
                        {
                            PanelKernel3x16Wide(ref x0, ref x1, ref x2, ref wh, w.Stride, ref acc);
                            AddOutliers(w, xData, xStride, r0, valid, col0 + half * 16, 16, accSpan);
                        }
                        else
                        {
                            PanelKernel3x16(ref x0, ref x1, ref x2, ref wh, w.Stride, ref acc);
                        }
                        for (int i = 0; i < valid; i++)
                        {
                            epi.Row(r0 + i, col0 + half * 16, ref Unsafe.Add(ref acc, i * 16), 16);
                        }
                    }
                }
            }
        }
        if (dop <= 1 || items == 1)
        {
            for (int it = 0; it < items; it++)
            {
                Item(it);
            }
        }
        else
        {
            WorkerPool.For(items, dop, Item);
        }
    }

    /// <summary>6 activation rows × one 32-channel panel (AVX-512BW); <paramref name="acc"/> receives [6][32] int32
    /// sums of <c>xu · w</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PanelKernel6x32(ref byte x0, ref byte x1, ref byte x2, ref byte x3, ref byte x4, ref byte x5, ref sbyte wp, int k, ref int acc)
    {
        Vector512<int> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default;
        Vector512<int> c30 = default, c31 = default, c40 = default, c41 = default, c50 = default, c51 = default;
        var ones = Vector512.Create((short)1);
        for (int k0 = 0; k0 < k; k0 += 32)
        {
            Vector512<short> s00 = default, s01 = default, s10 = default, s11 = default, s20 = default, s21 = default;
            Vector512<short> s30 = default, s31 = default, s40 = default, s41 = default, s50 = default, s51 = default;
            for (int p = k0; p < k0 + 32; p += 4)
            {
                var w0 = Vector512.LoadUnsafe(ref wp);
                var w1 = Vector512.LoadUnsafe(ref wp, 64);
                wp = ref Unsafe.Add(ref wp, 128);
                var b = Broadcast512(ref x0, p);
                s00 = Avx512BW.Add(s00, Avx512BW.MultiplyAddAdjacent(b, w0)); s01 = Avx512BW.Add(s01, Avx512BW.MultiplyAddAdjacent(b, w1));
                b = Broadcast512(ref x1, p);
                s10 = Avx512BW.Add(s10, Avx512BW.MultiplyAddAdjacent(b, w0)); s11 = Avx512BW.Add(s11, Avx512BW.MultiplyAddAdjacent(b, w1));
                b = Broadcast512(ref x2, p);
                s20 = Avx512BW.Add(s20, Avx512BW.MultiplyAddAdjacent(b, w0)); s21 = Avx512BW.Add(s21, Avx512BW.MultiplyAddAdjacent(b, w1));
                b = Broadcast512(ref x3, p);
                s30 = Avx512BW.Add(s30, Avx512BW.MultiplyAddAdjacent(b, w0)); s31 = Avx512BW.Add(s31, Avx512BW.MultiplyAddAdjacent(b, w1));
                b = Broadcast512(ref x4, p);
                s40 = Avx512BW.Add(s40, Avx512BW.MultiplyAddAdjacent(b, w0)); s41 = Avx512BW.Add(s41, Avx512BW.MultiplyAddAdjacent(b, w1));
                b = Broadcast512(ref x5, p);
                s50 = Avx512BW.Add(s50, Avx512BW.MultiplyAddAdjacent(b, w0)); s51 = Avx512BW.Add(s51, Avx512BW.MultiplyAddAdjacent(b, w1));
            }
            c00 += Avx512BW.MultiplyAddAdjacent(s00, ones); c01 += Avx512BW.MultiplyAddAdjacent(s01, ones);
            c10 += Avx512BW.MultiplyAddAdjacent(s10, ones); c11 += Avx512BW.MultiplyAddAdjacent(s11, ones);
            c20 += Avx512BW.MultiplyAddAdjacent(s20, ones); c21 += Avx512BW.MultiplyAddAdjacent(s21, ones);
            c30 += Avx512BW.MultiplyAddAdjacent(s30, ones); c31 += Avx512BW.MultiplyAddAdjacent(s31, ones);
            c40 += Avx512BW.MultiplyAddAdjacent(s40, ones); c41 += Avx512BW.MultiplyAddAdjacent(s41, ones);
            c50 += Avx512BW.MultiplyAddAdjacent(s50, ones); c51 += Avx512BW.MultiplyAddAdjacent(s51, ones);
        }
        c00.StoreUnsafe(ref acc); c01.StoreUnsafe(ref acc, 16); c10.StoreUnsafe(ref acc, 32); c11.StoreUnsafe(ref acc, 48);
        c20.StoreUnsafe(ref acc, 64); c21.StoreUnsafe(ref acc, 80); c30.StoreUnsafe(ref acc, 96); c31.StoreUnsafe(ref acc, 112);
        c40.StoreUnsafe(ref acc, 128); c41.StoreUnsafe(ref acc, 144); c50.StoreUnsafe(ref acc, 160); c51.StoreUnsafe(ref acc, 176);
    }

    /// <summary>The four activation bytes at <paramref name="x"/> + <paramref name="p"/> broadcast to every 32-bit lane.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<byte> Broadcast512(ref byte x, int p) => Vector512.Create(Unsafe.ReadUnaligned<int>(ref Unsafe.Add(ref x, p))).AsByte();

    /// <inheritdoc cref="Broadcast512"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Broadcast256(ref byte x, int p) => Vector256.Create(Unsafe.ReadUnaligned<int>(ref Unsafe.Add(ref x, p))).AsByte();

    /// <summary>3 activation rows × 16 channels (half a panel, starting at <paramref name="wp"/>) with AVX2;
    /// <paramref name="acc"/> receives [3][16] int32 sums.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PanelKernel3x16(ref byte x0, ref byte x1, ref byte x2, ref sbyte wp, int k, ref int acc)
    {
        Vector256<int> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default;
        var ones = Vector256.Create((short)1);
        for (int k0 = 0; k0 < k; k0 += 32)
        {
            Vector256<short> s00 = default, s01 = default, s10 = default, s11 = default, s20 = default, s21 = default;
            for (int p = k0; p < k0 + 32; p += 4)
            {
                var w0 = Vector256.LoadUnsafe(ref wp);
                var w1 = Vector256.LoadUnsafe(ref wp, 32);
                wp = ref Unsafe.Add(ref wp, 128);
                var b = Broadcast256(ref x0, p);
                s00 = Avx2.Add(s00, Avx2.MultiplyAddAdjacent(b, w0)); s01 = Avx2.Add(s01, Avx2.MultiplyAddAdjacent(b, w1));
                b = Broadcast256(ref x1, p);
                s10 = Avx2.Add(s10, Avx2.MultiplyAddAdjacent(b, w0)); s11 = Avx2.Add(s11, Avx2.MultiplyAddAdjacent(b, w1));
                b = Broadcast256(ref x2, p);
                s20 = Avx2.Add(s20, Avx2.MultiplyAddAdjacent(b, w0)); s21 = Avx2.Add(s21, Avx2.MultiplyAddAdjacent(b, w1));
            }
            c00 += Avx2.MultiplyAddAdjacent(s00, ones); c01 += Avx2.MultiplyAddAdjacent(s01, ones);
            c10 += Avx2.MultiplyAddAdjacent(s10, ones); c11 += Avx2.MultiplyAddAdjacent(s11, ones);
            c20 += Avx2.MultiplyAddAdjacent(s20, ones); c21 += Avx2.MultiplyAddAdjacent(s21, ones);
        }
        c00.StoreUnsafe(ref acc); c01.StoreUnsafe(ref acc, 8); c10.StoreUnsafe(ref acc, 16); c11.StoreUnsafe(ref acc, 24);
        c20.StoreUnsafe(ref acc, 32); c21.StoreUnsafe(ref acc, 40);
    }

    /// <summary>Adds the sparse residuals of 8-bit weights (beyond the ±64 dense part) for <paramref name="rows"/>
    /// activation rows from <paramref name="r0"/> and channels <paramref name="col0"/>.. (+<paramref name="count"/>);
    /// <paramref name="acc"/> is [rows][count] - exact integer arithmetic.</summary>
    private static void AddOutliers(QuantizedMatrix w, ReadOnlySpan<byte> x, int ldx, int r0, int rows, int col0, int count, Span<int> acc)
    {
        var start = w.OutlierStart;
        var ks = w.OutlierK;
        var vs = w.OutlierValue;
        int end = Math.Min(count, w.Rows - col0);
        for (int c = 0; c < end; c++)
        {
            for (int j = start[col0 + c]; j < start[col0 + c + 1]; j++)
            {
                int k = ks[j], v = vs[j];
                for (int i = 0; i < rows; i++)
                {
                    acc[i * count + c] += x[(r0 + i) * ldx + k] * v;
                }
            }
        }
    }

    /// <summary><see cref="PanelKernel6x32"/> for 8-bit weights clamped to ±64: pair sums reach 255·64·2 = 32640, so
    /// every step widens to int32 right away.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PanelKernel6x32Wide(ref byte x0, ref byte x1, ref byte x2, ref byte x3, ref byte x4, ref byte x5, ref sbyte wp, int k, ref int acc)
    {
        Vector512<int> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default;
        Vector512<int> c30 = default, c31 = default, c40 = default, c41 = default, c50 = default, c51 = default;
        var ones = Vector512.Create((short)1);
        for (int p = 0; p < k; p += 4)
        {
            var w0 = Vector512.LoadUnsafe(ref wp);
            var w1 = Vector512.LoadUnsafe(ref wp, 64);
            wp = ref Unsafe.Add(ref wp, 128);
            var b = Broadcast512(ref x0, p);
            c00 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w0), ones); c01 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w1), ones);
            b = Broadcast512(ref x1, p);
            c10 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w0), ones); c11 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w1), ones);
            b = Broadcast512(ref x2, p);
            c20 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w0), ones); c21 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w1), ones);
            b = Broadcast512(ref x3, p);
            c30 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w0), ones); c31 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w1), ones);
            b = Broadcast512(ref x4, p);
            c40 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w0), ones); c41 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w1), ones);
            b = Broadcast512(ref x5, p);
            c50 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w0), ones); c51 += Avx512BW.MultiplyAddAdjacent(Avx512BW.MultiplyAddAdjacent(b, w1), ones);
        }
        c00.StoreUnsafe(ref acc); c01.StoreUnsafe(ref acc, 16); c10.StoreUnsafe(ref acc, 32); c11.StoreUnsafe(ref acc, 48);
        c20.StoreUnsafe(ref acc, 64); c21.StoreUnsafe(ref acc, 80); c30.StoreUnsafe(ref acc, 96); c31.StoreUnsafe(ref acc, 112);
        c40.StoreUnsafe(ref acc, 128); c41.StoreUnsafe(ref acc, 144); c50.StoreUnsafe(ref acc, 160); c51.StoreUnsafe(ref acc, 176);
    }

    /// <summary><see cref="PanelKernel3x16"/> for 8-bit weights clamped to ±64 (widening every step).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void PanelKernel3x16Wide(ref byte x0, ref byte x1, ref byte x2, ref sbyte wp, int k, ref int acc)
    {
        Vector256<int> c00 = default, c01 = default, c10 = default, c11 = default, c20 = default, c21 = default;
        var ones = Vector256.Create((short)1);
        for (int p = 0; p < k; p += 4)
        {
            var w0 = Vector256.LoadUnsafe(ref wp);
            var w1 = Vector256.LoadUnsafe(ref wp, 32);
            wp = ref Unsafe.Add(ref wp, 128);
            var b = Broadcast256(ref x0, p);
            c00 += Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(b, w0), ones); c01 += Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(b, w1), ones);
            b = Broadcast256(ref x1, p);
            c10 += Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(b, w0), ones); c11 += Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(b, w1), ones);
            b = Broadcast256(ref x2, p);
            c20 += Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(b, w0), ones); c21 += Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(b, w1), ones);
        }
        c00.StoreUnsafe(ref acc); c01.StoreUnsafe(ref acc, 8); c10.StoreUnsafe(ref acc, 16); c11.StoreUnsafe(ref acc, 24);
        c20.StoreUnsafe(ref acc, 32); c21.StoreUnsafe(ref acc, 40);
    }

    /// <summary>
    /// The panel kernels' epilogue: the arithmetic of <see cref="Store"/>, lane for lane (16, 4 or 1 channels at a
    /// time), so the results are identical. The arrays are checked against the GEMM's dimensions when the GEMM
    /// starts, and indexed through refs here.
    /// </summary>
    private readonly struct Epilogue
    {
        private readonly int[] _rowSum, _xZero;
        private readonly float[] _wScale, _bias, _rqScale, _xScale, _y;
        private readonly float _rqOut;
        private readonly int _ldy, _channels;
        /// <summary>Store <c>gelu(y) · value</c> over the existing contents of Y (GeGLU) instead of the value.</summary>
        private readonly bool _geluGate;

        public Epilogue(QuantizedMatrix w, QuantizedActivations x, Requantization rq, float[] y, int ldy, bool geluGate)
        {
            _rowSum = w.RowSum;
            _wScale = w.Scale;
            _bias = w.Bias;
            _rqScale = rq?.Scale;
            _rqOut = rq?.OutputScale ?? 0f;
            _xZero = x.ZeroPoint;
            _xScale = x.Scale;
            _y = y;
            _ldy = ldy;
            _channels = w.Rows;
            _geluGate = geluGate;
            if (_rowSum.Length < _channels || _wScale.Length < _channels || (_bias is not null && _bias.Length < _channels)
                || (_rqScale is not null && _rqScale.Length < _channels) || _xZero.Length < x.Rows || _xScale.Length < x.Rows)
            {
                throw new ArgumentException("Quantization parameters do not match the matrix dimensions.");
            }
        }

        /// <summary><paramref name="count"/> consecutive channels of activation row <paramref name="row"/> from
        /// <paramref name="col0"/>; channels at or past the matrix's row count are padding.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        public void Row(int row, int col0, ref int acc, int count)
        {
            int n = Math.Min(count, _channels - col0);
            int zx = 128 + _xZero[row];
            float xs = _xScale[row];
            ref float dst = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_y), (nint)row * _ldy + col0);
            ref int rs = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_rowSum), col0);
            ref float ws = ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_wScale), col0);
            bool requant = _rqScale is not null, hasBias = _bias is not null;
            ref float rqs = ref requant ? ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_rqScale), col0) : ref Unsafe.NullRef<float>();
            ref float bias = ref hasBias ? ref Unsafe.Add(ref MemoryMarshal.GetArrayDataReference(_bias), col0) : ref Unsafe.NullRef<float>();
            int c = 0;
            if (Simd.Use512)
            {
                for (; c + 16 <= n; c += 16)
                {
                    var corr = Vector512.LoadUnsafe(ref acc, (nuint)c) - Vector512.Create(zx) * Vector512.LoadUnsafe(ref rs, (nuint)c);
                    if (requant)
                    {
                        var v = Vector512.Round(Vector512.ConvertToSingle(corr) * Vector512.LoadUnsafe(ref rqs, (nuint)c));
                        var r = Vector512.ConvertToInt32(Vector512.Min(Vector512.Max(v, Vector512.Create(-128f)), Vector512.Create(127f)));
                        (Vector512.ConvertToSingle(r) * Vector512.Create(_rqOut)).StoreUnsafe(ref dst, (nuint)c);
                    }
                    else
                    {
                        var v = Vector512.ConvertToSingle(corr) * Vector512.Create(xs) * Vector512.LoadUnsafe(ref ws, (nuint)c);
                        if (hasBias)
                        {
                            v += Vector512.LoadUnsafe(ref bias, (nuint)c);
                        }
                        if (_geluGate)
                        {
                            v = Xnn.Gelu512(Vector512.LoadUnsafe(ref dst, (nuint)c)) * v;
                        }
                        v.StoreUnsafe(ref dst, (nuint)c);
                    }
                }
            }
            for (; c + 4 <= n; c += 4)
            {
                var corr = Vector128.LoadUnsafe(ref acc, (nuint)c) - Vector128.Create(zx) * Vector128.LoadUnsafe(ref rs, (nuint)c);
                if (requant)
                {
                    var v = Vector128.Round(Vector128.ConvertToSingle(corr) * Vector128.LoadUnsafe(ref rqs, (nuint)c));
                    var r = Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(v, Vector128.Create(-128f)), Vector128.Create(127f)));
                    (Vector128.ConvertToSingle(r) * Vector128.Create(_rqOut)).StoreUnsafe(ref dst, (nuint)c);
                }
                else
                {
                    var v = Vector128.ConvertToSingle(corr) * Vector128.Create(xs) * Vector128.LoadUnsafe(ref ws, (nuint)c);
                    if (hasBias)
                    {
                        v += Vector128.LoadUnsafe(ref bias, (nuint)c);
                    }
                    if (_geluGate)
                    {
                        v = Xnn.Gelu(Vector128.LoadUnsafe(ref dst, (nuint)c)) * v;
                    }
                    v.StoreUnsafe(ref dst, (nuint)c);
                }
            }
            for (; c < n; c++)
            {
                int corr = Unsafe.Add(ref acc, c) - zx * Unsafe.Add(ref rs, c);
                ref float d = ref Unsafe.Add(ref dst, c);
                if (requant)
                {
                    int r = (int)Math.Clamp(MathF.Round((float)corr * Unsafe.Add(ref rqs, c), MidpointRounding.ToEven), -128f, 127f);
                    d = r * _rqOut;
                }
                else
                {
                    float v = (float)corr * xs * Unsafe.Add(ref ws, c);
                    if (hasBias)
                    {
                        v += Unsafe.Add(ref bias, c);
                    }
                    if (_geluGate)
                    {
                        v = Xnn.Gelu(Vector128.CreateScalar(d)).ToScalar() * v;
                    }
                    d = v;
                }
            }
        }
    }

    private static void RunBlock(QuantizedActivations x, QuantizedMatrix w, sbyte[] weights, float[] y, int ldy, int nb, int mb, Requantization rq)
    {
        int m0 = mb * BlockM, m1 = Math.Min(w.Rows, m0 + BlockM);
        RunRange(x, w, weights.AsSpan(m0 * w.Stride, (m1 - m0) * w.Stride), y, ldy, nb * BlockN, Math.Min(x.Rows, nb * BlockN + BlockN), m0, m1, rq);
    }

    /// <summary>Activation rows <c>n0..n1</c> × weight rows <c>m0..m1</c>; <paramref name="weights"/> holds rows
    /// <c>m0..m1</c> as bytes (row stride <see cref="QuantizedMatrix.Stride"/>).</summary>
    private static void RunRange(QuantizedActivations x, QuantizedMatrix w, ReadOnlySpan<sbyte> weights, float[] y, int ldy, int n0, int n1, int m0, int m1, Requantization rq)
    {
        int k = w.Stride;
        nint ldx = x.Stride;
        bool small = w.SmallRange;
        if (weights.Length < (m1 - m0) * k || x.Data.Length < n1 * ldx || x.Stride < k)
        {
            throw new ArgumentException("Weight or activation buffer is too small.");
        }
        ref sbyte wBase = ref MemoryMarshal.GetReference(weights);   // row m0
        ref byte xBase = ref MemoryMarshal.GetArrayDataReference(x.Data);
        ref float yr = ref MemoryMarshal.GetArrayDataReference(y);
        Span<int> accSpan = stackalloc int[16];
        ref int acc = ref MemoryMarshal.GetReference(accSpan);
        int i = n0;
        if (UseAvx512)
        {
            // 4-row tiles; a ragged last tile repeats its final row and only stores the valid ones.
            for (; i < n1; i += 4)
            {
                int valid = Math.Min(4, n1 - i);
                ref byte x0 = ref Unsafe.Add(ref xBase, i * ldx);
                ref byte x1 = ref Unsafe.Add(ref xBase, (i + Math.Min(1, valid - 1)) * ldx);
                ref byte x2 = ref Unsafe.Add(ref xBase, (i + Math.Min(2, valid - 1)) * ldx);
                ref byte x3 = ref Unsafe.Add(ref xBase, (i + Math.Min(3, valid - 1)) * ldx);
                int j = m0;
                for (; j + 4 <= m1; j += 4)
                {
                    Dot4x4Avx512(ref x0, ref x1, ref x2, ref x3, ref Unsafe.Add(ref wBase, (nint)(j - m0) * k), k, small, ref acc);
                    for (int a = 0; a < valid; a++)
                    {
                        Store1x4(x, w, ref yr, ldy, i + a, j, Vector128.LoadUnsafe(ref acc, (nuint)(4 * a)), rq);
                    }
                }
                for (; j < m1; j++)
                {
                    ref sbyte wr = ref Unsafe.Add(ref wBase, (nint)(j - m0) * k);
                    for (int a = 0; a < valid; a++)
                    {
                        Store(x, w, ref yr, ldy, i + a, j, Dot1x1(ref Unsafe.Add(ref xBase, (i + a) * ldx), ref wr, k, small), rq);
                    }
                }
            }
            return;
        }
        for (; i + TileN <= n1; i += TileN)
        {
            ref byte x0 = ref Unsafe.Add(ref xBase, i * ldx);
            ref byte x1 = ref Unsafe.Add(ref x0, ldx);
            int j = m0;
            for (; j + TileM <= m1; j += TileM)
            {
                ref sbyte w0 = ref Unsafe.Add(ref wBase, (nint)(j - m0) * k);
                Dot2x4(ref x0, ref x1, ref w0, ref Unsafe.Add(ref w0, k), ref Unsafe.Add(ref w0, 2 * k), ref Unsafe.Add(ref w0, 3 * k), k, small, ref acc);
                Store1x4(x, w, ref yr, ldy, i, j, Vector128.LoadUnsafe(ref acc), rq);
                Store1x4(x, w, ref yr, ldy, i + 1, j, Vector128.LoadUnsafe(ref acc, TileM), rq);
            }
            for (; j < m1; j++)
            {
                ref sbyte wr = ref Unsafe.Add(ref wBase, (nint)(j - m0) * k);
                Store(x, w, ref yr, ldy, i, j, Dot1x1(ref x0, ref wr, k, small), rq);
                Store(x, w, ref yr, ldy, i + 1, j, Dot1x1(ref x1, ref wr, k, small), rq);
            }
        }
        for (; i < n1; i++)
        {
            ref byte xr = ref Unsafe.Add(ref xBase, i * ldx);
            int j = m0;
            for (; j + TileM <= m1; j += TileM)
            {
                ref sbyte w0 = ref Unsafe.Add(ref wBase, (nint)(j - m0) * k);
                // Reuse the 2x4 kernel with the same row twice: still 4 outputs per pass.
                Dot2x4(ref xr, ref xr, ref w0, ref Unsafe.Add(ref w0, k), ref Unsafe.Add(ref w0, 2 * k), ref Unsafe.Add(ref w0, 3 * k), k, small, ref acc);
                Store1x4(x, w, ref yr, ldy, i, j, Vector128.LoadUnsafe(ref acc), rq);
            }
            for (; j < m1; j++)
            {
                Store(x, w, ref yr, ldy, i, j, Dot1x1(ref xr, ref Unsafe.Add(ref wBase, (nint)(j - m0) * k), k, small), rq);
            }
        }
    }

    /// <summary>Epilogue for one row × 4 consecutive columns: the arithmetic of <see cref="Store"/>, lane-wise
    /// (round half to even, clamp and the integer round trip are all exact per lane, so the bits are the same).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store1x4(QuantizedActivations x, QuantizedMatrix w, ref float y, int ldy, int row, int col, Vector128<int> acc, Requantization rq)
    {
        var corr = acc - Vector128.Create(128 + x.ZeroPoint[row]) * Vector128.Create<int>(w.RowSum.AsSpan(col, 4));
        ref float dst = ref Unsafe.Add(ref y, (nint)row * ldy + col);
        if (rq is not null)
        {
            var v = Vector128.Round(Vector128.ConvertToSingle(corr) * Vector128.Create<float>(rq.Scale.AsSpan(col, 4)));
            var r = Vector128.ConvertToInt32(Vector128.Min(Vector128.Max(v, Vector128.Create(-128f)), Vector128.Create(127f)));
            (Vector128.ConvertToSingle(r) * Vector128.Create(rq.OutputScale)).StoreUnsafe(ref dst);
            return;
        }
        var s = Vector128.ConvertToSingle(corr) * Vector128.Create(x.Scale[row]) * Vector128.Create<float>(w.Scale.AsSpan(col, 4));
        if (w.Bias is not null)
        {
            s += Vector128.Create<float>(w.Bias.AsSpan(col, 4));
        }
        s.StoreUnsafe(ref dst);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Store(QuantizedActivations x, QuantizedMatrix w, ref float y, int ldy, int row, int col, int acc, Requantization rq)
    {
        // Σ (q - zx)·w = Σ xu·w − (128 + zx)·Σ w, exact in int32.
        int corrected = acc - (128 + x.ZeroPoint[row]) * w.RowSum[col];
        ref float dst = ref Unsafe.Add(ref y, (nint)row * ldy + col);
        if (rq is not null)
        {
            // Through an integer like the int8 output tensor, so a zero result dequantizes to +0 (never -0).
            int r = (int)Math.Clamp(MathF.Round((float)corrected * rq.Scale[col], MidpointRounding.ToEven), -128f, 127f);
            dst = r * rq.OutputScale;
            return;
        }
        float v = (float)corrected * x.Scale[row] * w.Scale[col];
        if (w.Bias is not null)
        {
            v += w.Bias[col];
        }
        dst = v;
    }

    /// <summary>2 activation rows × 4 weight rows of unsigned·signed byte dot products (k multiple of 64).</summary>
    private static void Dot2x4(ref byte x0, ref byte x1, ref sbyte w0, ref sbyte w1, ref sbyte w2, ref sbyte w3, int k, bool small, ref int acc)
    {
        if (UseVnni)
        {
            Vector256<int> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector256<int> a10 = default, a11 = default, a12 = default, a13 = default;
            for (nuint p = 0; p < (nuint)k; p += 32)
            {
                var u0 = Vector256.LoadUnsafe(ref x0, p);
                var u1 = Vector256.LoadUnsafe(ref x1, p);
                var b = Vector256.LoadUnsafe(ref w0, p);
                a00 = AvxVnni.MultiplyWideningAndAdd(a00, u0, b);
                a10 = AvxVnni.MultiplyWideningAndAdd(a10, u1, b);
                b = Vector256.LoadUnsafe(ref w1, p);
                a01 = AvxVnni.MultiplyWideningAndAdd(a01, u0, b);
                a11 = AvxVnni.MultiplyWideningAndAdd(a11, u1, b);
                b = Vector256.LoadUnsafe(ref w2, p);
                a02 = AvxVnni.MultiplyWideningAndAdd(a02, u0, b);
                a12 = AvxVnni.MultiplyWideningAndAdd(a12, u1, b);
                b = Vector256.LoadUnsafe(ref w3, p);
                a03 = AvxVnni.MultiplyWideningAndAdd(a03, u0, b);
                a13 = AvxVnni.MultiplyWideningAndAdd(a13, u1, b);
            }
            Reduce8(a00, a01, a02, a03, a10, a11, a12, a13, ref acc);
            return;
        }
        if (UseAvx2 && small)
        {
            // pmaddubsw is exact here: |w| <= 8 so u8·s8 pair sums stay within int16.
            var ones = Vector256.Create((short)1);
            Vector256<int> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector256<int> a10 = default, a11 = default, a12 = default, a13 = default;
            for (nuint p = 0; p < (nuint)k; p += 32)
            {
                var u0 = Vector256.LoadUnsafe(ref x0, p);
                var u1 = Vector256.LoadUnsafe(ref x1, p);
                var b = Vector256.LoadUnsafe(ref w0, p);
                a00 = Avx2.Add(a00, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u0, b), ones));
                a10 = Avx2.Add(a10, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u1, b), ones));
                b = Vector256.LoadUnsafe(ref w1, p);
                a01 = Avx2.Add(a01, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u0, b), ones));
                a11 = Avx2.Add(a11, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u1, b), ones));
                b = Vector256.LoadUnsafe(ref w2, p);
                a02 = Avx2.Add(a02, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u0, b), ones));
                a12 = Avx2.Add(a12, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u1, b), ones));
                b = Vector256.LoadUnsafe(ref w3, p);
                a03 = Avx2.Add(a03, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u0, b), ones));
                a13 = Avx2.Add(a13, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(u1, b), ones));
            }
            Reduce8(a00, a01, a02, a03, a10, a11, a12, a13, ref acc);
            return;
        }
        if (UseAvx2)
        {
            // Full int8 weights: widen both operands to int16 and use pmaddwd (exact, no saturation).
            Vector256<int> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector256<int> a10 = default, a11 = default, a12 = default, a13 = default;
            for (nuint p = 0; p < (nuint)k; p += 16)
            {
                var u0 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref x0, p));
                var u1 = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref x1, p));
                var b = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref w0, p));
                a00 = Avx2.Add(a00, Avx2.MultiplyAddAdjacent(u0, b));
                a10 = Avx2.Add(a10, Avx2.MultiplyAddAdjacent(u1, b));
                b = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref w1, p));
                a01 = Avx2.Add(a01, Avx2.MultiplyAddAdjacent(u0, b));
                a11 = Avx2.Add(a11, Avx2.MultiplyAddAdjacent(u1, b));
                b = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref w2, p));
                a02 = Avx2.Add(a02, Avx2.MultiplyAddAdjacent(u0, b));
                a12 = Avx2.Add(a12, Avx2.MultiplyAddAdjacent(u1, b));
                b = Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref w3, p));
                a03 = Avx2.Add(a03, Avx2.MultiplyAddAdjacent(u0, b));
                a13 = Avx2.Add(a13, Avx2.MultiplyAddAdjacent(u1, b));
            }
            Reduce8(a00, a01, a02, a03, a10, a11, a12, a13, ref acc);
            return;
        }
        acc = Dot1x1(ref x0, ref w0, k, small);
        Unsafe.Add(ref acc, 1) = Dot1x1(ref x0, ref w1, k, small);
        Unsafe.Add(ref acc, 2) = Dot1x1(ref x0, ref w2, k, small);
        Unsafe.Add(ref acc, 3) = Dot1x1(ref x0, ref w3, k, small);
        Unsafe.Add(ref acc, 4) = Dot1x1(ref x1, ref w0, k, small);
        Unsafe.Add(ref acc, 5) = Dot1x1(ref x1, ref w1, k, small);
        Unsafe.Add(ref acc, 6) = Dot1x1(ref x1, ref w2, k, small);
        Unsafe.Add(ref acc, 7) = Dot1x1(ref x1, ref w3, k, small);
    }

    /// <summary>4 activation rows × 4 weight rows with 512-bit AVX-512BW (k multiple of 64).</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Dot4x4Avx512(ref byte x0, ref byte x1, ref byte x2, ref byte x3, ref sbyte w0, int k, bool small, ref int acc)
    {
        ref sbyte w1 = ref Unsafe.Add(ref w0, k);
        ref sbyte w2 = ref Unsafe.Add(ref w1, k);
        ref sbyte w3 = ref Unsafe.Add(ref w2, k);
        if (small)
        {
            Dot4x4Avx512Small(ref x0, ref x1, ref x2, ref x3, ref w0, ref w1, ref w2, ref w3, k, ref acc);
            return;
        }
        Vector512<int> a00 = default, a01 = default, a02 = default, a03 = default;
        Vector512<int> a10 = default, a11 = default, a12 = default, a13 = default;
        Vector512<int> a20 = default, a21 = default, a22 = default, a23 = default;
        Vector512<int> a30 = default, a31 = default, a32 = default, a33 = default;
        for (nuint p = 0; p < (nuint)k; p += 32)
        {
            var u0 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref x0, p));
            var u1 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref x1, p));
            var u2 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref x2, p));
            var u3 = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref x3, p));
            var b = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref w0, p));
            a00 = Avx512F.Add(a00, Avx512BW.MultiplyAddAdjacent(u0, b));
            a10 = Avx512F.Add(a10, Avx512BW.MultiplyAddAdjacent(u1, b));
            a20 = Avx512F.Add(a20, Avx512BW.MultiplyAddAdjacent(u2, b));
            a30 = Avx512F.Add(a30, Avx512BW.MultiplyAddAdjacent(u3, b));
            b = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref w1, p));
            a01 = Avx512F.Add(a01, Avx512BW.MultiplyAddAdjacent(u0, b));
            a11 = Avx512F.Add(a11, Avx512BW.MultiplyAddAdjacent(u1, b));
            a21 = Avx512F.Add(a21, Avx512BW.MultiplyAddAdjacent(u2, b));
            a31 = Avx512F.Add(a31, Avx512BW.MultiplyAddAdjacent(u3, b));
            b = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref w2, p));
            a02 = Avx512F.Add(a02, Avx512BW.MultiplyAddAdjacent(u0, b));
            a12 = Avx512F.Add(a12, Avx512BW.MultiplyAddAdjacent(u1, b));
            a22 = Avx512F.Add(a22, Avx512BW.MultiplyAddAdjacent(u2, b));
            a32 = Avx512F.Add(a32, Avx512BW.MultiplyAddAdjacent(u3, b));
            b = Avx512BW.ConvertToVector512Int16(Vector256.LoadUnsafe(ref w3, p));
            a03 = Avx512F.Add(a03, Avx512BW.MultiplyAddAdjacent(u0, b));
            a13 = Avx512F.Add(a13, Avx512BW.MultiplyAddAdjacent(u1, b));
            a23 = Avx512F.Add(a23, Avx512BW.MultiplyAddAdjacent(u2, b));
            a33 = Avx512F.Add(a33, Avx512BW.MultiplyAddAdjacent(u3, b));
        }
        HSum4(Fold(a00), Fold(a01), Fold(a02), Fold(a03)).StoreUnsafe(ref acc);
        HSum4(Fold(a10), Fold(a11), Fold(a12), Fold(a13)).StoreUnsafe(ref acc, 4);
        HSum4(Fold(a20), Fold(a21), Fold(a22), Fold(a23)).StoreUnsafe(ref acc, 8);
        HSum4(Fold(a30), Fold(a31), Fold(a32), Fold(a33)).StoreUnsafe(ref acc, 12);
    }

    /// <summary>
    /// Int4-range weights: <c>pmaddubsw</c> pair sums are bounded by 255·8·2 = 4080, so eight 64-byte steps
    /// are accumulated in int16 before widening (2 instead of 3 ALU ops per 64 MACs). Only the 16 int16
    /// tiles stay in registers; int32 totals live in a stack buffer touched once per 512-byte block.
    /// </summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Dot4x4Avx512Small(ref byte x0, ref byte x1, ref byte x2, ref byte x3, ref sbyte w0, ref sbyte w1, ref sbyte w2, ref sbyte w3, int k, ref int acc)
    {
        var ones = Vector512.Create((short)1);
        if (k <= 512)
        {
            Dot4x4Avx512SmallBlock(ref x0, ref x1, ref x2, ref x3, ref w0, ref w1, ref w2, ref w3, 0, k, ones, ref acc);
            return;
        }
        Span<Vector512<int>> tot = stackalloc Vector512<int>[16];
        tot.Clear();
        for (int p0 = 0; p0 < k; p0 += 512)
        {
            int pEnd = Math.Min(k, p0 + 512);
            Vector512<short> s00 = default, s01 = default, s02 = default, s03 = default;
            Vector512<short> s10 = default, s11 = default, s12 = default, s13 = default;
            Vector512<short> s20 = default, s21 = default, s22 = default, s23 = default;
            Vector512<short> s30 = default, s31 = default, s32 = default, s33 = default;
            for (nuint p = (nuint)p0; p < (nuint)pEnd; p += 64)
            {
                var u0 = Vector512.LoadUnsafe(ref x0, p);
                var u1 = Vector512.LoadUnsafe(ref x1, p);
                var u2 = Vector512.LoadUnsafe(ref x2, p);
                var u3 = Vector512.LoadUnsafe(ref x3, p);
                var b = Vector512.LoadUnsafe(ref w0, p);
                s00 = Avx512BW.Add(s00, Avx512BW.MultiplyAddAdjacent(u0, b));
                s10 = Avx512BW.Add(s10, Avx512BW.MultiplyAddAdjacent(u1, b));
                s20 = Avx512BW.Add(s20, Avx512BW.MultiplyAddAdjacent(u2, b));
                s30 = Avx512BW.Add(s30, Avx512BW.MultiplyAddAdjacent(u3, b));
                b = Vector512.LoadUnsafe(ref w1, p);
                s01 = Avx512BW.Add(s01, Avx512BW.MultiplyAddAdjacent(u0, b));
                s11 = Avx512BW.Add(s11, Avx512BW.MultiplyAddAdjacent(u1, b));
                s21 = Avx512BW.Add(s21, Avx512BW.MultiplyAddAdjacent(u2, b));
                s31 = Avx512BW.Add(s31, Avx512BW.MultiplyAddAdjacent(u3, b));
                b = Vector512.LoadUnsafe(ref w2, p);
                s02 = Avx512BW.Add(s02, Avx512BW.MultiplyAddAdjacent(u0, b));
                s12 = Avx512BW.Add(s12, Avx512BW.MultiplyAddAdjacent(u1, b));
                s22 = Avx512BW.Add(s22, Avx512BW.MultiplyAddAdjacent(u2, b));
                s32 = Avx512BW.Add(s32, Avx512BW.MultiplyAddAdjacent(u3, b));
                b = Vector512.LoadUnsafe(ref w3, p);
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
            HSum4(Fold(tot[4 * r]), Fold(tot[4 * r + 1]), Fold(tot[4 * r + 2]), Fold(tot[4 * r + 3])).StoreUnsafe(ref acc, (nuint)(4 * r));
        }
    }

    /// <summary>One ≤512-byte block of <see cref="Dot4x4Avx512Small"/>, reduced straight to 16 int sums.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Dot4x4Avx512SmallBlock(ref byte x0, ref byte x1, ref byte x2, ref byte x3, ref sbyte w0, ref sbyte w1, ref sbyte w2, ref sbyte w3, int p0, int pEnd, Vector512<short> ones, ref int acc)
    {
        Vector512<short> s00 = default, s01 = default, s02 = default, s03 = default;
        Vector512<short> s10 = default, s11 = default, s12 = default, s13 = default;
        Vector512<short> s20 = default, s21 = default, s22 = default, s23 = default;
        Vector512<short> s30 = default, s31 = default, s32 = default, s33 = default;
        for (nuint p = (nuint)p0; p < (nuint)pEnd; p += 64)
        {
            var u0 = Vector512.LoadUnsafe(ref x0, p);
            var u1 = Vector512.LoadUnsafe(ref x1, p);
            var u2 = Vector512.LoadUnsafe(ref x2, p);
            var u3 = Vector512.LoadUnsafe(ref x3, p);
            var b = Vector512.LoadUnsafe(ref w0, p);
            s00 = Avx512BW.Add(s00, Avx512BW.MultiplyAddAdjacent(u0, b));
            s10 = Avx512BW.Add(s10, Avx512BW.MultiplyAddAdjacent(u1, b));
            s20 = Avx512BW.Add(s20, Avx512BW.MultiplyAddAdjacent(u2, b));
            s30 = Avx512BW.Add(s30, Avx512BW.MultiplyAddAdjacent(u3, b));
            b = Vector512.LoadUnsafe(ref w1, p);
            s01 = Avx512BW.Add(s01, Avx512BW.MultiplyAddAdjacent(u0, b));
            s11 = Avx512BW.Add(s11, Avx512BW.MultiplyAddAdjacent(u1, b));
            s21 = Avx512BW.Add(s21, Avx512BW.MultiplyAddAdjacent(u2, b));
            s31 = Avx512BW.Add(s31, Avx512BW.MultiplyAddAdjacent(u3, b));
            b = Vector512.LoadUnsafe(ref w2, p);
            s02 = Avx512BW.Add(s02, Avx512BW.MultiplyAddAdjacent(u0, b));
            s12 = Avx512BW.Add(s12, Avx512BW.MultiplyAddAdjacent(u1, b));
            s22 = Avx512BW.Add(s22, Avx512BW.MultiplyAddAdjacent(u2, b));
            s32 = Avx512BW.Add(s32, Avx512BW.MultiplyAddAdjacent(u3, b));
            b = Vector512.LoadUnsafe(ref w3, p);
            s03 = Avx512BW.Add(s03, Avx512BW.MultiplyAddAdjacent(u0, b));
            s13 = Avx512BW.Add(s13, Avx512BW.MultiplyAddAdjacent(u1, b));
            s23 = Avx512BW.Add(s23, Avx512BW.MultiplyAddAdjacent(u2, b));
            s33 = Avx512BW.Add(s33, Avx512BW.MultiplyAddAdjacent(u3, b));
        }
        HSum4(W(s00, ones), W(s01, ones), W(s02, ones), W(s03, ones)).StoreUnsafe(ref acc);
        HSum4(W(s10, ones), W(s11, ones), W(s12, ones), W(s13, ones)).StoreUnsafe(ref acc, 4);
        HSum4(W(s20, ones), W(s21, ones), W(s22, ones), W(s23, ones)).StoreUnsafe(ref acc, 8);
        HSum4(W(s30, ones), W(s31, ones), W(s32, ones), W(s33, ones)).StoreUnsafe(ref acc, 12);
    }

    /// <summary>Widens an int16 accumulator to int32 pair sums and folds 512 -> 256 bits.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> W(Vector512<short> s, Vector512<short> ones) => Fold(Avx512BW.MultiplyAddAdjacent(s, ones));

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<int> Fold(Vector512<int> v) => Avx2.Add(v.GetLower(), v.GetUpper());

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Reduce8(Vector256<int> a00, Vector256<int> a01, Vector256<int> a02, Vector256<int> a03,
                                Vector256<int> a10, Vector256<int> a11, Vector256<int> a12, Vector256<int> a13, ref int acc)
    {
        // Pairwise horizontal adds reduce four 8-lane accumulators to one vector of four sums per row.
        HSum4(a00, a01, a02, a03).StoreUnsafe(ref acc);
        HSum4(a10, a11, a12, a13).StoreUnsafe(ref acc, 4);
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> HSum4(Vector256<int> a, Vector256<int> b, Vector256<int> c, Vector256<int> d)
    {
        var ab = Avx2.HorizontalAdd(a, b);       // a01 a23 b01 b23 | a45 a67 b45 b67
        var cd = Avx2.HorizontalAdd(c, d);
        var abcd = Avx2.HorizontalAdd(ab, cd);   // a0-3 b0-3 c0-3 d0-3 | a4-7 b4-7 c4-7 d4-7
        return Sse2.Add(abcd.GetLower(), abcd.GetUpper());
    }

    private static int Dot1x1(ref byte x, ref sbyte w, int k, bool small)
    {
        if (UseVnni)
        {
            Vector256<int> a = default;
            for (nuint p = 0; p < (nuint)k; p += 32)
            {
                a = AvxVnni.MultiplyWideningAndAdd(a, Vector256.LoadUnsafe(ref x, p), Vector256.LoadUnsafe(ref w, p));
            }
            return Vector256.Sum(a);
        }
        if (UseAvx2 && small)
        {
            var ones = Vector256.Create((short)1);
            Vector256<int> a = default;
            for (nuint p = 0; p < (nuint)k; p += 32)
            {
                a = Avx2.Add(a, Avx2.MultiplyAddAdjacent(Avx2.MultiplyAddAdjacent(Vector256.LoadUnsafe(ref x, p), Vector256.LoadUnsafe(ref w, p)), ones));
            }
            return Vector256.Sum(a);
        }
        if (UseAvx2)
        {
            Vector256<int> a = default;
            for (nuint p = 0; p < (nuint)k; p += 16)
            {
                a = Avx2.Add(a, Avx2.MultiplyAddAdjacent(Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref x, p)), Avx2.ConvertToVector256Int16(Vector128.LoadUnsafe(ref w, p))));
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
            for (nuint p = 0; p < (nuint)k; p += 16)
            {
                var q = AdvSimd.Xor(Vector128.LoadUnsafe(ref x, p), flip).AsSByte();
                var b = Vector128.LoadUnsafe(ref w, p);
                a = Dp.DotProduct(a, q, b);
                wsum = Dp.DotProduct(wsum, ones, b);
            }
            return Vector128.Sum(a) + 128 * Vector128.Sum(wsum);
        }
        if (Vector128.IsHardwareAccelerated)
        {
            Vector128<int> a = default;
            for (nuint p = 0; p < (nuint)k; p += 16)
            {
                var (xl, xh) = Vector128.Widen(Vector128.LoadUnsafe(ref x, p));
                var (wl, wh) = Vector128.Widen(Vector128.LoadUnsafe(ref w, p));
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
            s += Unsafe.Add(ref x, p) * Unsafe.Add(ref w, p);
        }
        return s;
    }
}
