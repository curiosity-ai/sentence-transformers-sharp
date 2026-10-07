using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>
/// Single-precision GEMM <c>C[M, N] = alpha · A[M, K] · B[K, N]</c> (row-major, arbitrary leading
/// dimensions) used for attention (<c>Q·Kᵀ</c>, <c>P·V</c>) and the vision/audio float layers.
/// <para>
/// Goto-style blocking: B is packed per (KC × NR) panel into a contiguous, L1-resident buffer; A rows are
/// broadcast against it by an MR × NR register tile of FMA accumulators (12 × 32 with AVX-512: 24 of the 32
/// vector registers; 6 × 16 with AVX2/FMA), i.e. 2 B loads and one broadcast per row per k step. Ragged tiles are
/// computed into a scratch tile and copied out. Platforms without 256-bit SIMD fall back to
/// <see cref="TensorPrimitives"/> row updates.
/// </para>
/// </summary>
internal static class SGemm
{
    private static readonly int MR = Avx512F.IsSupported ? 12 : 6;   // rows per register tile
    private static readonly int KC = 256;   // packed panel KC × NR floats (32 KB with AVX-512, 16 KB with AVX2)
    private const int RowChunk = 48;   // rows per parallel work item (8 register tiles)

    private static readonly bool UseAvx512 = Avx512F.IsSupported;
    private static readonly bool UseFma = Fma.IsSupported && Avx.IsSupported;
    private static readonly int NR = UseAvx512 ? 32 : 16;

    /// <summary><c>C = alpha · A·B</c>, or <c>C += A·B</c> when <paramref name="accumulate"/> is set (then
    /// <paramref name="alpha"/> must be 1). Every output is one sequential FMA chain over k starting from zero (or
    /// from C), like XNNPACK's broadcast GEMM kernels - which start from the bias, hence <paramref name="accumulate"/>.
    /// The operands are <see cref="Memory{T}"/> so that parallel work items can reach them (spans cannot cross threads).</summary>
    public static void Multiply(ReadOnlyMemory<float> a, int lda, ReadOnlyMemory<float> b, int ldb, Memory<float> c, int ldc,
                                int m, int n, int k, float alpha = 1f, ParallelOptions po = null, bool accumulate = false)
    {
        if (m == 0 || n == 0)
        {
            return;
        }
        CheckExtent(a.Length, m, k, lda, nameof(a));
        CheckExtent(b.Length, k, n, ldb, nameof(b));
        CheckExtent(c.Length, m, n, ldc, nameof(c));
        int dop = po?.MaxDegreeOfParallelism ?? 1;
        if (dop == -1)
        {
            dop = Environment.ProcessorCount;
        }
        if (!Vector256.IsHardwareAccelerated || !UseFma)
        {
            Portable(a.Span, lda, b.Span, ldb, c.Span, ldc, 0, m, n, k, alpha, accumulate);
            return;
        }
        int chunks = (m + RowChunk - 1) / RowChunk;
        if (dop <= 1 || chunks == 1)
        {
            Chunk(a.Span, lda, b.Span, ldb, c.Span, ldc, 0, m, n, k, alpha, accumulate);
        }
        else
        {
            WorkerPool.For(chunks, po, ch =>
                Chunk(a.Span, lda, b.Span, ldb, c.Span, ldc, ch * RowChunk, Math.Min(m, (ch + 1) * RowChunk), n, k, alpha, accumulate));
        }
    }

    /// <summary>The kernels index the operands without bounds checks, so a <paramref name="rows"/> ×
    /// <paramref name="cols"/> matrix with row stride <paramref name="ld"/> is validated against its buffer once, here.</summary>
    private static void CheckExtent(int length, int rows, int cols, int ld, string name)
    {
        if (rows > 0 && cols > 0 && (ld < cols || length < (long)(rows - 1) * ld + cols))
        {
            throw new ArgumentException($"Buffer of {length} floats is too small for {rows} x {cols} (row stride {ld}).", name);
        }
    }

    private static void Portable(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int m0, int m1, int n, int k, float alpha, bool accumulate)
    {
        for (int i = m0; i < m1; i++)
        {
            var crow = c.Slice(i * ldc, n);
            if (!accumulate)
            {
                crow.Clear();
            }
            for (int p = 0; p < k; p++)
            {
                float av = a[i * lda + p];
                var brow = b.Slice(p * ldb, n);
                for (int j = 0; j < n; j++)
                {
                    crow[j] = MathF.FusedMultiplyAdd(av, brow[j], crow[j]);
                }
            }
            if (alpha != 1f)
            {
                TensorPrimitives.Multiply(crow, alpha, crow);
            }
        }
    }

    /// <summary>Rows <paramref name="m0"/>..<paramref name="m1"/> of C (extents already validated). With
    /// <paramref name="prepacked"/>, B is ignored and the panels are read from there instead.</summary>
    private static void Chunk(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc, int m0, int m1, int n, int k, float alpha, bool accumulate, ReadOnlySpan<float> prepacked = default)
    {
        int nr = NR;
        bool usePrepacked = !prepacked.IsEmpty;
        var pack = usePrepacked ? default : AlignedScratch<float>.Get(KC * nr);
        Span<float> tile = stackalloc float[12 * 32];
        Span<float> aTile = stackalloc float[12 * 256];   // ragged 12-row tiles: A rows staged contiguously (KC ≤ 256)
        ref float ar = ref MemoryMarshal.GetReference(a);
        ref float cr = ref MemoryMarshal.GetReference(c);
        ref float tr = ref MemoryMarshal.GetReference(tile);
        for (int j0 = 0; j0 < n; j0 += nr)
        {
            int nc = Math.Min(nr, n - j0);
            for (int k0 = 0; k0 < k; k0 += KC)
            {
                int kc = Math.Min(KC, k - k0);
                bool first = k0 == 0 && !accumulate;
                bool last = k0 + kc >= k;
                ref float panel = ref MemoryMarshal.GetReference(pack);
                if (usePrepacked)
                {
                    panel = ref MemoryMarshal.GetReference(prepacked.Slice((j0 / nr * k + k0) * nr, kc * nr));
                }
                else
                {
                    PackB(b, ldb, k0, kc, j0, nc, nr, pack);
                }
                for (int i = m0; i < m1; i += MR)
                {
                    int rows = Math.Min(MR, m1 - i);
                    ref float a0 = ref Unsafe.Add(ref ar, (nint)i * lda + k0);
                    bool full = rows == MR && nc == nr;
                    ref float cDst = ref full ? ref Unsafe.Add(ref cr, (nint)i * ldc + j0) : ref tr;
                    int cStride = full ? ldc : nr;
                    if (!full)
                    {
                        // Ragged tile: stage the existing partial sums in the scratch tile.
                        for (int r = 0; r < MR; r++)
                        {
                            for (int cc = 0; cc < nr; cc++)
                            {
                                tile[r * nr + cc] = !first && r < rows && cc < nc ? c[(i + r) * ldc + j0 + cc] : 0f;
                            }
                        }
                    }
                    if (UseAvx512)
                    {
                        if (rows < MR)
                        {
                            // Rows past the end of a ragged tile repeat the last valid A row (results discarded).
                            for (int r = 0; r < MR; r++)
                            {
                                a.Slice((i + Math.Min(r, rows - 1)) * lda + k0, kc).CopyTo(aTile.Slice(r * kc, kc));
                            }
                            Kernel12x32(ref MemoryMarshal.GetReference(aTile), kc, ref panel, kc, ref cDst, cStride, first && full, last ? alpha : 1f);
                        }
                        else
                        {
                            Kernel12x32(ref a0, lda, ref panel, kc, ref cDst, cStride, first && full, last ? alpha : 1f);
                        }
                    }
                    else
                    {
                        // Rows past the end of a ragged tile re-read the last valid A row (results discarded).
                        ref float a1 = ref Unsafe.Add(ref a0, (nint)Math.Min(1, rows - 1) * lda);
                        ref float a2 = ref Unsafe.Add(ref a0, (nint)Math.Min(2, rows - 1) * lda);
                        ref float a3 = ref Unsafe.Add(ref a0, (nint)Math.Min(3, rows - 1) * lda);
                        ref float a4 = ref Unsafe.Add(ref a0, (nint)Math.Min(4, rows - 1) * lda);
                        ref float a5 = ref Unsafe.Add(ref a0, (nint)Math.Min(5, rows - 1) * lda);
                        Kernel6x16(ref a0, ref a1, ref a2, ref a3, ref a4, ref a5, ref panel, kc, ref cDst, cStride, first && full, last ? alpha : 1f);
                    }
                    if (!full)
                    {
                        for (int r = 0; r < rows; r++)
                        {
                            tile.Slice(r * nr, nc).CopyTo(c.Slice((i + r) * ldc + j0, nc));
                        }
                    }
                }
            }
        }
    }

    /// <summary>True when <see cref="PackB(ReadOnlySpan{float}, int, int, int, bool, float[])"/> /
    /// <see cref="MultiplyPacked"/> are available (the blocked FMA kernels); otherwise use <see cref="Multiply"/>.</summary>
    public static bool SupportsPacking => Vector256.IsHardwareAccelerated && UseFma;

    /// <summary>Floats needed by <see cref="PackB(ReadOnlySpan{float}, int, int, int, bool, float[])"/> for a
    /// <paramref name="k"/> × <paramref name="n"/> B, including 64-byte alignment slack.</summary>
    public static int PackedLength(int k, int n) => (n + NR - 1) / NR * NR * k + 16;

    /// <summary>
    /// Packs a whole <c>B[k, n]</c> once into the kernels' panel layout (<c>[n / NR][k][NR]</c>, zero padded), for
    /// reuse across many <see cref="MultiplyPacked"/> calls (attention multiplies every row block of a head by
    /// the same <c>Kᵀ</c> and <c>V</c>). With <paramref name="transposed"/>, <paramref name="b"/> holds
    /// <c>Bᵀ[n, k]</c> (row stride <paramref name="ldb"/>), e.g. the keys for <c>Q·Kᵀ</c>. Returns the offset of
    /// the panel data in <paramref name="dst"/>, chosen to be 64-byte aligned at packing time (the array is not
    /// pinned, so the kernels use unaligned loads and alignment is only a performance hint).
    /// </summary>
    public static int PackB(ReadOnlySpan<float> b, int ldb, int k, int n, bool transposed, float[] dst)
    {
        int nr = NR;
        int offset = Simd.AlignOffset(dst);
        var d = dst.AsSpan(offset, (n + nr - 1) / nr * nr * k);
        for (int j0 = 0, panel = 0; j0 < n; j0 += nr, panel++)
        {
            int nc = Math.Min(nr, n - j0);
            var pd = d.Slice(panel * k * nr, k * nr);
            if (nc < nr)
            {
                pd.Clear();
            }
            if (transposed)
            {
                for (int c = 0; c < nc; c++)
                {
                    var src = b.Slice((j0 + c) * ldb, k);
                    for (int p = 0; p < k; p++)
                    {
                        pd[p * nr + c] = src[p];
                    }
                }
            }
            else
            {
                for (int p = 0; p < k; p++)
                {
                    b.Slice(p * ldb + j0, nc).CopyTo(pd.Slice(p * nr, nc));
                }
            }
        }
        return offset;
    }

    /// <summary><see cref="Multiply"/> (single-threaded, <c>C = alpha · A·B</c>) with a B packed by
    /// <see cref="PackB(ReadOnlySpan{float}, int, int, int, bool, float[])"/>; same arithmetic, so the same bits.</summary>
    public static void MultiplyPacked(ReadOnlySpan<float> a, int lda, float[] packed, int packedOffset, Span<float> c, int ldc, int m, int n, int k, float alpha = 1f)
    {
        if (m == 0 || n == 0)
        {
            return;
        }
        CheckExtent(a.Length, m, k, lda, nameof(a));
        CheckExtent(c.Length, m, n, ldc, nameof(c));
        var panels = packed.AsSpan(packedOffset, (n + NR - 1) / NR * NR * k);
        Chunk(a, lda, default, 0, c, ldc, 0, m, n, k, alpha, accumulate: false, prepacked: panels);
    }

    /// <summary>Copies B[k0..k0+kc, j0..j0+nc] into a contiguous kc × nr panel (zero padded).</summary>
    private static void PackB(ReadOnlySpan<float> b, int ldb, int k0, int kc, int j0, int nc, int nr, Span<float> dst)
    {
        for (int p = 0; p < kc; p++)
        {
            var d = dst.Slice(p * nr, nr);
            b.Slice((k0 + p) * ldb + j0, nc).CopyTo(d);
            if (nc < nr)
            {
                d.Slice(nc).Clear();
            }
        }
    }

    /// <summary>C[12×32] (=|+=) A[12×kc]·Bpack[kc×32] (A rows <paramref name="lda"/> apart), then scaled by
    /// <paramref name="scale"/>: 24 accumulators, 2 B loads and 12 broadcasts per k step.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel12x32(ref float a, int lda, ref float bp, int kc, ref float c, int ldc, bool overwrite, float scale)
    {
        Vector512<float> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        Vector512<float> d00, d01, d10, d11, d20, d21, d30, d31, d40, d41, d50, d51;
        nuint l1 = (nuint)ldc, l2 = 2 * l1, l3 = 3 * l1, l4 = 4 * l1, l5 = 5 * l1;
        ref float d = ref Unsafe.Add(ref c, 6 * (nint)ldc);
        if (overwrite)
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector512<float>.Zero;
            d00 = d01 = d10 = d11 = d20 = d21 = d30 = d31 = d40 = d41 = d50 = d51 = Vector512<float>.Zero;
        }
        else
        {
            c00 = Vector512.LoadUnsafe(ref c); c01 = Vector512.LoadUnsafe(ref c, 16);
            c10 = Vector512.LoadUnsafe(ref c, l1); c11 = Vector512.LoadUnsafe(ref c, l1 + 16);
            c20 = Vector512.LoadUnsafe(ref c, l2); c21 = Vector512.LoadUnsafe(ref c, l2 + 16);
            c30 = Vector512.LoadUnsafe(ref c, l3); c31 = Vector512.LoadUnsafe(ref c, l3 + 16);
            c40 = Vector512.LoadUnsafe(ref c, l4); c41 = Vector512.LoadUnsafe(ref c, l4 + 16);
            c50 = Vector512.LoadUnsafe(ref c, l5); c51 = Vector512.LoadUnsafe(ref c, l5 + 16);
            d00 = Vector512.LoadUnsafe(ref d); d01 = Vector512.LoadUnsafe(ref d, 16);
            d10 = Vector512.LoadUnsafe(ref d, l1); d11 = Vector512.LoadUnsafe(ref d, l1 + 16);
            d20 = Vector512.LoadUnsafe(ref d, l2); d21 = Vector512.LoadUnsafe(ref d, l2 + 16);
            d30 = Vector512.LoadUnsafe(ref d, l3); d31 = Vector512.LoadUnsafe(ref d, l3 + 16);
            d40 = Vector512.LoadUnsafe(ref d, l4); d41 = Vector512.LoadUnsafe(ref d, l4 + 16);
            d50 = Vector512.LoadUnsafe(ref d, l5); d51 = Vector512.LoadUnsafe(ref d, l5 + 16);
        }
        ref float a6 = ref Unsafe.Add(ref a, 6 * (nint)lda);
        for (int p = 0; p < kc; p++)
        {
            var b0 = Vector512.LoadUnsafe(ref bp);
            var b1 = Vector512.LoadUnsafe(ref bp, 16);
            bp = ref Unsafe.Add(ref bp, 32);
            var x = Vector512.Create(Unsafe.Add(ref a, p));
            c00 = Avx512F.FusedMultiplyAdd(x, b0, c00); c01 = Avx512F.FusedMultiplyAdd(x, b1, c01);
            x = Vector512.Create(Unsafe.Add(ref a, lda + p));
            c10 = Avx512F.FusedMultiplyAdd(x, b0, c10); c11 = Avx512F.FusedMultiplyAdd(x, b1, c11);
            x = Vector512.Create(Unsafe.Add(ref a, 2 * lda + p));
            c20 = Avx512F.FusedMultiplyAdd(x, b0, c20); c21 = Avx512F.FusedMultiplyAdd(x, b1, c21);
            x = Vector512.Create(Unsafe.Add(ref a, 3 * lda + p));
            c30 = Avx512F.FusedMultiplyAdd(x, b0, c30); c31 = Avx512F.FusedMultiplyAdd(x, b1, c31);
            x = Vector512.Create(Unsafe.Add(ref a, 4 * lda + p));
            c40 = Avx512F.FusedMultiplyAdd(x, b0, c40); c41 = Avx512F.FusedMultiplyAdd(x, b1, c41);
            x = Vector512.Create(Unsafe.Add(ref a, 5 * lda + p));
            c50 = Avx512F.FusedMultiplyAdd(x, b0, c50); c51 = Avx512F.FusedMultiplyAdd(x, b1, c51);
            x = Vector512.Create(Unsafe.Add(ref a6, p));
            d00 = Avx512F.FusedMultiplyAdd(x, b0, d00); d01 = Avx512F.FusedMultiplyAdd(x, b1, d01);
            x = Vector512.Create(Unsafe.Add(ref a6, lda + p));
            d10 = Avx512F.FusedMultiplyAdd(x, b0, d10); d11 = Avx512F.FusedMultiplyAdd(x, b1, d11);
            x = Vector512.Create(Unsafe.Add(ref a6, 2 * lda + p));
            d20 = Avx512F.FusedMultiplyAdd(x, b0, d20); d21 = Avx512F.FusedMultiplyAdd(x, b1, d21);
            x = Vector512.Create(Unsafe.Add(ref a6, 3 * lda + p));
            d30 = Avx512F.FusedMultiplyAdd(x, b0, d30); d31 = Avx512F.FusedMultiplyAdd(x, b1, d31);
            x = Vector512.Create(Unsafe.Add(ref a6, 4 * lda + p));
            d40 = Avx512F.FusedMultiplyAdd(x, b0, d40); d41 = Avx512F.FusedMultiplyAdd(x, b1, d41);
            x = Vector512.Create(Unsafe.Add(ref a6, 5 * lda + p));
            d50 = Avx512F.FusedMultiplyAdd(x, b0, d50); d51 = Avx512F.FusedMultiplyAdd(x, b1, d51);
        }
        if (scale != 1f)
        {
            var s = Vector512.Create(scale);
            c00 *= s; c01 *= s; c10 *= s; c11 *= s; c20 *= s; c21 *= s; c30 *= s; c31 *= s; c40 *= s; c41 *= s; c50 *= s; c51 *= s;
            d00 *= s; d01 *= s; d10 *= s; d11 *= s; d20 *= s; d21 *= s; d30 *= s; d31 *= s; d40 *= s; d41 *= s; d50 *= s; d51 *= s;
        }
        c00.StoreUnsafe(ref c); c01.StoreUnsafe(ref c, 16);
        c10.StoreUnsafe(ref c, l1); c11.StoreUnsafe(ref c, l1 + 16);
        c20.StoreUnsafe(ref c, l2); c21.StoreUnsafe(ref c, l2 + 16);
        c30.StoreUnsafe(ref c, l3); c31.StoreUnsafe(ref c, l3 + 16);
        c40.StoreUnsafe(ref c, l4); c41.StoreUnsafe(ref c, l4 + 16);
        c50.StoreUnsafe(ref c, l5); c51.StoreUnsafe(ref c, l5 + 16);
        d00.StoreUnsafe(ref d); d01.StoreUnsafe(ref d, 16);
        d10.StoreUnsafe(ref d, l1); d11.StoreUnsafe(ref d, l1 + 16);
        d20.StoreUnsafe(ref d, l2); d21.StoreUnsafe(ref d, l2 + 16);
        d30.StoreUnsafe(ref d, l3); d31.StoreUnsafe(ref d, l3 + 16);
        d40.StoreUnsafe(ref d, l4); d41.StoreUnsafe(ref d, l4 + 16);
        d50.StoreUnsafe(ref d, l5); d51.StoreUnsafe(ref d, l5 + 16);
    }

    /// <summary>C[6×16] (=|+=) A[6×kc]·Bpack[kc×16], then scaled by <paramref name="scale"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static void Kernel6x16(ref float a0, ref float a1, ref float a2, ref float a3, ref float a4, ref float a5, ref float bp, int kc, ref float c, int ldc, bool overwrite, float scale)
    {
        Vector256<float> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        nuint l1 = (nuint)ldc, l2 = 2 * l1, l3 = 3 * l1, l4 = 4 * l1, l5 = 5 * l1;
        if (overwrite)
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector256<float>.Zero;
        }
        else
        {
            c00 = Vector256.LoadUnsafe(ref c); c01 = Vector256.LoadUnsafe(ref c, 8);
            c10 = Vector256.LoadUnsafe(ref c, l1); c11 = Vector256.LoadUnsafe(ref c, l1 + 8);
            c20 = Vector256.LoadUnsafe(ref c, l2); c21 = Vector256.LoadUnsafe(ref c, l2 + 8);
            c30 = Vector256.LoadUnsafe(ref c, l3); c31 = Vector256.LoadUnsafe(ref c, l3 + 8);
            c40 = Vector256.LoadUnsafe(ref c, l4); c41 = Vector256.LoadUnsafe(ref c, l4 + 8);
            c50 = Vector256.LoadUnsafe(ref c, l5); c51 = Vector256.LoadUnsafe(ref c, l5 + 8);
        }
        for (int p = 0; p < kc; p++)
        {
            var b0 = Vector256.LoadUnsafe(ref bp);
            var b1 = Vector256.LoadUnsafe(ref bp, 8);
            bp = ref Unsafe.Add(ref bp, 16);
            var x = Vector256.Create(Unsafe.Add(ref a0, p));
            c00 = Fma.MultiplyAdd(x, b0, c00);
            c01 = Fma.MultiplyAdd(x, b1, c01);
            x = Vector256.Create(Unsafe.Add(ref a1, p));
            c10 = Fma.MultiplyAdd(x, b0, c10);
            c11 = Fma.MultiplyAdd(x, b1, c11);
            x = Vector256.Create(Unsafe.Add(ref a2, p));
            c20 = Fma.MultiplyAdd(x, b0, c20);
            c21 = Fma.MultiplyAdd(x, b1, c21);
            x = Vector256.Create(Unsafe.Add(ref a3, p));
            c30 = Fma.MultiplyAdd(x, b0, c30);
            c31 = Fma.MultiplyAdd(x, b1, c31);
            x = Vector256.Create(Unsafe.Add(ref a4, p));
            c40 = Fma.MultiplyAdd(x, b0, c40);
            c41 = Fma.MultiplyAdd(x, b1, c41);
            x = Vector256.Create(Unsafe.Add(ref a5, p));
            c50 = Fma.MultiplyAdd(x, b0, c50);
            c51 = Fma.MultiplyAdd(x, b1, c51);
        }
        if (scale != 1f)
        {
            var s = Vector256.Create(scale);
            c00 *= s; c01 *= s; c10 *= s; c11 *= s; c20 *= s; c21 *= s;
            c30 *= s; c31 *= s; c40 *= s; c41 *= s; c50 *= s; c51 *= s;
        }
        c00.StoreUnsafe(ref c); c01.StoreUnsafe(ref c, 8);
        c10.StoreUnsafe(ref c, l1); c11.StoreUnsafe(ref c, l1 + 8);
        c20.StoreUnsafe(ref c, l2); c21.StoreUnsafe(ref c, l2 + 8);
        c30.StoreUnsafe(ref c, l3); c31.StoreUnsafe(ref c, l3 + 8);
        c40.StoreUnsafe(ref c, l4); c41.StoreUnsafe(ref c, l4 + 8);
        c50.StoreUnsafe(ref c, l5); c51.StoreUnsafe(ref c, l5 + 8);
    }

    /// <summary>Writes the transpose of an <c>[rows, cols]</c> block (row stride <paramref name="lds"/>) into
    /// <c>dst[cols, ldd]</c>, zero-filling columns <c>rows..ldd</c>.</summary>
    public static void Transpose(ReadOnlySpan<float> src, int lds, int rows, int cols, Span<float> dst, int ldd)
    {
        const int T = 16;
        for (int r0 = 0; r0 < rows; r0 += T)
        {
            int r1 = Math.Min(rows, r0 + T);
            for (int c0 = 0; c0 < cols; c0 += T)
            {
                int c1 = Math.Min(cols, c0 + T);
                for (int r = r0; r < r1; r++)
                {
                    for (int cc = c0; cc < c1; cc++)
                    {
                        dst[cc * ldd + r] = src[r * lds + cc];
                    }
                }
            }
        }
        if (ldd > rows)
        {
            for (int cc = 0; cc < cols; cc++)
            {
                dst.Slice(cc * ldd + rows, ldd - rows).Clear();
            }
        }
    }
}
