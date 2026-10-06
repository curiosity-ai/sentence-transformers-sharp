using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>
/// Single-precision GEMM <c>C[M, N] = alpha · A[M, K] · B[K, N]</c> (row-major, arbitrary leading
/// dimensions) used for attention (<c>Q·Kᵀ</c>, <c>P·V</c>) and the vision/audio float layers.
/// <para>
/// Goto-style blocking: B is packed per (KC × NR) panel into a contiguous, L1-resident buffer; A rows are
/// broadcast against it by a 6 × NR register tile of FMA accumulators (NR = 32 with AVX-512, 16 with
/// AVX2/FMA), i.e. 12 vector accumulators, 2 B loads and 1 broadcast per k step. Ragged tiles are
/// computed into a scratch tile and copied out. Platforms without 256-bit SIMD fall back to
/// <see cref="TensorPrimitives"/> row updates.
/// </para>
/// </summary>
internal static class SGemm
{
    private const int MR = 6;
    private static readonly int KC = Avx512F.IsSupported ? 128 : 256;   // packed panel KC × NR floats = 16 KB (half of L1d)
    private const int RowChunk = 48;   // rows per parallel work item (8 register tiles)

    private static readonly bool UseAvx512 = Avx512F.IsSupported;
    private static readonly bool UseFma = Fma.IsSupported && Avx.IsSupported;
    private static readonly int NR = UseAvx512 ? 32 : 16;

    public static unsafe void Multiply(ReadOnlySpan<float> a, int lda, ReadOnlySpan<float> b, int ldb, Span<float> c, int ldc,
                                       int m, int n, int k, float alpha = 1f, ParallelOptions po = null)
    {
        if (m == 0 || n == 0)
        {
            return;
        }
        int dop = po?.MaxDegreeOfParallelism ?? 1;
        if (dop == -1)
        {
            dop = Environment.ProcessorCount;
        }
        fixed (float* ap = a)
        fixed (float* bp = b)
        fixed (float* cp = c)
        {
            nint aa = (nint)ap, bb = (nint)bp, cc = (nint)cp;
            if (!Vector256.IsHardwareAccelerated || !UseFma)
            {
                Portable((float*)aa, lda, (float*)bb, ldb, (float*)cc, ldc, 0, m, n, k, alpha);
                return;
            }
            int chunks = (m + RowChunk - 1) / RowChunk;
            if (dop <= 1 || chunks == 1)
            {
                Chunk((float*)aa, lda, (float*)bb, ldb, (float*)cc, ldc, 0, m, n, k, alpha);
            }
            else
            {
                Parallel.For(0, chunks, po, ch =>
                    Chunk((float*)aa, lda, (float*)bb, ldb, (float*)cc, ldc, ch * RowChunk, Math.Min(m, (ch + 1) * RowChunk), n, k, alpha));
            }
        }
    }

    private static unsafe void Portable(float* a, int lda, float* b, int ldb, float* c, int ldc, int m0, int m1, int n, int k, float alpha)
    {
        for (int i = m0; i < m1; i++)
        {
            var crow = new Span<float>(c + (long)i * ldc, n);
            crow.Clear();
            for (int p = 0; p < k; p++)
            {
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(b + (long)p * ldb, n), a[(long)i * lda + p] * alpha, crow, crow);
            }
        }
    }

    private static unsafe void Chunk(float* a, int lda, float* b, int ldb, float* c, int ldc, int m0, int m1, int n, int k, float alpha)
    {
        int nr = NR;
        var packArr = ArrayPool<float>.Shared.Rent(KC * nr + 16);
        float* tile = stackalloc float[MR * 32];
        try
        {
            fixed (float* packBase = packArr)
            {
                // 64-byte align the packed panel.
                float* pack = (float*)(((nint)packBase + 63) & ~(nint)63);
                for (int j0 = 0; j0 < n; j0 += nr)
                {
                    int nc = Math.Min(nr, n - j0);
                    for (int k0 = 0; k0 < k; k0 += KC)
                    {
                        int kc = Math.Min(KC, k - k0);
                        bool first = k0 == 0;
                        bool last = k0 + kc >= k;
                        PackB(b, ldb, k0, kc, j0, nc, nr, pack);
                        for (int i = m0; i < m1; i += MR)
                        {
                            int rows = Math.Min(MR, m1 - i);
                            float* a0 = a + (long)i * lda + k0;
                            bool full = rows == MR && nc == nr;
                            float* cDst = full ? c + (long)i * ldc + j0 : tile;
                            int cStride = full ? ldc : nr;
                            if (!full)
                            {
                                // Ragged tile: stage the existing partial sums in the scratch tile.
                                for (int r = 0; r < MR; r++)
                                {
                                    for (int cc = 0; cc < nr; cc++)
                                    {
                                        tile[r * nr + cc] = !first && r < rows && cc < nc ? c[(long)(i + r) * ldc + j0 + cc] : 0f;
                                    }
                                }
                            }
                            // Rows past the end of a ragged tile re-read the last valid A row (results discarded).
                            float* a1 = a0 + (long)Math.Min(1, rows - 1) * lda;
                            float* a2 = a0 + (long)Math.Min(2, rows - 1) * lda;
                            float* a3 = a0 + (long)Math.Min(3, rows - 1) * lda;
                            float* a4 = a0 + (long)Math.Min(4, rows - 1) * lda;
                            float* a5 = a0 + (long)Math.Min(5, rows - 1) * lda;
                            if (UseAvx512)
                            {
                                Kernel6x32(a0, a1, a2, a3, a4, a5, pack, kc, cDst, cStride, first && full, last ? alpha : 1f);
                            }
                            else
                            {
                                Kernel6x16(a0, a1, a2, a3, a4, a5, pack, kc, cDst, cStride, first && full, last ? alpha : 1f);
                            }
                            if (!full)
                            {
                                for (int r = 0; r < rows; r++)
                                {
                                    new ReadOnlySpan<float>(tile + r * nr, nc).CopyTo(new Span<float>(c + (long)(i + r) * ldc + j0, nc));
                                }
                            }
                        }
                    }
                }
            }
        }
        finally
        {
            ArrayPool<float>.Shared.Return(packArr);
        }
    }

    /// <summary>Copies B[k0..k0+kc, j0..j0+nc] into a contiguous kc × nr panel (zero padded).</summary>
    private static unsafe void PackB(float* b, int ldb, int k0, int kc, int j0, int nc, int nr, float* dst)
    {
        for (int p = 0; p < kc; p++)
        {
            float* src = b + (long)(k0 + p) * ldb + j0;
            float* d = dst + p * nr;
            if (nc == nr)
            {
                Buffer.MemoryCopy(src, d, nr * sizeof(float), nr * sizeof(float));
            }
            else
            {
                int cc = 0;
                for (; cc < nc; cc++)
                {
                    d[cc] = src[cc];
                }
                for (; cc < nr; cc++)
                {
                    d[cc] = 0f;
                }
            }
        }
    }

    /// <summary>C[6×32] (=|+=) A[6×kc]·Bpack[kc×32], then scaled by <paramref name="scale"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel6x32(float* a0, float* a1, float* a2, float* a3, float* a4, float* a5, float* bp, int kc, float* c, int ldc, bool overwrite, float scale)
    {
        Vector512<float> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        if (overwrite)
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector512<float>.Zero;
        }
        else
        {
            c00 = Avx512F.LoadVector512(c); c01 = Avx512F.LoadVector512(c + 16);
            c10 = Avx512F.LoadVector512(c + ldc); c11 = Avx512F.LoadVector512(c + ldc + 16);
            c20 = Avx512F.LoadVector512(c + 2 * ldc); c21 = Avx512F.LoadVector512(c + 2 * ldc + 16);
            c30 = Avx512F.LoadVector512(c + 3 * ldc); c31 = Avx512F.LoadVector512(c + 3 * ldc + 16);
            c40 = Avx512F.LoadVector512(c + 4 * ldc); c41 = Avx512F.LoadVector512(c + 4 * ldc + 16);
            c50 = Avx512F.LoadVector512(c + 5 * ldc); c51 = Avx512F.LoadVector512(c + 5 * ldc + 16);
        }
        for (int p = 0; p < kc; p++)
        {
            var b0 = Avx512F.LoadAlignedVector512(bp);
            var b1 = Avx512F.LoadAlignedVector512(bp + 16);
            bp += 32;
            var x = Vector512.Create(a0[p]);
            c00 = Avx512F.FusedMultiplyAdd(x, b0, c00);
            c01 = Avx512F.FusedMultiplyAdd(x, b1, c01);
            x = Vector512.Create(a1[p]);
            c10 = Avx512F.FusedMultiplyAdd(x, b0, c10);
            c11 = Avx512F.FusedMultiplyAdd(x, b1, c11);
            x = Vector512.Create(a2[p]);
            c20 = Avx512F.FusedMultiplyAdd(x, b0, c20);
            c21 = Avx512F.FusedMultiplyAdd(x, b1, c21);
            x = Vector512.Create(a3[p]);
            c30 = Avx512F.FusedMultiplyAdd(x, b0, c30);
            c31 = Avx512F.FusedMultiplyAdd(x, b1, c31);
            x = Vector512.Create(a4[p]);
            c40 = Avx512F.FusedMultiplyAdd(x, b0, c40);
            c41 = Avx512F.FusedMultiplyAdd(x, b1, c41);
            x = Vector512.Create(a5[p]);
            c50 = Avx512F.FusedMultiplyAdd(x, b0, c50);
            c51 = Avx512F.FusedMultiplyAdd(x, b1, c51);
        }
        if (scale != 1f)
        {
            var s = Vector512.Create(scale);
            c00 = Avx512F.Multiply(c00, s); c01 = Avx512F.Multiply(c01, s);
            c10 = Avx512F.Multiply(c10, s); c11 = Avx512F.Multiply(c11, s);
            c20 = Avx512F.Multiply(c20, s); c21 = Avx512F.Multiply(c21, s);
            c30 = Avx512F.Multiply(c30, s); c31 = Avx512F.Multiply(c31, s);
            c40 = Avx512F.Multiply(c40, s); c41 = Avx512F.Multiply(c41, s);
            c50 = Avx512F.Multiply(c50, s); c51 = Avx512F.Multiply(c51, s);
        }
        Avx512F.Store(c, c00); Avx512F.Store(c + 16, c01);
        Avx512F.Store(c + ldc, c10); Avx512F.Store(c + ldc + 16, c11);
        Avx512F.Store(c + 2 * ldc, c20); Avx512F.Store(c + 2 * ldc + 16, c21);
        Avx512F.Store(c + 3 * ldc, c30); Avx512F.Store(c + 3 * ldc + 16, c31);
        Avx512F.Store(c + 4 * ldc, c40); Avx512F.Store(c + 4 * ldc + 16, c41);
        Avx512F.Store(c + 5 * ldc, c50); Avx512F.Store(c + 5 * ldc + 16, c51);
    }

    /// <summary>C[6×16] (=|+=) A[6×kc]·Bpack[kc×16], then scaled by <paramref name="scale"/>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    private static unsafe void Kernel6x16(float* a0, float* a1, float* a2, float* a3, float* a4, float* a5, float* bp, int kc, float* c, int ldc, bool overwrite, float scale)
    {
        Vector256<float> c00, c01, c10, c11, c20, c21, c30, c31, c40, c41, c50, c51;
        if (overwrite)
        {
            c00 = c01 = c10 = c11 = c20 = c21 = c30 = c31 = c40 = c41 = c50 = c51 = Vector256<float>.Zero;
        }
        else
        {
            c00 = Avx.LoadVector256(c); c01 = Avx.LoadVector256(c + 8);
            c10 = Avx.LoadVector256(c + ldc); c11 = Avx.LoadVector256(c + ldc + 8);
            c20 = Avx.LoadVector256(c + 2 * ldc); c21 = Avx.LoadVector256(c + 2 * ldc + 8);
            c30 = Avx.LoadVector256(c + 3 * ldc); c31 = Avx.LoadVector256(c + 3 * ldc + 8);
            c40 = Avx.LoadVector256(c + 4 * ldc); c41 = Avx.LoadVector256(c + 4 * ldc + 8);
            c50 = Avx.LoadVector256(c + 5 * ldc); c51 = Avx.LoadVector256(c + 5 * ldc + 8);
        }
        for (int p = 0; p < kc; p++)
        {
            var b0 = Avx.LoadAlignedVector256(bp);
            var b1 = Avx.LoadAlignedVector256(bp + 8);
            bp += 16;
            var x = Vector256.Create(a0[p]);
            c00 = Fma.MultiplyAdd(x, b0, c00);
            c01 = Fma.MultiplyAdd(x, b1, c01);
            x = Vector256.Create(a1[p]);
            c10 = Fma.MultiplyAdd(x, b0, c10);
            c11 = Fma.MultiplyAdd(x, b1, c11);
            x = Vector256.Create(a2[p]);
            c20 = Fma.MultiplyAdd(x, b0, c20);
            c21 = Fma.MultiplyAdd(x, b1, c21);
            x = Vector256.Create(a3[p]);
            c30 = Fma.MultiplyAdd(x, b0, c30);
            c31 = Fma.MultiplyAdd(x, b1, c31);
            x = Vector256.Create(a4[p]);
            c40 = Fma.MultiplyAdd(x, b0, c40);
            c41 = Fma.MultiplyAdd(x, b1, c41);
            x = Vector256.Create(a5[p]);
            c50 = Fma.MultiplyAdd(x, b0, c50);
            c51 = Fma.MultiplyAdd(x, b1, c51);
        }
        if (scale != 1f)
        {
            var s = Vector256.Create(scale);
            c00 *= s; c01 *= s; c10 *= s; c11 *= s; c20 *= s; c21 *= s;
            c30 *= s; c31 *= s; c40 *= s; c41 *= s; c50 *= s; c51 *= s;
        }
        Avx.Store(c, c00); Avx.Store(c + 8, c01);
        Avx.Store(c + ldc, c10); Avx.Store(c + ldc + 8, c11);
        Avx.Store(c + 2 * ldc, c20); Avx.Store(c + 2 * ldc + 8, c21);
        Avx.Store(c + 3 * ldc, c30); Avx.Store(c + 3 * ldc + 8, c31);
        Avx.Store(c + 4 * ldc, c40); Avx.Store(c + 4 * ldc + 8, c41);
        Avx.Store(c + 5 * ldc, c50); Avx.Store(c + 5 * ldc + 8, c51);
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
