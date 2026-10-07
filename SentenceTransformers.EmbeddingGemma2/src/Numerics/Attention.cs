using System.Buffers;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>
/// Bidirectional multi-head attention over stacked sequences (no padding: each sequence attends only
/// to its own tokens, which is exactly what the reference padding mask produces for the real rows).
/// Grouped-query attention: query head <c>h</c> reads KV head <c>h · kvHeads / heads</c>, matching the
/// graph's reshape of <c>[heads, L, d]</c> into <c>[kvHeads, heads/kvHeads · L, d]</c>.
/// <para>
/// Work is split into (sequence, head, block of 64 query rows) items so even a single long document
/// spreads over every core: each item computes its score block <c>Q_blk·Kᵀ</c>, the row softmax and
/// <c>P·V</c> while the block is still cache-resident. <c>Kᵀ</c> and <c>V</c> are packed once per KV head.
/// </para>
/// </summary>
internal static class Attention
{
    private const int RowBlock = 64;

    /// <param name="q">[total, heads·hd]</param>
    /// <param name="k">[total, kvHeads·hd]</param>
    /// <param name="v">[total, kvHeads·hd]</param>
    /// <param name="output">[total, heads·hd]</param>
    /// <param name="scale">Logit scale (EmbeddingGemma 2's text encoder uses 1: the scale is folded into the q-norm weights).</param>
    /// <param name="maskedKeys">True when the reference runs each sequence padded to its signature length with
    /// masked keys (text encoder); false when every key row is real (vision encoder). Selects the softmax
    /// summation order XNNPACK uses for that row length.</param>
    public static void Run(float[] q, float[] k, float[] v, float[] output, int[] offsets, int heads, int kvHeads, int hd, float scale, ParallelOptions po, bool maskedKeys = false)
    {
        using var _ = Profiler.Measure("attention");
        int sequences = offsets.Length - 1;
        int dop = po?.MaxDegreeOfParallelism ?? 1;
        if (dop == -1)
        {
            dop = Environment.ProcessorCount;
        }
        int kvStride = kvHeads * hd;
        var pool = ArrayPool<float>.Shared;

        // Per (sequence, KV head): Kᵀ and V packed once into the GEMM kernels' panel layout (shared by every row
        // block of every query head that reads the KV head), or, without the blocked kernels, Kᵀ as [hd, nPad].
        bool packed = SGemm.SupportsPacking;
        var kt = new float[sequences * kvHeads][];
        var vp = packed ? new float[sequences * kvHeads][] : null;
        var ktOff = new int[sequences * kvHeads];
        var vpOff = new int[sequences * kvHeads];
        var nPad = new int[sequences];
        var items = new List<(int Seq, int Head, int Row0)>();
        for (int s = 0; s < sequences; s++)
        {
            int n = offsets[s + 1] - offsets[s];
            nPad[s] = (n + 15) & ~15;
            for (int h = 0; h < heads; h++)
            {
                for (int r = 0; r < n; r += RowBlock)
                {
                    items.Add((s, h, r));
                }
            }
        }
        try
        {
            void Transpose(int gi)
            {
                int s = gi / kvHeads, g = gi % kvHeads;
                int n = offsets[s + 1] - offsets[s];
                if (packed)
                {
                    var kb = pool.Rent(SGemm.PackedLength(hd, n));
                    ktOff[gi] = SGemm.PackB(k.AsSpan(offsets[s] * kvStride + g * hd), kvStride, hd, n, transposed: true, kb);
                    kt[gi] = kb;
                    var vb = pool.Rent(SGemm.PackedLength(n, hd));
                    vpOff[gi] = SGemm.PackB(v.AsSpan(offsets[s] * kvStride + g * hd), kvStride, n, hd, transposed: false, vb);
                    vp[gi] = vb;
                    return;
                }
                var buf = pool.Rent(hd * nPad[s]);
                SGemm.Transpose(k.AsSpan(offsets[s] * kvStride + g * hd), kvStride, n, hd, buf, nPad[s]);
                kt[gi] = buf;
            }
            void Block(int idx)
            {
                var (s, h, r0) = items[idx];
                int start = offsets[s];
                int n = offsets[s + 1] - start;
                int rows = Math.Min(RowBlock, n - r0);
                int g = h * kvHeads / heads;
                int qStride = heads * hd;
                int ld = nPad[s];
                var scores = pool.Rent(rows * ld);
                try
                {
                    int gi = s * kvHeads + g;
                    var qBlock = q.AsSpan((start + r0) * qStride + h * hd);
                    var oBlock = output.AsSpan((start + r0) * qStride + h * hd);
                    if (packed)
                    {
                        SGemm.MultiplyPacked(qBlock, qStride, kt[gi], ktOff[gi], scores, ld, rows, n, hd, scale);
                        Ops.SoftmaxRows(scores, rows, n, ld, maskedKeys);
                        SGemm.MultiplyPacked(scores, ld, vp[gi], vpOff[gi], oBlock, qStride, rows, hd, n);
                    }
                    else
                    {
                        SGemm.Multiply(qBlock, qStride, kt[gi], ld, scores, ld, rows, ld, hd, scale);
                        Ops.SoftmaxRows(scores, rows, n, ld, maskedKeys);
                        SGemm.Multiply(scores, ld, v.AsSpan(start * kvStride + g * hd), kvStride, oBlock, qStride, rows, hd, n);
                    }
                }
                finally
                {
                    pool.Return(scores);
                }
            }

            if (dop <= 1)
            {
                for (int gi = 0; gi < kt.Length; gi++)
                {
                    Transpose(gi);
                }
                for (int i = 0; i < items.Count; i++)
                {
                    Block(i);
                }
            }
            else
            {
                WorkerPool.For(kt.Length, po, Transpose);
                WorkerPool.For(items.Count, po, Block);
            }
        }
        finally
        {
            foreach (var buf in kt)
            {
                if (buf is not null)
                {
                    pool.Return(buf);
                }
            }
            foreach (var buf in vp ?? Array.Empty<float[]>())
            {
                if (buf is not null)
                {
                    pool.Return(buf);
                }
            }
        }
    }
}
