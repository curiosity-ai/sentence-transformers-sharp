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
/// <c>P·V</c> while the block is still cache-resident. <c>Kᵀ</c> is materialized once per KV head.
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

        // Kᵀ per (sequence, KV head): [hd, nPad] with zero padding so score rows stay 16-float aligned.
        var kt = new float[sequences * kvHeads][];
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
                    SGemm.Multiply(q.AsSpan((start + r0) * qStride + h * hd), qStride, kt[s * kvHeads + g], ld, scores, ld, rows, ld, hd, scale);
                    Ops.SoftmaxRows(scores, rows, n, ld, maskedKeys);
                    SGemm.Multiply(scores, ld, v.AsSpan(start * kvStride + g * hd), kvStride, output.AsSpan((start + r0) * qStride + h * hd), qStride, rows, hd, n);
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
                Parallel.For(0, kt.Length, po, Transpose);
                Parallel.For(0, items.Count, po, Block);
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
        }
    }
}
