namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>Splits row-wise elementwise work into chunks for <see cref="Parallel.For(int, int, ParallelOptions, Action{int})"/>;
/// small tensors (or single-threaded options) run inline to avoid scheduling overhead.</summary>
internal static class ParallelRows
{
    /// <summary>Minimum number of elements before work is split across threads.</summary>
    private const int MinElementsPerChunk = 16 * 1024;

    public static void For(int rows, int rowSize, ParallelOptions po, Action<int, int> body)
    {
        int dop = po?.MaxDegreeOfParallelism ?? 1;
        if (dop == -1)
        {
            dop = Environment.ProcessorCount;
        }
        long elements = (long)rows * rowSize;
        int chunks = (int)Math.Min(Math.Min(rows, dop * 4L), Math.Max(1, elements / MinElementsPerChunk));
        if (dop <= 1 || chunks <= 1)
        {
            body(0, rows);
            return;
        }
        int per = (rows + chunks - 1) / chunks;
        Parallel.For(0, chunks, po, c =>
        {
            int r0 = c * per, r1 = Math.Min(rows, r0 + per);
            if (r0 < r1)
            {
                body(r0, r1);
            }
        });
    }
}
