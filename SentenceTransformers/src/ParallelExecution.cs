namespace SentenceTransformers;

/// <summary>
/// Controls how the pure-managed inference kernels (Harrier.Small.Pure) run their fork/join loops - the
/// per-layer matmuls, attention and quantization. The behaviour is driven by the caller-supplied
/// <see cref="ParallelOptions"/>: when its <see cref="ParallelOptions.MaxDegreeOfParallelism"/> is
/// greater than one the iteration range is dispatched across threads with
/// <see cref="Parallel.ForAsync(int, int, ParallelOptions, Func{int, CancellationToken, ValueTask})"/>;
/// otherwise the iterations run sequentially on the calling thread.
///
/// Parallelism is therefore opt-in per call - encoding never silently fans out across every core unless
/// the host passes options that request it.
/// </summary>
public static class ParallelExecution
{
    /// <summary>
    /// Runs <paramref name="body"/> for every index in <c>[fromInclusive, toExclusive)</c>. When
    /// <paramref name="parallelOptions"/> requests more than one degree of parallelism the iterations are
    /// dispatched with
    /// <see cref="Parallel.ForAsync(int, int, ParallelOptions, Func{int, CancellationToken, ValueTask})"/>;
    /// otherwise (including when <paramref name="parallelOptions"/> is <c>null</c>) they run in order on
    /// the calling thread. The returned task completes once every iteration has finished.
    /// </summary>
    public static async Task ForAsync(int fromInclusive, int toExclusive, ParallelOptions parallelOptions, Func<int, CancellationToken, ValueTask> body)
    {
        if (parallelOptions is not null && parallelOptions.MaxDegreeOfParallelism > 1)
        {
            await Parallel.ForAsync(fromInclusive, toExclusive, parallelOptions, body);
        }
        else
        {
            // Single-threaded (or no options supplied): run in order on the calling thread. This is the
            // default; it must NOT also run the parallel branch above, or every body would execute twice.
            await RunSequentialAsync(fromInclusive, toExclusive, parallelOptions?.CancellationToken ?? default, body);
        }
    }

    private static async Task RunSequentialAsync(int fromInclusive, int toExclusive, CancellationToken ct, Func<int, CancellationToken, ValueTask> body)
    {
        for (int i = fromInclusive; i < toExclusive; i++)
        {
            ct.ThrowIfCancellationRequested();
            await body(i, ct).ConfigureAwait(false);
        }
    }
}
