namespace SentenceTransformers;

/// <summary>
/// Global switch controlling how the pure-managed inference kernels (Harrier.Small.Pure) run their
/// fork/join loops - the per-layer matmuls, attention and quantization. When <see cref="Enabled"/> is
/// <c>true</c> the iteration range is dispatched across threads with
/// <see cref="Parallel.ForAsync(int, int, CancellationToken, Func{int, CancellationToken, ValueTask})"/>;
/// when it is <c>false</c> the iterations run sequentially on the calling thread.
///
/// Parallelism is <b>disabled by default</b> (single-threaded), so encoding never silently fans out
/// across every core - the host application opts into multi-threading explicitly by setting
/// <see cref="Enabled"/> to <c>true</c> (typically once at start-up). The flag is read at the start of
/// each loop, so it can be toggled at any time.
/// </summary>
public static class ParallelExecution
{
    /// <summary>When <c>true</c>, kernel loops dispatch with <see cref="Parallel.ForAsync(int, int, CancellationToken, Func{int, CancellationToken, ValueTask})"/>;
    /// when <c>false</c> (the default) they run single-threaded on the calling thread.</summary>
    public static bool Enabled;

    /// <summary>
    /// Runs <paramref name="body"/> for every index in <c>[fromInclusive, toExclusive)</c>. When
    /// <see cref="Enabled"/> is <c>true</c> the iterations are dispatched with
    /// <see cref="Parallel.ForAsync(int, int, CancellationToken, Func{int, CancellationToken, ValueTask})"/>;
    /// otherwise they run in order on the calling thread. The returned task completes once every
    /// iteration has finished.
    /// </summary>
    public static Task ForAsync(int fromInclusive, int toExclusive, CancellationToken ct, Func<int, CancellationToken, ValueTask> body)
    {
        if (Enabled)
        {
            return Parallel.ForAsync(fromInclusive, toExclusive, ct, body);
        }
        return RunSequentialAsync(fromInclusive, toExclusive, ct, body);
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
