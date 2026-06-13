using System.Diagnostics;

namespace SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// Opt-in, very low-overhead per-stage timer for the pure forward pass, used to attribute encode time
/// to the parts that scale linearly with the sequence length (the projection matmuls) versus the part
/// that scales quadratically (self-attention). Disabled by default so it adds nothing to the hot path;
/// set <see cref="Enabled"/> to start accumulating, then call <see cref="Report"/> to print and reset.
/// </summary>
public static class ForwardProfile
{
    public static bool Enabled;

    // Stage -> accumulated stopwatch ticks. Insertion-ordered so the report reads top-to-bottom.
    private static readonly System.Collections.Generic.Dictionary<string, long> _ticks = new();
    private static readonly object _lock = new();

    public static long Start() => Enabled ? Stopwatch.GetTimestamp() : 0L;

    public static void Stop(string stage, long startTimestamp)
    {
        if (!Enabled) return;
        long delta = Stopwatch.GetTimestamp() - startTimestamp;
        lock (_lock)
        {
            _ticks.TryGetValue(stage, out var cur);
            _ticks[stage] = cur + delta;
        }
    }

    public static void Reset()
    {
        lock (_lock) { _ticks.Clear(); }
    }

    public static void Report(string header)
    {
        lock (_lock)
        {
            double toMs(long t) => t * 1000.0 / Stopwatch.Frequency;
            long total = 0;
            foreach (var v in _ticks.Values) total += v;

            Console.WriteLine($"  -- {header} --");
            foreach (var kv in _ticks)
            {
                double ms = toMs(kv.Value);
                double pct = total > 0 ? 100.0 * kv.Value / total : 0;
                Console.WriteLine($"     {kv.Key,-14} {ms,10:n1} ms  {pct,5:n1}%");
            }
            Console.WriteLine($"     {"TOTAL",-14} {toMs(total),10:n1} ms");
        }
    }
}
