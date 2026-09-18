using System.Collections.Concurrent;
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

    /// <summary>Test/diagnostic switch: force the portable head-fused attention path even on x64, so the
    /// FMA fast path can be compared against it for correctness. Not used in production.</summary>
    public static bool ForcePortableAttention;

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

    /// <summary>
    /// Stages of the packed (<c>.stq</c>) matmul, timed separately so the cost can be attributed
    /// rather than guessed at. Kept as an enum with thread-local counters instead of the string
    /// dictionary above: these fire once per output tile rather than once per layer, so a lock and a
    /// hash lookup per sample would dominate what they are trying to measure.
    /// </summary>
    public enum Stage
    {
        /// <summary>Walsh-Hadamard rotation of the activations before a packed matmul.</summary>
        Rotate = 0,
        /// <summary>Dynamic int8 quantization of the activations.</summary>
        QuantizeActivations,
        /// <summary>Expanding a tile's packed codes back to signed bytes.</summary>
        UnpackWeights,
        /// <summary>The VNNI accumulate, per-group reduction and rescale.</summary>
        Dot,
        Count,
    }

    [ThreadStatic] private static long[] _stageTicks;
    private static readonly ConcurrentBag<long[]> _allStageTicks = new();

    public static long StageStart() => Enabled ? Stopwatch.GetTimestamp() : 0L;

    public static void StageStop(Stage stage, long startTimestamp)
    {
        if (!Enabled) return;
        var local = _stageTicks;
        if (local is null)
        {
            local = _stageTicks = new long[(int)Stage.Count];
            _allStageTicks.Add(local);
        }
        local[(int)stage] += Stopwatch.GetTimestamp() - startTimestamp;
    }

    public static void ResetStages()
    {
        foreach (var arr in _allStageTicks)
        {
            Array.Clear(arr);
        }
    }

    /// <summary>Prints the packed-matmul stage breakdown, summed across every worker thread. The
    /// total exceeds wall-clock time when encoding runs in parallel - it is CPU time, and the point
    /// is the ratio between stages.</summary>
    public static void ReportStages(string header)
    {
        var totals = new long[(int)Stage.Count];
        foreach (var arr in _allStageTicks)
        {
            for (int i = 0; i < totals.Length; i++) totals[i] += arr[i];
        }

        long sum = 0;
        foreach (var t in totals) sum += t;
        if (sum == 0)
        {
            Console.WriteLine($"  -- {header}: no samples (was ForwardProfile.Enabled set?) --");
            return;
        }

        Console.WriteLine($"  -- {header} (CPU time across all workers) --");
        for (int i = 0; i < (int)Stage.Count; i++)
        {
            double ms = totals[i] * 1000.0 / Stopwatch.Frequency;
            Console.WriteLine($"     {(Stage)i,-20} {ms,10:n1} ms  {100.0 * totals[i] / sum,5:n1}%");
        }
        Console.WriteLine($"     {"TOTAL",-20} {sum * 1000.0 / Stopwatch.Frequency,10:n1} ms");
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
