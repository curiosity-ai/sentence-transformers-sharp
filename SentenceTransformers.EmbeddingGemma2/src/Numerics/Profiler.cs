using System.Collections.Concurrent;
using System.Diagnostics;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>Opt-in, near-zero-overhead wall-clock profiler for the forward passes (diagnostics and tests).</summary>
internal static class Profiler
{
    public static bool Enabled;
    private static readonly ConcurrentDictionary<string, long> _ticks = new();
    private static readonly ConcurrentDictionary<string, long> _calls = new();

    public readonly struct Scope : IDisposable
    {
        private readonly string _name;
        private readonly long _start;

        public Scope(string name)
        {
            _name = name;
            _start = Enabled ? Stopwatch.GetTimestamp() : 0;
        }

        public void Dispose()
        {
            if (_name is not null && Enabled)
            {
                long elapsed = Stopwatch.GetTimestamp() - _start;
                _ticks.AddOrUpdate(_name, elapsed, (_, v) => v + elapsed);
                _calls.AddOrUpdate(_name, 1, (_, v) => v + 1);
            }
        }
    }

    public static Scope Measure(string name) => new(Enabled ? name : null);

    public static void Reset()
    {
        _ticks.Clear();
        _calls.Clear();
    }

    public static string Report()
    {
        var total = _ticks.Values.Sum();
        return string.Join(Environment.NewLine, _ticks.OrderByDescending(k => k.Value)
            .Select(k => $"{k.Key,-24} {k.Value * 1000.0 / Stopwatch.Frequency,10:F2} ms {_calls.GetValueOrDefault(k.Key),8} calls"));
    }
}
