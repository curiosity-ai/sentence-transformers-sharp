using System.Diagnostics;
using System.Globalization;
using BERTTokenizers.Base;
using SentenceTransformers.ArcticXs;
using SentenceTransformers.Harrier.Medium;
using SentenceTransformers.Harrier.Small;
using SentenceTransformers.Harrier.Small.Pure.Tokenizer;
using SentenceTransformers.MiniLM;
using SentenceTransformers.Qwen3;
using SentenceTransformers.Tests.Support;
using Xunit.Abstractions;

namespace SentenceTransformers.Tests;

/// <summary>
/// Speed tests for every <c>TokenizeRawAligned</c> implementation:
/// <list type="bullet">
///   <item>the shared WordPiece path in <see cref="TokenizerBase"/> (exercised via MiniLM and Arctic),</item>
///   <item>the HuggingFace byte-level BPE overrides (Qwen3, Harrier Medium, Harrier Small), and</item>
///   <item>the pure-C# Gemma BPE override (<see cref="HarrierSmallPureTokenizer"/>).</item>
/// </list>
/// Each test tokenizes the full ~600 KB <c>test-data/text.txt</c> document once (after a warmup),
/// reports throughput, and drives the <see cref="System.IProgress{T}"/> overload of
/// <c>TokenizeRawAligned</c> — which reports progress from inside the tokenizer itself. A progress
/// line is streamed roughly every 120 ms whenever a run takes longer than 300 ms; the collected
/// reports are also asserted for monotonicity and a final 100%.
/// </summary>
public class TokenizeRawAlignedSpeedTests
{
    /// <summary>Only stream progress lines once a run has been going for this long.</summary>
    private const int ProgressThresholdMs = 300;

    /// <summary>Minimum spacing between streamed progress lines.</summary>
    private static readonly TimeSpan ProgressUpdateInterval = TimeSpan.FromMilliseconds(120);

    // Large enough that nothing gets truncated for the BPE tokenizers that take a max-token limit.
    private const int MaxTokens = 1_000_000;

    // Load the sample once for the whole class — it is read-only and shared across tests.
    private static readonly string SampleText = File.ReadAllText(TestPaths.SpeedSampleText);

    // A ~100 KB LF-only slice (cut at a newline) used to build inputs for the scaling guard. It
    // contains no "\r\n" or "   ", which is exactly the shape that used to make SplitAligned O(n²).
    private static readonly string BaseUnit = MakeBaseUnit(100_000);

    private readonly ITestOutputHelper _output;

    public TokenizeRawAlignedSpeedTests(ITestOutputHelper output) => _output = output;

    [Fact]
    public void MiniLMTokenizer_TokenizeRawAligned_Speed()
    {
        var tok = new MiniLMTokenizer();
        Run("MiniLM (WordPiece base)", p => tok.TokenizeRawAligned(SampleText, p));
    }

    [Fact]
    public void ArcticTokenizer_TokenizeRawAligned_Speed()
    {
        var tok = new ArcticTokenizer();
        Run("Arctic (WordPiece base)", p => tok.TokenizeRawAligned(SampleText, p));
    }

    // Regression guards for the SplitAligned O(n²) fix. The WordPiece path is the one that regressed,
    // so these cover MiniLM and Arctic specifically. They assert scaling (a ratio) rather than an
    // absolute time so they aren't sensitive to how fast the machine is.

    [Fact]
    public void MiniLMTokenizer_TokenizeRawAligned_ScalesLinearly()
        => AssertScalesLinearly("MiniLM", new MiniLMTokenizer());

    [Fact]
    public void ArcticTokenizer_TokenizeRawAligned_ScalesLinearly()
        => AssertScalesLinearly("Arctic", new ArcticTokenizer());

    [Fact]
    public void QwenTokenizer_TokenizeRawAligned_Speed()
    {
        using var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, MaxTokens);
        Run("Qwen3 (HF byte-level BPE)", p => tok.TokenizeRawAligned(SampleText, p));
    }

    [Fact]
    public void HarrierMediumTokenizer_TokenizeRawAligned_Speed()
    {
        using var tok = new HarrierMediumTokenizer(TestPaths.HarrierMediumTokenizerJson, MaxTokens);
        Run("Harrier Medium (HF byte-level BPE)", p => tok.TokenizeRawAligned(SampleText, p));
    }

    [Fact]
    public void HarrierSmallTokenizer_TokenizeRawAligned_Speed()
    {
        using var tok = new HarrierSmallTokenizer(TestPaths.HarrierSmallTokenizerJson, MaxTokens);
        Run("Harrier Small (HF byte-level BPE)", p => tok.TokenizeRawAligned(SampleText, p));
    }

    [Fact]
    public void HarrierSmallPureTokenizer_TokenizeRawAligned_Speed()
    {
        var tok = HarrierSmallPureTokenizer.FromFile(TestPaths.HarrierSmallTokenizerJson, MaxTokens);
        Run("Harrier Small Pure (managed Gemma BPE)", p => tok.TokenizeRawAligned(SampleText, p));
    }

    /// <summary>
    /// Warms up the tokenizer, then times a single <c>TokenizeRawAligned</c> pass over the whole
    /// sample while collecting the progress reported from inside the tokenizer. Streams a progress
    /// line (throttled) for slow runs, and asserts the output is non-empty, offsets are
    /// non-decreasing, and the reported progress is monotonic and finishes at 100%.
    /// </summary>
    private void Run(string label, Func<IProgress<TokenizeProgress>, List<TokenizedTokenAligned>> tokenize)
    {
        Assert.False(string.IsNullOrEmpty(SampleText), "Speed sample document is empty.");

        // Warm up (no progress reporting) so JIT/vocabulary lookups don't pollute the timing.
        _ = tokenize(null);

        var reports = new List<TokenizeProgress>();
        var sw = Stopwatch.StartNew();
        var lastLog = TimeSpan.Zero;

        var logger = new ActionProgress(p =>
        {
            reports.Add(p);
            var elapsed = sw.Elapsed;
            if (elapsed.TotalMilliseconds >= ProgressThresholdMs && elapsed - lastLog >= ProgressUpdateInterval)
            {
                lastLog = elapsed;
                Log(string.Create(CultureInfo.InvariantCulture,
                    $"[{label}] {p.Fraction * 100,3:n0}% — {p.TokensProduced:n0} tokens, {elapsed.TotalMilliseconds:n0} ms elapsed"));
            }
        });

        var aligned = tokenize(logger);
        sw.Stop();
        var ms = sw.Elapsed.TotalMilliseconds;

        Assert.NotEmpty(aligned);

        // Offsets must be non-decreasing across the whole document — a genuine invariant of every
        // implementation and a cheap correctness guard alongside the timing.
        for (int i = 1; i < aligned.Count; i++)
        {
            Assert.True(aligned[i].Start >= aligned[i - 1].Start,
                $"[{label}] token {i} starts at {aligned[i].Start} but previous at {aligned[i - 1].Start}");
        }

        // The IProgress overload must report progress that only moves forward and finishes at 100%.
        Assert.NotEmpty(reports);
        for (int i = 1; i < reports.Count; i++)
        {
            Assert.True(reports[i].Fraction >= reports[i - 1].Fraction,
                $"[{label}] progress went backwards: {reports[i - 1].Fraction} then {reports[i].Fraction}");
        }
        Assert.Equal(1.0, reports[^1].Fraction, 3);
        Assert.Equal(aligned.Count, reports[^1].TokensProduced);

        var seconds = ms / 1000.0;
        var tokensPerSec = seconds > 0 ? aligned.Count / seconds : double.PositiveInfinity;
        var mbPerSec = seconds > 0 ? SampleText.Length / seconds / (1024.0 * 1024.0) : double.PositiveInfinity;

        var suffix = ms >= ProgressThresholdMs ? "  [exceeded 300 ms]" : string.Empty;
        Log(string.Create(CultureInfo.InvariantCulture,
            $"[{label}] {aligned.Count:n0} tokens from {SampleText.Length:n0} chars in {ms:n1} ms " +
            $"({tokensPerSec:n0} tok/s, {mbPerSec:n2} MB/s), {reports.Count:n0} progress reports{suffix}"));
    }

    /// <summary>
    /// Guards against the <c>SplitAligned</c> O(n²) regression: tokenizing 6× the input should take
    /// roughly 6× the time (linear). The old quadratic split scaled ~34× for 6× input, so a ratio
    /// well below quadratic confirms the linear behaviour without depending on absolute machine speed.
    /// </summary>
    private void AssertScalesLinearly(string label, TokenizerBase tok)
    {
        var small = BaseUnit;
        var big   = string.Concat(Enumerable.Repeat(BaseUnit, 6));

        // Warm up so JIT/vocabulary lookups don't distort the first measurement.
        _ = tok.TokenizeRawAligned(small);

        // Best-of-3 (minimum) to shed GC/scheduling noise from either measurement.
        var smallMs = BestOfMs(3, () => tok.TokenizeRawAligned(small));
        var bigMs   = BestOfMs(3, () => tok.TokenizeRawAligned(big));

        var ratio = smallMs > 0 ? bigMs / smallMs : double.PositiveInfinity;
        Log(string.Create(CultureInfo.InvariantCulture,
            $"[{label}] scaling: {small.Length:n0} chars {smallMs:n1} ms → {big.Length:n0} chars {bigMs:n1} ms " +
            $"(6× input ⇒ {ratio:n1}× time)"));

        Assert.True(ratio < 20,
            $"[{label}] TokenizeRawAligned scaled {ratio:n1}× for 6× input — near-quadratic; the SplitAligned O(n²) regression may be back.");
    }

    /// <summary>Runs <paramref name="action"/> <paramref name="runs"/> times and returns the fastest elapsed ms.</summary>
    private static double BestOfMs(int runs, Func<object> action)
    {
        var best = double.MaxValue;
        for (int i = 0; i < runs; i++)
        {
            var sw = Stopwatch.StartNew();
            _ = action();
            sw.Stop();
            best = Math.Min(best, sw.Elapsed.TotalMilliseconds);
        }
        return best;
    }

    /// <summary>Returns the sample truncated to ~<paramref name="approxChars"/>, snapped to a newline.</summary>
    private static string MakeBaseUnit(int approxChars)
    {
        var end = Math.Min(approxChars, SampleText.Length);
        while (end < SampleText.Length && SampleText[end] != '\n') end++;
        return SampleText.Substring(0, end);
    }

    /// <summary>
    /// Writes to both the xUnit output (captured per test) and the console so progress lines stream
    /// live during <c>dotnet test</c>.
    /// </summary>
    private void Log(string message)
    {
        _output.WriteLine(message);
        Console.WriteLine(message);
    }

    /// <summary>
    /// A synchronous <see cref="IProgress{T}"/> — unlike <see cref="Progress{T}"/> it invokes the
    /// callback inline on the reporting thread, so reports are collected in order and before the
    /// tokenize call returns.
    /// </summary>
    private sealed class ActionProgress : IProgress<TokenizeProgress>
    {
        private readonly Action<TokenizeProgress> _onReport;
        public ActionProgress(Action<TokenizeProgress> onReport) => _onReport = onReport;
        public void Report(TokenizeProgress value) => _onReport(value);
    }
}
