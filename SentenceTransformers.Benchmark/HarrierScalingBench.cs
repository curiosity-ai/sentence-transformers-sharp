using System.Diagnostics;
using System.Text;
using System.Text.Json;
using SentenceTransformers;

/// <summary>
/// Token-count scalability sweep for Harrier Small. Drives the SAME calibrated input strings
/// (one per target sequence length, including the &lt;bos&gt;/&lt;eos&gt; specials) through the public
/// end-to-end <c>EncodeAsync</c> path of both the pure-C# reimplementation and the ONNX build, and
/// records wall-clock per encode at 128 / 256 / 512 / 1024 / 2048 / 4096 tokens.
///
/// The calibrated inputs (and the measured token counts) are written to a JSON file so the Python
/// sentence-transformers harness can replay byte-identical text and produce a directly comparable
/// third column. Because all three share the same Gemma BPE tokenizer, the per-target token counts
/// line up across implementations.
/// </summary>
public static class HarrierScalingBench
{
    private static readonly int[] Targets = { 128, 256, 512, 1024, 2048, 4096 };

    // Deterministic, content-agnostic word pool. The model's cost depends on sequence length, not on
    // which tokens appear, so a repeated pool is fine and keeps the input reproducible.
    private static readonly string[] Pool =
    {
        "curiosity", "mosaik", "search", "embedding", "vector", "database", "ranking", "hybrid",
        "context", "token", "runtime", "inference", "latency", "throughput", "quantization",
        "semantic", "retrieval", "document", "chunk", "pipeline", "provider", "benchmark",
        "scalability", "transformer", "attention", "encoder", "multilingual", "normalize",
    };

    public static async Task RunAsync()
    {
        Console.WriteLine($"Harrier Small token-count scalability sweep (ProcessorCount={Environment.ProcessorCount})");
        Console.WriteLine($"Targets (tokens incl. <bos>/<eos>): {string.Join(", ", Targets)}");
        Console.WriteLine();

        // ---- 1. Load the pure encoder; use its tokenizer to calibrate the input strings. ----
        Console.WriteLine("Loading Harrier.Small.Pure (fp32) ...");
        using var pure = await SentenceTransformers.Harrier.Small.Pure.SentenceEncoder.CreateAsync(
            quantization: SentenceTransformers.Harrier.Small.Pure.Model.Quantization.None,
            reportProgress: Progress("pure-weights"));
        Console.WriteLine();

        int CountTokens(string s) => pure.Tokenizer.Encode(new[] { s })[0].InputIds.Length;

        var inputs = new List<(int Target, int Tokens, string Text)>();
        foreach (var target in Targets)
        {
            var (text, tokens) = Calibrate(CountTokens, target);
            inputs.Add((target, tokens, text));
            Console.WriteLine($"  calibrated target {target,5} -> {tokens,5} tokens ({text.Length} chars)");
        }
        Console.WriteLine();

        // Persist the calibrated inputs so the Python harness replays identical text.
        var inputsPath = "/tmp/harrier_scaling_inputs.json";
        await File.WriteAllTextAsync(inputsPath, JsonSerializer.Serialize(
            inputs.Select(i => new { target = i.Target, tokens = i.Tokens, text = i.Text }),
            new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"Wrote calibrated inputs -> {inputsPath}");
        Console.WriteLine();

        // ---- 2. Time the pure encoder at each token count. ----
        var pureMs = new double[inputs.Count];
        Console.WriteLine("=== Harrier.Small.Pure (fp32) ===");
        for (int i = 0; i < inputs.Count; i++)
        {
            pureMs[i] = await TimeEncodeAsync(s => pure.EncodeAsync(s), inputs[i].Text, inputs[i].Tokens);
        }
        Console.WriteLine();

        // ---- 3. Time the ONNX encoder (default Q4F16) at each token count. ----
        var onnxMs = new double[inputs.Count];
        bool onnxOk = true;
        try
        {
            Console.WriteLine("Loading Harrier.Small (ONNX Q4F16) ...");
            using var onnx = await SentenceTransformers.Harrier.Small.SentenceEncoder.CreateAsync(
                reportProgress: Progress("onnx-weights"));
            Console.WriteLine();
            Console.WriteLine("=== Harrier.Small (ONNX Q4F16) ===");
            for (int i = 0; i < inputs.Count; i++)
            {
                onnxMs[i] = await TimeEncodeAsync(s => onnx.EncodeAsync(s), inputs[i].Text, inputs[i].Tokens);
            }
        }
        catch (Exception ex)
        {
            onnxOk = false;
            Console.WriteLine($"ONNX run failed: {ex.Message}");
        }
        Console.WriteLine();

        // ---- 4. Summary table + machine-readable results for merging with Python. ----
        Console.WriteLine("=== Scalability: elapsed ms per encode vs token count ===");
        Console.WriteLine($"{"Tokens",8} | {"Pure fp32 (ms)",16} | {"ONNX Q4F16 (ms)",16}");
        Console.WriteLine(new string('-', 50));
        for (int i = 0; i < inputs.Count; i++)
        {
            var onnxCell = onnxOk ? onnxMs[i].ToString("n1") : "n/a";
            Console.WriteLine($"{inputs[i].Tokens,8} | {pureMs[i],16:n1} | {onnxCell,16}");
        }

        var resultsPath = "/tmp/harrier_scaling_csharp.json";
        await File.WriteAllTextAsync(resultsPath, JsonSerializer.Serialize(
            inputs.Select((inp, i) => new
            {
                target = inp.Target,
                tokens = inp.Tokens,
                pure_ms = pureMs[i],
                onnx_ms = onnxOk ? (double?)onnxMs[i] : null,
            }),
            new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine();
        Console.WriteLine($"Wrote C# results -> {resultsPath}");
    }

    /// <summary>
    /// Re-runs only the pure encoder at a chosen weight precision, replaying the byte-identical
    /// calibrated inputs persisted by <see cref="RunAsync"/> (so it lines up with the existing
    /// fp32 / ONNX / Python columns without recalibrating). Writes <c>/tmp/harrier_scaling_pure_{quant}.json</c>.
    /// </summary>
    public static async Task RunPureQuantAsync(SentenceTransformers.Harrier.Small.Pure.Model.Quantization quant)
    {
        const string inputsPath = "/tmp/harrier_scaling_inputs.json";
        if (!File.Exists(inputsPath))
        {
            Console.WriteLine($"Calibrated inputs not found at {inputsPath}; run 'harrier-scaling' first.");
            return;
        }

        using var doc = JsonDocument.Parse(await File.ReadAllTextAsync(inputsPath));
        var inputs = doc.RootElement.EnumerateArray()
            .Select(e => (Tokens: e.GetProperty("tokens").GetInt32(), Text: e.GetProperty("text").GetString()!))
            .ToList();

        Console.WriteLine($"Loading Harrier.Small.Pure ({quant}) ... (ProcessorCount={Environment.ProcessorCount})");
        using var pure = await SentenceTransformers.Harrier.Small.Pure.SentenceEncoder.CreateAsync(
            quantization: quant,
            reportProgress: Progress("pure-weights"));
        Console.WriteLine();

        var ms = new double[inputs.Count];
        Console.WriteLine($"=== Harrier.Small.Pure ({quant}) ===");
        for (int i = 0; i < inputs.Count; i++)
        {
            ms[i] = await TimeEncodeAsync(s => pure.EncodeAsync(s), inputs[i].Text, inputs[i].Tokens);
        }

        var resultsPath = $"/tmp/harrier_scaling_pure_{quant}.json";
        await File.WriteAllTextAsync(resultsPath, JsonSerializer.Serialize(
            inputs.Select((inp, i) => new { tokens = inp.Tokens, pure_ms = ms[i] }),
            new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine();
        Console.WriteLine($"Wrote results -> {resultsPath}");
    }

    /// <summary>Warms up once, then times the median over an iteration count that shrinks as the
    /// sequence (and therefore the per-encode cost) grows, so the large-N points stay affordable.</summary>
    private static async Task<double> TimeEncodeAsync(Func<string[], Task<float[][]>> encode, string text, int tokens)
    {
        var batch = new[] { text };

        // Warmup (also forces any lazy init / JIT on the hot path).
        await encode(batch);

        int iterations =
            tokens <= 256  ? 7 :
            tokens <= 512  ? 5 :
            tokens <= 1024 ? 3 :
            tokens <= 2048 ? 2 : 1;

        var samples = new double[iterations];
        for (int i = 0; i < iterations; i++)
        {
            var sw = Stopwatch.StartNew();
            await encode(batch);
            sw.Stop();
            samples[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(samples);
        double median = samples[samples.Length / 2];
        Console.WriteLine($"  {tokens,5} tokens: {median,9:n1} ms  (median of {iterations}, min {samples[0]:n1})");
        return median;
    }

    /// <summary>
    /// Builds a string whose tokenized length (including the &lt;bos&gt;/&lt;eos&gt; specials added by the
    /// tokenizer) equals <paramref name="target"/>. Grows by whole pool words to get close, then tops up
    /// with single-token fillers to land exactly on (or within one of) the target.
    /// </summary>
    private static (string Text, int Tokens) Calibrate(Func<string, int> countTokens, int target)
    {
        var sb = new StringBuilder();
        int idx = 0;

        while (true)
        {
            int prevLen = sb.Length;
            if (sb.Length > 0) sb.Append(' ');
            sb.Append(Pool[idx++ % Pool.Length]);
            int c = countTokens(sb.ToString());
            if (c >= target)
            {
                if (c == target) return (sb.ToString(), c);
                sb.Length = prevLen; // overshoot: drop the last whole word and top up below
                break;
            }
        }

        // Top up with " a" (a single Gemma BPE token: metaspace + 'a') until we hit the target exactly.
        while (true)
        {
            int prevLen = sb.Length;
            sb.Append(" a");
            int c = countTokens(sb.ToString());
            if (c >= target)
            {
                if (c > target) sb.Length = prevLen; // can't land exactly; report the nearest under
                break;
            }
        }
        return (sb.ToString(), countTokens(sb.ToString()));
    }

    private static Action<DownloadProgress> Progress(string label)
    {
        int lastPct = -1;
        return p =>
        {
            int pct = (int)(p.Fraction * 100);
            if (pct != lastPct && pct % 10 == 0)
            {
                lastPct = pct;
                Console.WriteLine($"  [{label}] {pct,3}%  ({p.DownloadedBytes / 1024 / 1024} MB)");
            }
        };
    }
}
