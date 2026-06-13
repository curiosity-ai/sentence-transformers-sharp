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

    /// <summary>Drives the pure encoder on all cores (without changing the library's single-threaded
    /// default) so its numbers are comparable to ONNX Runtime / PyTorch, which use every core.</summary>
    private static ParallelOptions MaxCores => new() { MaxDegreeOfParallelism = Environment.ProcessorCount };

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

        var texts = inputs.Select(i => i.Text).ToArray();
        var tokenCounts = inputs.Select(i => i.Tokens).ToArray();

        // ---- 2. Pure encoder (fp32 + Int8), run on all cores for a fair architectural comparison. ----
        var pureFp32 = await TimeColumnAsync("Harrier.Small.Pure (fp32)",
            s => pure.EncodeAsync(s, MaxCores), texts, tokenCounts);

        Console.WriteLine("Loading Harrier.Small.Pure (Int8) ...");
        double[] pureInt8;
        using (var pureQ = await SentenceTransformers.Harrier.Small.Pure.SentenceEncoder.CreateAsync(
                   quantization: SentenceTransformers.Harrier.Small.Pure.Model.Quantization.Int8))
        {
            pureInt8 = await TimeColumnAsync("Harrier.Small.Pure (Int8)",
                s => pureQ.EncodeAsync(s, MaxCores), texts, tokenCounts);
        }

        // ---- 3. ONNX encoder (Q4F16 default + Int8 'quantized' variant). ----
        var onnxQ4 = await TimeOnnxAsync("Harrier.Small (ONNX Q4F16)", texts, tokenCounts,
            modelUrl: null, dataUrl: null);
        var onnxInt8 = await TimeOnnxAsync("Harrier.Small (ONNX Int8)", texts, tokenCounts,
            modelUrl: SentenceTransformers.Harrier.Small.SentenceEncoder.Quantizations.QuantizedModelUrl,
            dataUrl: SentenceTransformers.Harrier.Small.SentenceEncoder.Quantizations.QuantizedModelDataUrl);

        // ---- 4. Summary table + machine-readable results for merging with Python. ----
        Console.WriteLine("=== Scalability: elapsed ms per encode vs token count (all on max cores) ===");
        Console.WriteLine($"{"Tokens",8} | {"Pure fp32",12} | {"Pure Int8",12} | {"ONNX Q4F16",12} | {"ONNX Int8",12}");
        Console.WriteLine(new string('-', 70));
        for (int i = 0; i < inputs.Count; i++)
        {
            Console.WriteLine($"{tokenCounts[i],8} | {pureFp32[i],12:n1} | {pureInt8[i],12:n1} | {Cell(onnxQ4, i),12} | {Cell(onnxInt8, i),12}");
        }

        var resultsPath = "/tmp/harrier_scaling_csharp.json";
        await File.WriteAllTextAsync(resultsPath, JsonSerializer.Serialize(
            inputs.Select((inp, i) => new
            {
                target = inp.Target,
                tokens = inp.Tokens,
                pure_fp32_ms = pureFp32[i],
                pure_int8_ms = pureInt8[i],
                onnx_q4f16_ms = onnxQ4?[i],
                onnx_int8_ms = onnxInt8?[i],
            }),
            new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine();
        Console.WriteLine($"Wrote C# results -> {resultsPath}");

        static string Cell(double[] col, int i) => col is null ? "n/a" : col[i].ToString("n1");
    }

    private static async Task<double[]> TimeColumnAsync(string label, Func<string[], Task<float[][]>> encode, string[] texts, int[] tokens)
    {
        Console.WriteLine($"=== {label} ===");
        var ms = new double[texts.Length];
        for (int i = 0; i < texts.Length; i++)
        {
            ms[i] = await TimeEncodeAsync(encode, texts[i], tokens[i]);
        }
        Console.WriteLine();
        return ms;
    }

    /// <summary>Downloads (if needed) and times an ONNX variant; returns null if it cannot be loaded.
    /// ONNX Runtime uses every core by default, so no ParallelOptions plumbing is needed here.</summary>
    private static async Task<double[]> TimeOnnxAsync(string label, string[] texts, int[] tokens, string modelUrl, string dataUrl)
    {
        try
        {
            Console.WriteLine($"Loading {label} ...");
            using var onnx = await SentenceTransformers.Harrier.Small.SentenceEncoder.CreateAsync(
                modelUrl: modelUrl,
                modelDataUrl: dataUrl,
                downloadToPath: modelUrl is null
                    ? null
                    : Path.Combine(Path.GetTempPath(), "SentenceTransformers.Harrier.Small.Int8", "model.onnx"),
                reportProgress: Progress("onnx-weights"));
            Console.WriteLine();
            return await TimeColumnAsync(label, s => onnx.EncodeAsync(s), texts, tokens);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"{label} failed: {ex.Message}");
            Console.WriteLine();
            return null;
        }
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
            ms[i] = await TimeEncodeAsync(s => pure.EncodeAsync(s, MaxCores), inputs[i].Text, inputs[i].Tokens);
        }

        var resultsPath = $"/tmp/harrier_scaling_pure_{quant}.json";
        await File.WriteAllTextAsync(resultsPath, JsonSerializer.Serialize(
            inputs.Select((inp, i) => new { tokens = inp.Tokens, pure_ms = ms[i] }),
            new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine();
        Console.WriteLine($"Wrote results -> {resultsPath}");
    }

    /// <summary>
    /// Correctness gate: encodes the 5 multilingual reference sentences with the pure encoder (fp32 and
    /// Int8) and compares each embedding to the PyTorch golden reference via cosine similarity. Also
    /// writes the pure embeddings to <c>/tmp/harrier_verify_{quant}_{tag}.json</c> so a before/after run
    /// can confirm an optimization did not move the output beyond float-reassociation noise.
    /// </summary>
    public static async Task RunVerifyAsync(string tag)
    {
        // The golden reference lives in the test project's Resources.
        string[] candidates =
        {
            Path.Combine(AppContext.BaseDirectory, "harrier-small-reference.json"),
            "/home/user/sentence-transformers-sharp/SentenceTransformers.Test/Resources/harrier-small-reference.json",
        };
        var refPath = candidates.FirstOrDefault(File.Exists);
        if (refPath is null)
        {
            Console.WriteLine($"Reference JSON not found (looked in: {string.Join(", ", candidates)}).");
            return;
        }

        using var refDoc = JsonDocument.Parse(await File.ReadAllTextAsync(refPath));
        var sentences = refDoc.RootElement.GetProperty("sentences").EnumerateArray().Select(e => e.GetString()!).ToArray();
        var golden = refDoc.RootElement.GetProperty("embeddings").EnumerateArray()
            .Select(row => row.EnumerateArray().Select(x => x.GetSingle()).ToArray()).ToArray();

        foreach (var quant in new[]
        {
            SentenceTransformers.Harrier.Small.Pure.Model.Quantization.None,
            SentenceTransformers.Harrier.Small.Pure.Model.Quantization.Int8,
        })
        {
            using var enc = await SentenceTransformers.Harrier.Small.Pure.SentenceEncoder.CreateAsync(quantization: quant);
            var emb = await enc.EncodeAsync(sentences, MaxCores);

            Console.WriteLine($"=== verify {quant} ({tag}) vs PyTorch golden ===");
            double minCos = 1.0, sumCos = 0;
            for (int i = 0; i < sentences.Length; i++)
            {
                double cos = Cosine(emb[i], golden[i]);
                minCos = Math.Min(minCos, cos);
                sumCos += cos;
                Console.WriteLine($"  sent[{i}] cos(golden) = {cos:F6}  | {Trim(sentences[i])}");
            }
            Console.WriteLine($"  min={minCos:F6} avg={sumCos / sentences.Length:F6}");
            Console.WriteLine();

            var outPath = $"/tmp/harrier_verify_{quant}_{tag}.json";
            await File.WriteAllTextAsync(outPath, JsonSerializer.Serialize(
                emb.Select(v => v.Select(x => (double)x).ToArray()).ToArray()));
        }
    }

    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return dot / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12);
    }

    private static string Trim(string s) => s.Length <= 28 ? s : s.Substring(0, 28) + "…";

    /// <summary>
    /// Per-stage profile of the pure fp32 forward pass at a few sequence lengths, at a chosen degree of
    /// parallelism. Shows which stages scale linearly (projection matmuls) vs quadratically (attention).
    /// </summary>
    public static async Task RunProfileAsync(int maxDop)
    {
        const string inputsPath = "/tmp/harrier_scaling_inputs.json";
        if (!File.Exists(inputsPath))
        {
            Console.WriteLine($"Calibrated inputs not found at {inputsPath}; run 'harrier-scaling' first.");
            return;
        }

        using var docp = JsonDocument.Parse(await File.ReadAllTextAsync(inputsPath));
        var byTokens = docp.RootElement.EnumerateArray()
            .ToDictionary(e => e.GetProperty("tokens").GetInt32(), e => e.GetProperty("text").GetString()!);

        Console.WriteLine($"Loading Harrier.Small.Pure (fp32) for profiling (MaxDop={maxDop}) ...");
        using var pure = await SentenceTransformers.Harrier.Small.Pure.SentenceEncoder.CreateAsync(
            quantization: SentenceTransformers.Harrier.Small.Pure.Model.Quantization.None);
        Console.WriteLine();

        var po = new ParallelOptions { MaxDegreeOfParallelism = maxDop };

        foreach (var tokens in new[] { 512, 2048, 4096 })
        {
            if (!byTokens.TryGetValue(tokens, out var text)) continue;
            var batch = new[] { text };

            // Warmup with profiling off.
            SentenceTransformers.Harrier.Small.Pure.Model.ForwardProfile.Enabled = false;
            await pure.EncodeAsync(batch, po);

            // One profiled encode.
            SentenceTransformers.Harrier.Small.Pure.Model.ForwardProfile.Reset();
            SentenceTransformers.Harrier.Small.Pure.Model.ForwardProfile.Enabled = true;
            await pure.EncodeAsync(batch, po);
            SentenceTransformers.Harrier.Small.Pure.Model.ForwardProfile.Enabled = false;
            SentenceTransformers.Harrier.Small.Pure.Model.ForwardProfile.Report($"{tokens} tokens, MaxDop={maxDop}");
            Console.WriteLine();
        }
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
