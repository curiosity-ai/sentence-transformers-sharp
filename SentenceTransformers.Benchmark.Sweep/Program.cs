using System.Diagnostics;
using System.Text;
using System.Text.Json;
using BERTTokenizers.Base;
using SentenceTransformers;

// Token-count scalability sweep across every available sentence-encoder model. For each model it sweeps
// from 128 tokens up to that model's own context-window limit (powers of two), encodes a single input of
// each length on all cores, and reports time taken plus input/output throughput in MB/s (à la the
// SentenceTransformers.Benchmark project). Run:
//
//   dotnet run -c Release --project SentenceTransformers.Benchmark.Sweep
//   dotnet run -c Release --project SentenceTransformers.Benchmark.Sweep -- harrier    # substring filter
//
// Pure-managed encoders are constructed with a max-core default ParallelOptions so EncodeAsync uses every
// core; ONNX Runtime already does. Inputs are generated to a target word count and the actual (possibly
// truncated) token count is measured and reported.

var filter = args.Length > 0 ? args[0] : null;
int[] targets = { 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768 };
var maxCore = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

Console.WriteLine($"Sentence-encoder token-count sweep (ProcessorCount={Environment.ProcessorCount})");
Console.WriteLine($"Per model: 128 tokens up to the model's context window. Times + MB/s, single encode, all cores.");
Console.WriteLine();

var models = new List<ModelDef>
{
    new("MiniLM-L6-v2 (ONNX)", () =>
    {
        var e = new SentenceTransformers.MiniLM.SentenceEncoder();
        return Task.FromResult(new Runner(s => e.EncodeAsync(s), e.Tokenizer, e.MaxChunkLength, e));
    }),
    new("ArcticXs (ONNX)", () =>
    {
        var e = new SentenceTransformers.ArcticXs.SentenceEncoder();
        return Task.FromResult(new Runner(s => e.EncodeAsync(s), e.Tokenizer, e.MaxChunkLength, e));
    }),
    new("Qwen3-0.6B (ONNX)", async () =>
    {
        var e = await SentenceTransformers.Qwen3.SentenceEncoder.CreateAsync();
        return new Runner(s => e.EncodeAsync(s), e.Tokenizer, e.MaxChunkLength, e);
    }),
    new("Harrier.Small.Pure fp32", async () =>
    {
        var e = await SentenceTransformers.Harrier.Small.Pure.SentenceEncoder.CreateAsync(
            quantization: SentenceTransformers.Harrier.Small.Pure.Model.Quantization.None, parallelOptions: maxCore);
        return new Runner(s => e.EncodeAsync(s), e.Tokenizer, e.MaxChunkLength, e);
    }),
    new("Harrier.Small.Pure Int8", async () =>
    {
        var e = await SentenceTransformers.Harrier.Small.Pure.SentenceEncoder.CreateAsync(
            quantization: SentenceTransformers.Harrier.Small.Pure.Model.Quantization.Int8, parallelOptions: maxCore);
        return new Runner(s => e.EncodeAsync(s), e.Tokenizer, e.MaxChunkLength, e);
    }),
    new("Harrier.Small ONNX Q4F16", async () =>
    {
        var e = await SentenceTransformers.Harrier.Small.SentenceEncoder.CreateAsync();
        return new Runner(s => e.EncodeAsync(s), e.Tokenizer, e.MaxChunkLength, e);
    }),
    new("Harrier.Small ONNX Int8", async () =>
    {
        var e = await SentenceTransformers.Harrier.Small.SentenceEncoder.CreateAsync(
            modelUrl: SentenceTransformers.Harrier.Small.SentenceEncoder.Quantizations.QuantizedModelUrl,
            modelDataUrl: SentenceTransformers.Harrier.Small.SentenceEncoder.Quantizations.QuantizedModelDataUrl,
            downloadToPath: Path.Combine(Path.GetTempPath(), "SentenceTransformers.Harrier.Small.Int8", "model.onnx"));
        return new Runner(s => e.EncodeAsync(s), e.Tokenizer, e.MaxChunkLength, e);
    }),
    new("Harrier.Medium ONNX Q4F16", async () =>
    {
        var e = await SentenceTransformers.Harrier.Medium.SentenceEncoder.CreateAsync();
        return new Runner(s => e.EncodeAsync(s), e.Tokenizer, e.MaxChunkLength, e);
    }),
};

var rows = new List<Row>();

foreach (var model in models)
{
    if (filter is not null && !model.Name.Contains(filter, StringComparison.OrdinalIgnoreCase))
    {
        continue;
    }

    Console.WriteLine($"=== {model.Name} ===");
    try
    {
        var r = await model.Create();
        using (r.Disposable)
        {
            int CountTokens(string s) => r.Tokenizer.Encode(new[] { s })[0].InputIds.Length;

            foreach (var target in targets)
            {
                if (target > r.MaxTokens) break; // past this model's context window

                // Generate ~target words; ratio is ~1 token/word for these tokenizers, and the encoder
                // truncates at the context window, so the measured token count lands near the target.
                var text = string.Join(' ', Enumerable.Range(0, target).Select(Pool.Word));
                int tokens = Math.Min(CountTokens(text), r.MaxTokens);
                long inBytes = Encoding.UTF8.GetByteCount(text);

                try
                {
                    var (ms, dim) = await TimeEncodeAsync(r.Encode, text, tokens);
                    double sec = ms / 1000.0;
                    double inMBs = inBytes / 1024.0 / 1024.0 / sec;
                    double outMBs = (long)dim * sizeof(float) / 1024.0 / 1024.0 / sec;
                    double tokPerSec = tokens / sec;
                    rows.Add(new Row(model.Name, tokens, dim, ms, inMBs, outMBs, tokPerSec));
                    Console.WriteLine($"  {tokens,6} tok: {ms,10:n1} ms | {inMBs,7:n3} MB/s in | {outMBs,7:n3} MB/s out | {tokPerSec,8:n1} tok/s");
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"  {tokens,6} tok: failed ({ex.GetType().Name}: {ex.Message.Split('\n')[0]})");
                }
            }
        }
    }
    catch (Exception ex)
    {
        Console.WriteLine($"  skipped ({ex.GetType().Name}: {ex.Message.Split('\n')[0]})");
    }
    Console.WriteLine();
    GC.Collect();
    GC.WaitForPendingFinalizers();
}

PrintTable(rows);

// Unique per-filter path so a per-model driver (each model in its own process, to isolate OS OOM-kills
// of the 0.6B models at long contexts) does not overwrite other models' results.
var slug = filter is null ? "all" : new string(filter.Where(char.IsLetterOrDigit).ToArray()).ToLowerInvariant();
var outPath = $"/tmp/sweep_{slug}.json";
await File.WriteAllTextAsync(outPath, JsonSerializer.Serialize(rows, new JsonSerializerOptions { WriteIndented = true }));
Console.WriteLine();
Console.WriteLine($"Wrote {outPath}");


// -------------------- helpers --------------------

static async Task<(double Ms, int Dim)> TimeEncodeAsync(Func<string[], Task<float[][]>> encode, string text, int tokens)
{
    var batch = new[] { text };

    // Warm up only for the cheaper points; an O(n^2) encode at 16k-32k tokens is minutes long.
    if (tokens <= 2048)
    {
        await encode(batch);
    }

    int iterations =
        tokens <= 256  ? 5 :
        tokens <= 1024 ? 3 :
        tokens <= 2048 ? 2 : 1;

    int dim = 0;
    var samples = new double[iterations];
    for (int i = 0; i < iterations; i++)
    {
        var sw = Stopwatch.StartNew();
        var emb = await encode(batch);
        sw.Stop();
        samples[i] = sw.Elapsed.TotalMilliseconds;
        dim = emb.Length > 0 ? emb[0].Length : 0;
    }
    Array.Sort(samples);
    return (samples[samples.Length / 2], dim);
}

static void PrintTable(List<Row> rows)
{
    if (rows.Count == 0) return;

    int nameW = Math.Max(24, rows.Max(r => r.Model.Length));
    Console.WriteLine("=== Results (single encode, all cores) ===");
    Console.WriteLine($"{"Model".PadRight(nameW)} | {"Tokens",7} | {"Dim",5} | {"ms",10} | {"MB/s in",8} | {"MB/s out",8} | {"tok/s",9}");
    Console.WriteLine(new string('-', nameW + 64));
    foreach (var r in rows)
    {
        Console.WriteLine($"{r.Model.PadRight(nameW)} | {r.Tokens,7} | {r.Dim,5} | {r.Ms,10:n1} | {r.InMBs,8:n3} | {r.OutMBs,8:n3} | {r.TokPerSec,9:n1}");
    }
}

internal sealed record ModelDef(string Name, Func<Task<Runner>> Create);

internal sealed record Runner(
    Func<string[], Task<float[][]>> Encode,
    TokenizerBase                   Tokenizer,
    int                             MaxTokens,
    IDisposable                     Disposable);

internal sealed record Row(string Model, int Tokens, int Dim, double Ms, double InMBs, double OutMBs, double TokPerSec);

internal static class Pool
{
    private static readonly string[] _words =
    {
        "curiosity", "mosaik", "search", "embedding", "vector", "database", "ranking", "hybrid",
        "context", "token", "runtime", "inference", "latency", "throughput", "quantization",
        "semantic", "retrieval", "document", "chunk", "pipeline", "provider", "benchmark",
        "scalability", "transformer", "attention", "encoder", "multilingual", "normalize",
    };

    public static string Word(int i) => _words[i % _words.Length];
}
