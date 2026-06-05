using System.Diagnostics;
using System.Text;
using SentenceTransformers;


// ---- Build a few-paragraph input dataset ----
var texts = new[]
{
    MakeParagraph(seed: 1, paragraphs: 3, sentencesPerParagraph: 5),
    MakeParagraph(seed: 2, paragraphs: 3, sentencesPerParagraph: 5),
    MakeParagraph(seed: 3, paragraphs: 3, sentencesPerParagraph: 5),
    MakeParagraph(seed: 4, paragraphs: 3, sentencesPerParagraph: 5),
};

var cfg = new BenchConfig(
    BatchSize: 8,
    WarmupIters: 1,
    Iterations: 5
);

var results = new List<BenchResult>();

// Embedded models are ready immediately.
var syncEncoders = new (string Name, Func<ISentenceEncoder> Factory)[]
{
    ("MiniLM-L6-v2", () => new SentenceTransformers.MiniLM.SentenceEncoder()),
    ("ArcticXs",     () => new SentenceTransformers.ArcticXs.SentenceEncoder()),
};

foreach (var (name, factory) in syncEncoders)
{
    using var encoder = factory();
    results.Add(await EncoderBench.RunAsync(name, encoder, texts, cfg));
    Console.WriteLine();
}

// The BPE model packages (Qwen3, Harrier Medium, Harrier Small) each ship a
// Resources/tokenizer.json that copies to the SAME path in this project's output and
// collide non-deterministically. With the colliding file present, every BPE encoder loads
// whichever tokenizer won the copy race -> wrong vocab -> out-of-range token ids at
// inference. Delete it so each encoder falls back to its own embedded tokenizer (extracted
// to a per-package temp path). See SentenceTransformers.Test/HarrierVariantsTest.cs.
var collidingTokenizer = Path.Combine(AppContext.BaseDirectory, "Resources", "tokenizer.json");
if (File.Exists(collidingTokenizer))
{
    File.Delete(collidingTokenizer);
    Console.WriteLine($"(removed colliding {collidingTokenizer} so each BPE model uses its own tokenizer)");
    Console.WriteLine();
}

// Downloaded models: weights are fetched on first use, then cached on disk.
var asyncEncoders = new (string Name, Func<Task<ISentenceEncoder>> Factory)[]
{
    ("Qwen3-0.6B",     async () => await SentenceTransformers.Qwen3.SentenceEncoder.CreateAsync()),
    ("Harrier-Medium", async () => await SentenceTransformers.Harrier.Medium.SentenceEncoder.CreateAsync()),
    ("Harrier-Small",  async () => await SentenceTransformers.Harrier.Small.SentenceEncoder.CreateAsync()),
};

foreach (var (name, factory) in asyncEncoders)
{
    try
    {
        Console.WriteLine($"[{name}] downloading / loading model ...");
        using var encoder = await factory();
        results.Add(await EncoderBench.RunAsync(name, encoder, texts, cfg));
    }
    catch (Exception e)
    {
        Console.WriteLine($"[{name}] SKIPPED: {e.Message}");
    }
    Console.WriteLine();
}

EncoderBench.PrintTable(results);

GC.Collect();
GC.WaitForPendingFinalizers();
GC.Collect();

static string MakeParagraph(int seed, int paragraphs, int sentencesPerParagraph)
{
    var rng = new Random(seed);
    var vocab = new[]
    {
        "curiosity","mosaik","search","embedding","vector","database","ranking","hybrid",
        "context","token","runtime","inference","latency","throughput","quantization",
        "semantic","retrieval","document","chunk","pipeline","provider","benchmark"
    };

    string MakeSentence()
    {
        int words = rng.Next(10, 20);
        var sb = new StringBuilder();
        for (int i = 0; i < words; i++)
        {
            if (i > 0)
            {
                sb.Append(' ');
            }
            sb.Append(vocab[rng.Next(vocab.Length)]);
        }
        sb.Append('.');
        return sb.ToString();
    }

    var outSb = new StringBuilder();
    for (int p = 0; p < paragraphs; p++)
    {
        if (p > 0)
        {
            outSb.Append("\n\n");
        }
        for (int s = 0; s < sentencesPerParagraph; s++)
        {
            if (s > 0)
            {
                outSb.Append(' ');
            }
            outSb.Append(MakeSentence());
        }
    }
    return outSb.ToString();
}

public sealed record BenchConfig(int BatchSize, int WarmupIters, int Iterations);

public sealed record BenchResult(
    string Name,
    int BatchSize,
    int Dimensions,
    int Iterations,
    double TotalSeconds,
    long TokensPerBatch,
    double TokensPerSecond,
    double MsPerIteration
);

public static class EncoderBench
{
    public static async Task<BenchResult> RunAsync(string name, ISentenceEncoder encoder, string[] corpus, BenchConfig cfg)
    {
        // Build a batch by repeating corpus
        var batch = new string[cfg.BatchSize];
        for (int i = 0; i < batch.Length; i++)
        {
            batch[i] = corpus[i % corpus.Length];
        }

        // Count the (non-padding) tokens the model actually processes for this batch.
        // Using each model's own tokenizer + attention mask keeps the metric comparable
        // even though the families tokenize differently.
        long tokensPerBatch = CountTokens(encoder, batch);

        // Warmup
        for (int i = 0; i < cfg.WarmupIters; i++)
        {
            var warm = await encoder.EncodeAsync(batch);
            if (warm.Length == 0)
            {
                throw new Exception("Warmup produced no embeddings.");
            }
        }

        // Measure
        var sw = Stopwatch.StartNew();
        float[][] last = Array.Empty<float[]>();
        for (int i = 0; i < cfg.Iterations; i++)
        {
            last = await encoder.EncodeAsync(batch);
        }
        sw.Stop();

        int dim = (last.Length > 0) ? last[0].Length : 0;
        double seconds = sw.Elapsed.TotalSeconds;
        long totalTokens = tokensPerBatch * cfg.Iterations;
        double tokensPerSecond = totalTokens / seconds;
        double msPerIteration = sw.Elapsed.TotalMilliseconds / cfg.Iterations;

        Console.WriteLine($"[{name}]");
        Console.WriteLine($"  Batch:             {cfg.BatchSize}");
        Console.WriteLine($"  Dim:               {dim}");
        Console.WriteLine($"  Tokens/batch:      {tokensPerBatch:n0}");
        Console.WriteLine($"  Time:              {seconds:n2} s for {cfg.Iterations} iterations");
        Console.WriteLine($"  Avg:               {msPerIteration:n1} ms/iter");
        Console.WriteLine($"  Throughput:        {tokensPerSecond:n1} tokens/s");

        return new BenchResult(name, cfg.BatchSize, dim, cfg.Iterations, seconds, tokensPerBatch, tokensPerSecond, msPerIteration);
    }

    private static long CountTokens(ISentenceEncoder encoder, string[] batch)
    {
        long total = 0;
        foreach (var (_, _, attentionMask) in encoder.Tokenizer.Encode(batch))
        {
            foreach (var m in attentionMask)
            {
                total += m;
            }
        }
        return total;
    }

    public static void PrintTable(List<BenchResult> results)
    {
        if (results.Count == 0)
        {
            return;
        }

        // Ratios are expressed relative to MiniLM (the smallest/fastest baseline).
        var baseline = results.FirstOrDefault(r => r.Name.StartsWith("MiniLM", StringComparison.OrdinalIgnoreCase))
                       ?? results[0];

        var headers = new[]
        {
            "Model",
            "Dim",
            "Batch",
            "ms/iter",
            "Tokens/s",
            "vs MiniLM"
        };

        var rows = results.Select(r => new[]
        {
            r.Name,
            r.Dimensions.ToString(),
            r.BatchSize.ToString(),
            r.MsPerIteration.ToString("n1"),
            r.TokensPerSecond.ToString("n1"),
            (r.TokensPerSecond / baseline.TokensPerSecond).ToString("n2") + "x"
        }).ToList();

        var widths = new int[headers.Length];
        for (int i = 0; i < headers.Length; i++)
        {
            widths[i] = headers[i].Length;
            foreach (var row in rows)
            {
                if (row[i].Length > widths[i])
                {
                    widths[i] = row[i].Length;
                }
            }
        }

        string Pad(string s, int w) => s.PadRight(w);

        Console.WriteLine($"=== Benchmark Results (tokens/s, ratio vs {baseline.Name}) ===");
        Console.WriteLine(string.Join(" | ", headers.Select((h, i) => Pad(h, widths[i]))));
        Console.WriteLine(string.Join("-+-", widths.Select(w => new string('-', w))));
        foreach (var row in rows)
        {
            Console.WriteLine(string.Join(" | ", row.Select((c, i) => Pad(c, widths[i]))));
        }
    }
}
