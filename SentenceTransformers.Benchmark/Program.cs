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

var syncEncoders = new (string Name, ISentenceEncoder Encoder)[]
{
    ("MiniLM-L6-v2", new SentenceTransformers.MiniLM.SentenceEncoder()),
    ("ArcticXs",     new SentenceTransformers.ArcticXs.SentenceEncoder()),
};

foreach (var (name, encoder) in syncEncoders)
{
    using (encoder)
    {
        var run = EncoderBench.Run(name, encoder, texts, cfg);
        results.Add(run);
        Console.WriteLine();
    }
}

using (var qwen3Encoder = await SentenceTransformers.Qwen3.SentenceEncoder.CreateAsync())
{
    var run = EncoderBench.Run("Qwen3-0.6B", qwen3Encoder, texts, cfg);
    results.Add(run);
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
    long InputKB,
    long OutputKB,
    double InputMBPerHour,
    double OutputMBPerHour,
    double EmbeddingsPerHour
);

public static class EncoderBench
{
    public static BenchResult Run(string name, ISentenceEncoder encoder, string[] corpus, BenchConfig cfg)
    {
        // Build a batch by repeating corpus
        var batch = new string[cfg.BatchSize];
        for (int i = 0; i < batch.Length; i++)
        {
            batch[i] = corpus[i % corpus.Length];
        }

        // Warmup
        for (int i = 0; i < cfg.WarmupIters; i++)
        {
            var warm = encoder.Encode(batch);
            if (warm.Length == 0)
            {
                throw new Exception("Warmup produced no embeddings.");
            }
        }

        // Measure
        long inputBytes = 0;
        for (int i = 0; i < cfg.Iterations; i++)
        {
            foreach (var s in batch)
            {
                inputBytes += Encoding.UTF8.GetByteCount(s);
            }
        }

        var sw = Stopwatch.StartNew();
        float[][] last = Array.Empty<float[]>();
        for (int i = 0; i < cfg.Iterations; i++)
        {
            last = encoder.Encode(batch);
        }
        sw.Stop();

        int dim = (last.Length > 0) ? last[0].Length : 0;
        long outputBytes = (long)cfg.Iterations * cfg.BatchSize * dim * sizeof(float);

        double seconds = sw.Elapsed.TotalSeconds;
        double inputMBperHour  = 3600 * inputBytes / 1024.0/ 1024.0 / seconds;
        double outputMBperHour = 3600 * outputBytes / 1024.0 / 1024.0 / seconds;
        double embeddingsPerHour = 3600 * (cfg.Iterations * (double)cfg.BatchSize) / seconds;

        Console.WriteLine($"[{name}]");
        Console.WriteLine($"  Batch:             {cfg.BatchSize}");
        Console.WriteLine($"  Dim:               {dim}");
        Console.WriteLine($"  Time:              {sw.Elapsed.TotalSeconds:n1} s for {cfg.Iterations} iterations");
        Console.WriteLine($"  Avg:               {sw.Elapsed.TotalSeconds / cfg.Iterations:n1} s per iteration");
        Console.WriteLine($"  Input throughput:  {inputMBperHour:n1} UTF-8 MB per hour");
        Console.WriteLine($"  Output throughput: {outputMBperHour:n1} MB per hour");
        Console.WriteLine($"  Vector throughput: {embeddingsPerHour:n1} per hour");

        return new BenchResult(name, cfg.BatchSize, dim, cfg.Iterations, sw.Elapsed.TotalMilliseconds, inputBytes, outputBytes, inputMBperHour, outputMBperHour, embeddingsPerHour);
    }

    public static void PrintTable(List<BenchResult> results)
    {
        if (results.Count == 0)
        {
            return;
        }

        var headers = new[]
        {
            "Model",
            "Dim",
            "Batch",
            "ms/iter",
            "Emb/hour",
            "In MB/hour",
            "Out MB/hour"
        };

        var rows = results.Select(r => new[]
        {
            r.Name,
            r.Dimensions.ToString(),
            r.BatchSize.ToString(),
            (r.TotalSeconds / r.Iterations).ToString("n1"),
            r.EmbeddingsPerHour.ToString("n1"),
            r.InputMBPerHour.ToString("n1"),
            r.OutputMBPerHour.ToString("n1")
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

        Console.WriteLine("=== Benchmark Results ===");
        Console.WriteLine(string.Join(" | ", headers.Select((h, i) => Pad(h, widths[i]))));
        Console.WriteLine(string.Join("-+-", widths.Select(w => new string('-', w))));
        foreach (var row in rows)
        {
            Console.WriteLine(string.Join(" | ", row.Select((c, i) => Pad(c, widths[i]))));
        }
    }
}
