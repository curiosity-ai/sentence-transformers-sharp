using System.Diagnostics;
using System.Text;
using SentenceTransformers;


var qwen3Task = SentenceTransformers.Qwen3.SentenceEncoder.CreateAsync();

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
    WarmupIters: 10,
    MeasureIters: 50
);

var results = new List<BenchResult>();

var syncEncoders = new (string Name, ISentenceEncoder Encoder)[]
{
    ("MiniLM-L6-v2", new SentenceTransformers.MiniLM.SentenceEncoder()),
    ("ArcticXs",     new SentenceTransformers.ArcticXs.SentenceEncoder()),
};

foreach (var (name, encoder) in syncEncoders)
{
    using var _ = encoder as IDisposable;
    if (_ is null)
    {
        throw new InvalidOperationException($"Encoder '{name}' does not implement IDisposable.");
    }
    var run = EncoderBench.Run(name, encoder, texts, cfg);
    results.Add(run);
    Console.WriteLine();
}

using (var qwen3Encoder = await qwen3Task)
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

public sealed record BenchConfig(int BatchSize, int WarmupIters, int MeasureIters);

public sealed record BenchResult(
    string Name,
    int BatchSize,
    int EmbeddingDim,
    int MeasureIters,
    double TotalMs,
    long InputBytes,
    long OutputBytes,
    double InputBytesPerSec,
    double OutputBytesPerSec,
    double EmbeddingsPerSec,
    long WorkingSetDeltaBytes,
    long GcHeapDeltaBytes
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

        // Measure load/steady memory deltas around encode loop (rough but useful)
        GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, blocking: true, compacting: true);
        var proc = Process.GetCurrentProcess();
        proc.Refresh();
        long wsBefore = proc.WorkingSet64;
        long heapBefore = GC.GetTotalMemory(forceFullCollection: true);

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
        for (int i = 0; i < cfg.MeasureIters; i++)
        {
            foreach (var s in batch)
            {
                inputBytes += Encoding.UTF8.GetByteCount(s);
            }
        }

        var sw = Stopwatch.StartNew();
        float[][] last = Array.Empty<float[]>();
        for (int i = 0; i < cfg.MeasureIters; i++)
        {
            last = encoder.Encode(batch);
        }
        sw.Stop();

        int dim = (last.Length > 0) ? last[0].Length : 0;
        long outputBytes = (long)cfg.MeasureIters * cfg.BatchSize * dim * sizeof(float);

        proc.Refresh();
        long wsAfter = proc.WorkingSet64;
        long heapAfter = GC.GetTotalMemory(forceFullCollection: false);

        double seconds = sw.Elapsed.TotalSeconds;
        double inBps = inputBytes / seconds;
        double outBps = outputBytes / seconds;
        double embPerSec = (cfg.MeasureIters * (double)cfg.BatchSize) / seconds;

        Console.WriteLine($"[{name}]");
        Console.WriteLine($"  Batch: {cfg.BatchSize}");
        Console.WriteLine($"  Dim:   {dim}");
        Console.WriteLine($"  Time:  {sw.Elapsed.TotalMilliseconds:F1} ms for {cfg.MeasureIters} iterations");
        Console.WriteLine($"  Avg:   {sw.Elapsed.TotalMilliseconds / cfg.MeasureIters:F2} ms/iter");
        Console.WriteLine($"  Input throughput (UTF-8 bytes/s): {inBps:F0}");
        Console.WriteLine($"  Output throughput (bytes/s):      {outBps:F0}");
        Console.WriteLine($"  Embeddings/s:                      {embPerSec:F2}");
        Console.WriteLine($"  WorkingSet Δ (bytes):             {wsAfter - wsBefore}");
        Console.WriteLine($"  GC heap Δ (bytes):                {heapAfter - heapBefore}");

        return new BenchResult(
            name, cfg.BatchSize, dim, cfg.MeasureIters,
            sw.Elapsed.TotalMilliseconds,
            inputBytes, outputBytes,
            inBps, outBps, embPerSec,
            wsAfter - wsBefore,
            heapAfter - heapBefore
        );
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
            "Emb/s",
            "In B/s",
            "Out B/s",
            "WS Δ",
            "GC Δ"
        };

        var rows = results.Select(r => new[]
        {
            r.Name,
            r.EmbeddingDim.ToString(),
            r.BatchSize.ToString(),
            (r.TotalMs / r.MeasureIters).ToString("F2"),
            r.EmbeddingsPerSec.ToString("F2"),
            r.InputBytesPerSec.ToString("F0"),
            r.OutputBytesPerSec.ToString("F0"),
            r.WorkingSetDeltaBytes.ToString(),
            r.GcHeapDeltaBytes.ToString(),
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
