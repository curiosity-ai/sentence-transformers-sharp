using System.Diagnostics;
using SentenceTransformers.Harrier.Small.Pure;
using SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// End-to-end encode timing for the Pure encoder across every weight precision it can run, including
/// an <c>.stq</c> file when one is supplied.
///
/// <para><b>Defeating the vector cache.</b> <see cref="SentenceEncoder.EncodeAsync(string[], CancellationToken)"/>
/// memoizes the last 16 encoded vectors by input hash. Re-encoding one small batch in a loop - the
/// obvious way to write this benchmark, and what it used to do - therefore measures a dictionary
/// lookup from the second iteration onwards, not inference. The corpus below is far larger than the
/// cache, so at most 16 of its entries can ever be served from it (under 5% here, and identically for
/// every mode).</para>
///
/// <para><b>Warm-up.</b> Each mode runs several full passes before timing starts, so the timed region
/// never includes JIT of the tiered-up kernels, the first-touch cost of the pooled scratch buffers, or
/// the page faults from first reading the weights. Tiered compilation promotes the hot matmul loops
/// only after they have run enough times, so a single warm-up pass is not enough.</para>
/// </summary>
public static class HarrierPureBench
{
    private const int WarmupRounds = 3;
    private const int TimedRounds = 5;

    /// <param name="args">Optional <c>.stq</c> paths to measure alongside the built-in modes. Pass
    /// <c>--quick</c> to skip the fp32 mode, which is three times slower than any other and dominates
    /// the wall clock when iterating on a kernel.</param>
    public static async Task RunAsync(params string[] args)
    {
        var stqPaths = (args ?? Array.Empty<string>()).Where(a => !a.StartsWith("--", StringComparison.Ordinal)).ToList();
        bool quick = (args ?? Array.Empty<string>()).Contains("--quick");
        if (stqPaths.Count == 0 && Environment.GetEnvironmentVariable("HARRIER_STQ_PATH") is { Length: > 0 } fromEnv)
        {
            stqPaths.Add(fromEnv);
        }

        var corpus = BuildCorpus(count: 192, wordsPerSentence: 18, seed: 20260918);

        Console.WriteLine($"Corpus: {corpus.Length} distinct sentences, ~{corpus[0].Split(' ').Length} words each");
        Console.WriteLine($"Warm-up rounds: {WarmupRounds}, timed rounds: {TimedRounds}, cores: {Environment.ProcessorCount}");
        Console.WriteLine();
        Console.WriteLine($"  {"mode",-26} {"load s",8} {"ms/iter",10} {"emb/s",9} {"MB res.",9}");

        var modes = quick
            ? new[] { Quantization.Int8, Quantization.Int4 }
            : new[] { Quantization.None, Quantization.Int8, Quantization.Int4 };

        foreach (var quant in modes)
        {
            await MeasureAsync(quant.ToString(), corpus,
                () => SentenceEncoder.CreateAsync(quantization: quant, parallelOptions: Options()));
        }

        foreach (var stqPath in stqPaths)
        {
            if (File.Exists(stqPath))
            {
                await MeasureAsync($"stq {Path.GetFileNameWithoutExtension(stqPath)}", corpus,
                    () => SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: Options()));
            }
            else
            {
                Console.WriteLine($"  (skipping .stq: '{stqPath}' not found)");
            }
        }
    }

    /// <summary>
    /// Attributes one encode's time to the forward-pass stages, for a given precision. The warm-up and
    /// the profiled pass use disjoint halves of the corpus, so the profiled pass cannot be served from
    /// the vector cache and every stage is measured on real work.
    /// </summary>
    public static async Task ProfileAsync(string stqPath, int maxDop)
    {
        var corpus = BuildCorpus(count: 128, wordsPerSentence: 18, seed: 20260918);
        var warm = corpus[..64];
        var profiled = corpus[64..];
        var po = new ParallelOptions { MaxDegreeOfParallelism = maxDop };

        async Task RunAsync(string label, SentenceEncoder encoder)
        {
            using (encoder)
            {
                ForwardProfile.Enabled = false;
                for (int i = 0; i < WarmupRounds; i++)
                {
                    await encoder.EncodeAsync(warm, po);
                }

                ForwardProfile.Reset();
                ForwardProfile.Enabled = true;
                await encoder.EncodeAsync(profiled, po);
                ForwardProfile.Enabled = false;
                ForwardProfile.Report($"{label}, {profiled.Length} sentences, MaxDop={maxDop}");
                Console.WriteLine();
            }
        }

        await RunAsync("Int4 (load-time)", await SentenceEncoder.CreateAsync(quantization: Quantization.Int4, parallelOptions: po));

        if (!string.IsNullOrWhiteSpace(stqPath) && File.Exists(stqPath))
        {
            await RunAsync($"stq {Path.GetFileNameWithoutExtension(stqPath)}", await SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: po));
        }
    }

    private static ParallelOptions Options() => new() { MaxDegreeOfParallelism = Environment.ProcessorCount };

    private static async Task MeasureAsync(string label, string[] corpus, Func<Task<SentenceEncoder>> create)
    {
        var loadSw = Stopwatch.StartNew();
        using var encoder = await create();
        loadSw.Stop();

        // Measure the resident cost of the weights themselves, after settling the heap.
        GC.Collect();
        GC.WaitForPendingFinalizers();
        GC.Collect();
        double managedMb = GC.GetTotalMemory(forceFullCollection: true) / 1024.0 / 1024.0;

        for (int i = 0; i < WarmupRounds; i++)
        {
            await encoder.EncodeAsync(corpus, Options());
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < TimedRounds; i++)
        {
            await encoder.EncodeAsync(corpus, Options());
        }
        sw.Stop();

        double ms = sw.Elapsed.TotalMilliseconds / TimedRounds;
        Console.WriteLine($"  {label,-26} {loadSw.Elapsed.TotalSeconds,8:F1} {ms,10:F1} {1000.0 * corpus.Length / ms,9:F1} {managedMb,9:F0}");
    }

    /// <summary>
    /// Builds a deterministic corpus of distinct sentences of near-constant token length. Synthetic
    /// rather than real prose so the benchmark stays self-contained and reproducible; what matters
    /// here is that every mode sees byte-identical input and that no two sentences repeat.
    /// </summary>
    private static string[] BuildCorpus(int count, int wordsPerSentence, int seed)
    {
        string[] vocabulary =
        {
            "retrieval", "embedding", "quantization", "ternary", "rotation", "kernel", "tensor", "matrix",
            "encoder", "attention", "gradient", "inference", "throughput", "latency", "vector", "index",
            "corpus", "sentence", "token", "weight", "scale", "group", "packed", "channel",
            "benchmark", "baseline", "accuracy", "precision", "recall", "ranking", "similarity", "distance",
        };

        var rng = new Random(seed);
        var sentences = new string[count];
        var seen = new HashSet<string>(StringComparer.Ordinal);
        for (int i = 0; i < count; i++)
        {
            string sentence;
            do
            {
                var words = new string[wordsPerSentence];
                for (int w = 0; w < wordsPerSentence; w++)
                {
                    words[w] = vocabulary[rng.Next(vocabulary.Length)];
                }
                sentence = string.Join(' ', words) + ".";
            }
            while (!seen.Add(sentence));   // distinct inputs only, or the vector cache would serve them
            sentences[i] = sentence;
        }
        return sentences;
    }
}
