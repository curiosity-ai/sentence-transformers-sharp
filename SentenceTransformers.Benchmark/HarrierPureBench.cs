using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SentenceTransformers.Harrier.Small.Pure;
using SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// End-to-end encode timing for the Pure encoder, comparing the load-time <c>Int8</c> path against
/// <c>.stq</c> files. Every mode runs inside one process invocation, so a change of host between
/// runs cannot masquerade as a change in the code.
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
///
/// <para><b>Host capabilities are printed on every run.</b> The kernels dispatch on what the CPU
/// supports - 256- against 512-bit, VNNI against a widening fallback - so a host that quietly changed
/// between runs would otherwise read as a code regression.</para>
/// </summary>
public static class HarrierPureBench
{
    private const int WarmupRounds = 3;
    private const int TimedRounds = 5;

    /// <param name="args">Optional <c>.stq</c> paths to measure alongside <c>Int8</c>;
    /// <c>--profile</c> prints the packed-matmul stage breakdown per mode, and <c>--ab-rotation</c>
    /// measures both sides of the rotation fan-out decision in this same process.</param>
    public static async Task RunAsync(params string[] args)
    {
        var stqPaths = (args ?? Array.Empty<string>()).Where(a => !a.StartsWith("--", StringComparison.Ordinal)).ToList();
        bool profile = (args ?? Array.Empty<string>()).Contains("--profile");
        bool abRotation = (args ?? Array.Empty<string>()).Contains("--ab-rotation");
        bool abSharing = (args ?? Array.Empty<string>()).Contains("--ab-sharing");
        bool abThreads = (args ?? Array.Empty<string>()).Contains("--ab-fastpath");

        // The library is consumed single-threaded in production - callers parallelize above it - so
        // the thread count is explicit here rather than always ProcessorCount.
        int threadsIdx = Array.IndexOf(args ?? Array.Empty<string>(), "--threads");
        if (threadsIdx >= 0 && threadsIdx + 1 < args.Length)
        {
            _threads = int.Parse(args[threadsIdx + 1]);
            stqPaths.Remove(args[threadsIdx + 1]);
        }
        if (stqPaths.Count == 0 && Environment.GetEnvironmentVariable("HARRIER_STQ_PATH") is { Length: > 0 } fromEnv)
        {
            stqPaths.Add(fromEnv);
        }

        PrintHost();

        var corpus = BuildCorpus(count: 192, wordsPerSentence: 18, seed: 20260918);
        Console.WriteLine($"Corpus: {corpus.Length} distinct sentences, ~{corpus[0].Split(' ').Length} words each");
        Console.WriteLine($"Warm-up rounds: {WarmupRounds}, timed rounds: {TimedRounds}, MaxDegreeOfParallelism: {_threads}");
        Console.WriteLine();
        Console.WriteLine($"  {"mode",-26} {"load s",8} {"ms/iter",10} {"emb/s",9} {"par",5} {"MB res.",9} {"MB alloc/it",12} {"GC 0/1/2",10}");

        // fp32 and Int4 are deliberately not run. fp32 is three times slower than anything else here
        // and Int4 is superseded by the .stq 4-bit path, so both only cost wall clock on every
        // iteration of this benchmark. Uncomment either to re-measure it.
        //   await MeasureAsync("None", corpus, () => SentenceEncoder.CreateAsync(quantization: Quantization.None, parallelOptions: Options()), profile);
        //   await MeasureAsync("Int4", corpus, () => SentenceEncoder.CreateAsync(quantization: Quantization.Int4, parallelOptions: Options()), profile);
        await MeasureAsync("Int8", corpus, () => SentenceEncoder.CreateAsync(quantization: Quantization.Int8, parallelOptions: Options()), profile);

        foreach (var stqPath in stqPaths)
        {
            if (!File.Exists(stqPath))
            {
                Console.WriteLine($"  (skipping .stq: '{stqPath}' not found)");
                continue;
            }

            string name = Path.GetFileNameWithoutExtension(stqPath);

            if (abThreads)
            {
                // A/B the single-thread fast path in this one process.
                try
                {
                    StqMatrix.SingleThreadFastPath = false;
                    await MeasureAsync($"stq {name} via ForAsync", corpus,
                        () => SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: Options()), profile);

                    StqMatrix.SingleThreadFastPath = true;
                    await MeasureAsync($"stq {name} inline", corpus,
                        () => SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: Options()), profile);
                }
                finally
                {
                    StqMatrix.SingleThreadFastPath = true;
                }
                continue;
            }

            if (abSharing)
            {
                // A/B rotating once per projection group against once per projection, in this one
                // process, because the difference is small enough to be swamped by run-to-run spread.
                try
                {
                    Gemma3Model.ShareGroupRotation = false;
                    await MeasureAsync($"stq {name} rot/proj", corpus,
                        () => SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: Options()), profile);

                    Gemma3Model.ShareGroupRotation = true;
                    await MeasureAsync($"stq {name} rot/group", corpus,
                        () => SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: Options()), profile);
                }
                finally
                {
                    Gemma3Model.ShareGroupRotation = true;
                }
                continue;
            }

            if (!abRotation)
            {
                await MeasureAsync($"stq {name}", corpus,
                    () => SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: Options()), profile);
                continue;
            }

            // --ab-rotation: measure both sides of the rotation fan-out decision inside this one
            // process, so the two variants cannot be separated by a host change the way two separate
            // runs can. Off by default because it doubles the .stq runs.
            long defaultThreshold = StqMatrix.RotationParallelThreshold;
            try
            {
                StqMatrix.RotationParallelThreshold = long.MaxValue;   // always inline
                await MeasureAsync($"stq {name} rot=inline", corpus,
                    () => SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: Options()), profile);

                StqMatrix.RotationParallelThreshold = 0;               // always parallel
                await MeasureAsync($"stq {name} rot=parallel", corpus,
                    () => SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: Options()), profile);
            }
            finally
            {
                StqMatrix.RotationParallelThreshold = defaultThreshold;
            }
        }
    }

    /// <summary>Prints what the kernels will actually dispatch to on this host, so a silent change of
    /// machine is visible in the output rather than showing up as a mysterious regression.</summary>
    private static void PrintHost()
    {
        Console.WriteLine($"Host: {RuntimeInformation.OSArchitecture} / {RuntimeInformation.FrameworkDescription}");
        Console.WriteLine($"  cores {Environment.ProcessorCount}   server GC {System.Runtime.GCSettings.IsServerGC}");
        Console.WriteLine($"  vectors: Vector128 {Vector128.IsHardwareAccelerated}, Vector256 {Vector256.IsHardwareAccelerated}, " +
                          $"Vector512 {Vector512.IsHardwareAccelerated}, Vector<float> width {System.Numerics.Vector<float>.Count}");
        Console.WriteLine($"  x86: Avx2 {Avx2.IsSupported}, Avx512F {Avx512F.IsSupported}, Avx512BW {Avx512BW.IsSupported}, " +
                          $"AvxVnni {AvxVnni.IsSupported}, AvxVnniInt8 {AvxVnniInt8.IsSupported}, AvxVnniInt8.V512 {AvxVnniInt8.V512.IsSupported}");
        Console.WriteLine($"  kernels: Vnni.IsSupported {Vnni.IsSupported}, Vnni.Use512 {Vnni.Use512}");
        Console.WriteLine();
    }

    /// <summary>
    /// Attributes one encode's time to the packed-matmul stages. Warm-up and the profiled pass use
    /// disjoint halves of the corpus, so the profiled pass cannot be served from the vector cache.
    /// </summary>
    public static async Task ProfileAsync(string stqPath, int maxDop)
    {
        PrintHost();

        if (string.IsNullOrWhiteSpace(stqPath) || !File.Exists(stqPath))
        {
            Console.WriteLine($"No .stq file to profile ('{stqPath}').");
            return;
        }

        var corpus = BuildCorpus(count: 128, wordsPerSentence: 18, seed: 20260918);
        var warm = corpus[..64];
        var profiled = corpus[64..];
        var po = new ParallelOptions { MaxDegreeOfParallelism = maxDop };

        using var encoder = await SentenceEncoder.LoadQuantizedAsync(stqPath, parallelOptions: po);

        ForwardProfile.Enabled = false;
        for (int i = 0; i < WarmupRounds; i++)
        {
            await encoder.EncodeAsync(warm, po);
        }

        ForwardProfile.ResetStages();
        ForwardProfile.Enabled = true;
        await encoder.EncodeAsync(profiled, po);
        ForwardProfile.Enabled = false;
        ForwardProfile.ReportStages($"stq {Path.GetFileNameWithoutExtension(stqPath)}, {profiled.Length} sentences, MaxDop={maxDop}");
        Console.WriteLine();
    }

    private static int _threads = Environment.ProcessorCount;

    private static ParallelOptions Options() => new() { MaxDegreeOfParallelism = _threads };

    private static async Task MeasureAsync(string label, string[] corpus, Func<Task<SentenceEncoder>> create, bool profile)
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

        // Allocation rate matters as much as time: anything sizeable allocated per encode should be
        // coming from a pool instead, and the gen-0 count is the tell.
        long allocBefore = GC.GetTotalAllocatedBytes(precise: true);
        int g0 = GC.CollectionCount(0), g1 = GC.CollectionCount(1), g2 = GC.CollectionCount(2);

        // CPU time over wall time is the average number of cores actually busy. It answers directly
        // whether a mode is using the threads it was given, rather than leaving it to be inferred
        // from the ParallelOptions that were passed in.
        var cpuBefore = Process.GetCurrentProcess().TotalProcessorTime;
        var sw = Stopwatch.StartNew();
        for (int i = 0; i < TimedRounds; i++)
        {
            await encoder.EncodeAsync(corpus, Options());
        }
        sw.Stop();
        double parallelism = (Process.GetCurrentProcess().TotalProcessorTime - cpuBefore).TotalMilliseconds / sw.Elapsed.TotalMilliseconds;

        double allocMbPerIter = (GC.GetTotalAllocatedBytes(precise: true) - allocBefore) / 1024.0 / 1024.0 / TimedRounds;
        string gc = $"{GC.CollectionCount(0) - g0}/{GC.CollectionCount(1) - g1}/{GC.CollectionCount(2) - g2}";
        double ms = sw.Elapsed.TotalMilliseconds / TimedRounds;

        Console.WriteLine($"  {label,-26} {loadSw.Elapsed.TotalSeconds,8:F1} {ms,10:F1} {1000.0 * corpus.Length / ms,9:F1} " +
                          $"{parallelism,5:F1} {managedMb,9:F0} {allocMbPerIter,12:F1} {gc,10}");

        if (profile)
        {
            ForwardProfile.ResetStages();
            ForwardProfile.Reset();
            ForwardProfile.Enabled = true;
            await encoder.EncodeAsync(corpus, Options());
            ForwardProfile.Enabled = false;
            // Both reports, because they answer different questions: the named stages cover the whole
            // forward pass and so are comparable between modes, while the Stage enum only the packed
            // kernel emits and breaks its matmul down internally.
            ForwardProfile.Report(label);
            ForwardProfile.ReportStages(label);
            Console.WriteLine();
        }
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
