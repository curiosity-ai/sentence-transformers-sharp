using SentenceTransformers;
using HarrierBig = SentenceTransformers.Harrier;
using HarrierSmall = SentenceTransformers.HarrierSmall;

/// <summary>
/// Integration test: downloads every published quantization variant for both Harrier
/// (0.6b) and Harrier Small (270m), loads each, encodes a small batch of multilingual
/// sentences, and prints the resulting embedding dimensions and a few cosine
/// similarities so the variants can be eyeballed against each other.
/// </summary>
public static class HarrierVariantsTest
{
    public static async Task RunAsync()
    {
        // Both SentenceTransformers.Harrier and SentenceTransformers.Harrier.Small ship a
        // Resources/tokenizer.json that copies to the same path in the consumer's output, and
        // they collide non-deterministically. Delete the colliding file so each encoder falls
        // back to its own embedded tokenizer (extracted under Path.GetTempPath() per package).
        var collidingTokenizer = Path.Combine(AppContext.BaseDirectory, "Resources", "tokenizer.json");
        if (File.Exists(collidingTokenizer))
        {
            File.Delete(collidingTokenizer);
            Console.WriteLine($"(removed colliding {collidingTokenizer})");
        }

        var sentences = new[]
        {
            "Good morning, how are you?",   // English
            "Buenos días, ¿cómo estás?",    // Spanish
            "おはようございます、お元気ですか？", // Japanese
            "Hello world",                   // unrelated
            "The cat sat on the mat",        // unrelated
        };

        Console.WriteLine("=== Harrier Small (270m) variants ===");
        foreach (var (label, modelUrl, dataUrl) in HarrierSmallVariants())
        {
            await RunVariantAsync(
                label,
                async () =>
                {
                    var dir = Path.Combine(Path.GetTempPath(), "SentenceTransformers.Harrier.Small.Variants", label);
                    Directory.CreateDirectory(dir);
                    var graphFile = Path.GetFileName(new Uri(modelUrl).LocalPath);
                    var path = Path.Combine(dir, graphFile);
                    return await HarrierSmall.SentenceEncoder.CreateAsync(modelUrl: modelUrl, modelDataUrl: dataUrl, downloadToPath: path);
                },
                sentences);
        }

        Console.WriteLine();
        Console.WriteLine("=== Harrier (0.6b) variants ===");
        foreach (var (label, modelUrl, dataUrls) in HarrierBigVariants())
        {
            await RunVariantAsync(
                label,
                async () =>
                {
                    var dir = Path.Combine(Path.GetTempPath(), "SentenceTransformers.Harrier.Variants", label);
                    Directory.CreateDirectory(dir);
                    var graphFile = Path.GetFileName(new Uri(modelUrl).LocalPath);
                    var path = Path.Combine(dir, graphFile);
                    // Manually pre-fetch every weight file under its upstream filename, since
                    // CreateAsync only fetches one companion file.
                    await HarrierBig.SentenceEncoder.DownloadModelAsync(modelUrl, path);
                    foreach (var d in dataUrls)
                    {
                        var dataPath = Path.Combine(dir, Path.GetFileName(new Uri(d).LocalPath));
                        await HarrierBig.SentenceEncoder.DownloadModelAsync(d, dataPath);
                    }
                    return new HarrierBig.SentenceEncoder(modelOnnxPath: path);
                },
                sentences);
        }
    }

    private static async Task RunVariantAsync(string label, Func<Task<ISentenceEncoder>> factory, string[] sentences)
    {
        Console.WriteLine($"--- {label} ---");
        try
        {
            using var enc = await factory();
            var vectors = await enc.EncodeAsync(sentences);
            Console.WriteLine($"  embedding dim: {vectors[0].Length}");
            // Print every pairwise cosine similarity so quantization quality is visible.
            for (int i = 0; i < vectors.Length; i++)
            {
                for (int j = i + 1; j < vectors.Length; j++)
                {
                    var sim = Dot(vectors[i], vectors[j]);
                    Console.WriteLine($"  sim({i},{j}) = {sim:F4}  ({Trim(sentences[i])} vs {Trim(sentences[j])})");
                }
            }
        }
        catch (Exception e)
        {
            Console.WriteLine($"  FAILED: {e.Message}");
            throw;
        }
    }

    private static IEnumerable<(string Label, string ModelUrl, string DataUrl)> HarrierSmallVariants() => new[]
    {
        ("full",          HarrierSmall.SentenceEncoder.Quantizations.FullModelUrl,    HarrierSmall.SentenceEncoder.Quantizations.FullModelDataUrl),
        ("fp16",          HarrierSmall.SentenceEncoder.Quantizations.Fp16ModelUrl,    HarrierSmall.SentenceEncoder.Quantizations.Fp16ModelDataUrl),
        ("q4",            HarrierSmall.SentenceEncoder.Quantizations.Q4ModelUrl,      HarrierSmall.SentenceEncoder.Quantizations.Q4ModelDataUrl),
        ("q4f16",         HarrierSmall.SentenceEncoder.Quantizations.Q4Fp16ModelUrl,  HarrierSmall.SentenceEncoder.Quantizations.Q4Fp16ModelDataUrl),
        ("quantized",     HarrierSmall.SentenceEncoder.Quantizations.QuantizedModelUrl, HarrierSmall.SentenceEncoder.Quantizations.QuantizedModelDataUrl),
    };

    private static IEnumerable<(string Label, string ModelUrl, string[] DataUrls)> HarrierBigVariants() => new[]
    {
        ("full",      HarrierBig.SentenceEncoder.Quantizations.FullModelUrl,      new[] { HarrierBig.SentenceEncoder.Quantizations.FullModelDataUrl, HarrierBig.SentenceEncoder.Quantizations.FullModelDataUrl2 }),
        ("fp16",      HarrierBig.SentenceEncoder.Quantizations.Fp16ModelUrl,      new[] { HarrierBig.SentenceEncoder.Quantizations.Fp16ModelDataUrl }),
        ("q4",        HarrierBig.SentenceEncoder.Quantizations.Q4ModelUrl,        new[] { HarrierBig.SentenceEncoder.Quantizations.Q4ModelDataUrl }),
        ("q4f16",     HarrierBig.SentenceEncoder.Quantizations.Q4Fp16ModelUrl,    new[] { HarrierBig.SentenceEncoder.Quantizations.Q4Fp16ModelDataUrl }),
        ("quantized", HarrierBig.SentenceEncoder.Quantizations.QuantizedModelUrl, new[] { HarrierBig.SentenceEncoder.Quantizations.QuantizedModelDataUrl }),
    };

    private static float Dot(float[] a, float[] b)
    {
        float s = 0f;
        for (int i = 0; i < a.Length; i++) s += a[i] * b[i];
        return s;
    }

    private static string Trim(string s) => s.Length > 30 ? s.Substring(0, 27) + "..." : s;
}
