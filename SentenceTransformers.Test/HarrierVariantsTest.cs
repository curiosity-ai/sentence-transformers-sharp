using System.Reflection;
using System.Text.Json;
using SentenceTransformers;
using HarrierMedium = SentenceTransformers.Harrier.Medium;
using HarrierSmall = SentenceTransformers.Harrier.Small;

/// <summary>
/// Integration test: downloads every published quantization variant for both Harrier Medium
/// (0.6b) and Harrier Small (270m), loads each, encodes a small batch of multilingual
/// sentences, and reports two things per variant:
///   1. The pairwise cosine similarities between the encoded sentences, so quantization quality
///      can be eyeballed.
///   2. The per-sentence cosine similarity against golden reference embeddings produced by the
///      original PyTorch / sentence-transformers implementation (see
///      scripts/generate_harrier_reference.py and Resources/harrier-*-reference.json). This is a
///      correctness check: a faithful port reproduces the reference embeddings almost exactly
///      (fp32/fp16), and even the aggressively quantized variants should stay well-correlated.
///      Any variant whose worst sentence falls below its parity threshold fails the test, which
///      surfaces implementation mistakes (tokenization, pooling, input building, normalization).
/// </summary>
public static class HarrierVariantsTest
{
    // Keep in sync with scripts/generate_harrier_reference.py (used to build the reference files).
    private static readonly string[] Sentences =
    {
        "Good morning, how are you?",      // English
        "Buenos días, ¿cómo estás?",       // Spanish
        "おはようございます、お元気ですか？",  // Japanese
        "Hello world",                     // unrelated
        "The cat sat on the mat",          // unrelated
    };

    private static int _failures;

    public static async Task RunAsync()
    {
        // Both SentenceTransformers.Harrier.Medium and SentenceTransformers.Harrier.Small ship a
        // Resources/tokenizer.json that copies to the same path in the consumer's output, and
        // they collide non-deterministically. Delete the colliding file so each encoder falls
        // back to its own embedded tokenizer (extracted under Path.GetTempPath() per package).
        var collidingTokenizer = Path.Combine(AppContext.BaseDirectory, "Resources", "tokenizer.json");
        if (File.Exists(collidingTokenizer))
        {
            File.Delete(collidingTokenizer);
            Console.WriteLine($"(removed colliding {collidingTokenizer})");
        }

        var smallReference = LoadReference("harrier-small-reference.json");
        var mediumReference = LoadReference("harrier-medium-reference.json");

        Console.WriteLine("=== Harrier Small (270m) variants ===");
        foreach (var (label, modelUrl, dataUrl, parityThreshold) in HarrierSmallVariants())
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
                smallReference,
                parityThreshold);
        }

        Console.WriteLine();
        Console.WriteLine("=== Harrier Medium (0.6b) variants ===");
        foreach (var (label, modelUrl, dataUrls, parityThreshold) in HarrierMediumVariants())
        {
            await RunVariantAsync(
                label,
                async () =>
                {
                    var dir = Path.Combine(Path.GetTempPath(), "SentenceTransformers.Harrier.Medium.Variants", label);
                    Directory.CreateDirectory(dir);
                    var graphFile = Path.GetFileName(new Uri(modelUrl).LocalPath);
                    var path = Path.Combine(dir, graphFile);
                    // Manually pre-fetch every weight file under its upstream filename, since
                    // CreateAsync only fetches one companion file.
                    await HarrierMedium.SentenceEncoder.DownloadModelAsync(modelUrl, path);
                    foreach (var d in dataUrls)
                    {
                        var dataPath = Path.Combine(dir, Path.GetFileName(new Uri(d).LocalPath));
                        await HarrierMedium.SentenceEncoder.DownloadModelAsync(d, dataPath);
                    }
                    return new HarrierMedium.SentenceEncoder(modelOnnxPath: path);
                },
                mediumReference,
                parityThreshold);
        }

        Console.WriteLine();
        if (_failures > 0)
        {
            throw new Exception($"{_failures} Harrier variant(s) fell below their parity threshold against the PyTorch reference embeddings. See the PARITY FAIL lines above.");
        }
        Console.WriteLine("All Harrier variants passed the reference parity check.");
    }

    private static async Task RunVariantAsync(string label, Func<Task<ISentenceEncoder>> factory, Reference reference, double parityThreshold)
    {
        Console.WriteLine($"--- {label} ---");
        try
        {
            using var enc = await factory();
            var vectors = await enc.EncodeAsync(Sentences);
            Console.WriteLine($"  embedding dim: {vectors[0].Length}");

            // Print every pairwise cosine similarity so quantization quality is visible.
            for (int i = 0; i < vectors.Length; i++)
            {
                for (int j = i + 1; j < vectors.Length; j++)
                {
                    var sim = Cosine(vectors[i], vectors[j]);
                    Console.WriteLine($"  sim({i},{j}) = {sim:F4}  ({Trim(Sentences[i])} vs {Trim(Sentences[j])})");
                }
            }

            // Compare each sentence's embedding to the PyTorch reference for the same sentence.
            CompareToReference(label, vectors, reference, parityThreshold);
        }
        catch (Exception e)
        {
            Console.WriteLine($"  FAILED: {e.Message}");
            throw;
        }
    }

    /// <summary>
    /// Reports the cosine similarity between each encoded sentence and its golden reference
    /// embedding, and records a failure if the worst sentence is below <paramref name="parityThreshold"/>.
    /// </summary>
    private static void CompareToReference(string label, float[][] vectors, Reference reference, double parityThreshold)
    {
        if (vectors.Length != reference.Embeddings.Length)
        {
            throw new Exception($"variant produced {vectors.Length} vectors but the reference has {reference.Embeddings.Length}.");
        }
        if (vectors[0].Length != reference.Dim)
        {
            throw new Exception($"variant embedding dim {vectors[0].Length} does not match reference dim {reference.Dim}.");
        }

        double min = double.PositiveInfinity, sum = 0;
        Console.WriteLine($"  parity vs PyTorch reference ({reference.Model}):");
        for (int i = 0; i < vectors.Length; i++)
        {
            var cos = Cosine(vectors[i], reference.Embeddings[i]);
            sum += cos;
            if (cos < min) min = cos;
            Console.WriteLine($"    ref-cos[{i}] = {cos:F4}  ({Trim(Sentences[i])})");
        }
        var mean = sum / vectors.Length;
        var pass = min >= parityThreshold;
        if (!pass) _failures++;
        Console.WriteLine($"    -> min={min:F4} mean={mean:F4} threshold={parityThreshold:F3}  {(pass ? "PARITY OK" : "PARITY FAIL")}");
    }

    // Each variant carries the minimum acceptable per-sentence cosine similarity against the
    // PyTorch reference. fp32/fp16 are near-exact; q4* and int8-style quantization legitimately
    // drift more, so their thresholds are looser - but still high enough to catch a real bug.
    private static IEnumerable<(string Label, string ModelUrl, string DataUrl, double ParityThreshold)> HarrierSmallVariants() => new[]
    {
        ("full",          HarrierSmall.SentenceEncoder.Quantizations.FullModelUrl,    HarrierSmall.SentenceEncoder.Quantizations.FullModelDataUrl,    0.998),
        ("fp16",          HarrierSmall.SentenceEncoder.Quantizations.Fp16ModelUrl,    HarrierSmall.SentenceEncoder.Quantizations.Fp16ModelDataUrl,    0.998),
        ("q4",            HarrierSmall.SentenceEncoder.Quantizations.Q4ModelUrl,      HarrierSmall.SentenceEncoder.Quantizations.Q4ModelDataUrl,      0.95),
        ("q4f16",         HarrierSmall.SentenceEncoder.Quantizations.Q4Fp16ModelUrl,  HarrierSmall.SentenceEncoder.Quantizations.Q4Fp16ModelDataUrl,  0.95),
        ("quantized",     HarrierSmall.SentenceEncoder.Quantizations.QuantizedModelUrl, HarrierSmall.SentenceEncoder.Quantizations.QuantizedModelDataUrl, 0.95),
    };

    private static IEnumerable<(string Label, string ModelUrl, string[] DataUrls, double ParityThreshold)> HarrierMediumVariants() => new[]
    {
        ("full",      HarrierMedium.SentenceEncoder.Quantizations.FullModelUrl,      new[] { HarrierMedium.SentenceEncoder.Quantizations.FullModelDataUrl, HarrierMedium.SentenceEncoder.Quantizations.FullModelDataUrl2 }, 0.998),
        ("fp16",      HarrierMedium.SentenceEncoder.Quantizations.Fp16ModelUrl,      new[] { HarrierMedium.SentenceEncoder.Quantizations.Fp16ModelDataUrl }, 0.998),
        ("q4",        HarrierMedium.SentenceEncoder.Quantizations.Q4ModelUrl,        new[] { HarrierMedium.SentenceEncoder.Quantizations.Q4ModelDataUrl }, 0.95),
        ("q4f16",     HarrierMedium.SentenceEncoder.Quantizations.Q4Fp16ModelUrl,    new[] { HarrierMedium.SentenceEncoder.Quantizations.Q4Fp16ModelDataUrl }, 0.95),
        ("quantized", HarrierMedium.SentenceEncoder.Quantizations.QuantizedModelUrl, new[] { HarrierMedium.SentenceEncoder.Quantizations.QuantizedModelDataUrl }, 0.95),
    };

    /// <summary>Reads a reference-embedding JSON file embedded under Resources/ and validates its sentences.</summary>
    private static Reference LoadReference(string fileName)
    {
        var asm = Assembly.GetExecutingAssembly();
        var resourceName = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(fileName, StringComparison.Ordinal))
            ?? throw new FileNotFoundException($"Embedded reference '{fileName}' not found. Did you run scripts/generate_harrier_reference.py?");
        using var stream = asm.GetManifestResourceStream(resourceName)!;
        var reference = JsonSerializer.Deserialize<Reference>(stream, new JsonSerializerOptions { PropertyNameCaseInsensitive = true })
            ?? throw new InvalidOperationException($"Failed to parse reference '{fileName}'.");

        if (reference.Sentences.Length != Sentences.Length)
        {
            throw new InvalidOperationException($"Reference '{fileName}' has {reference.Sentences.Length} sentences but the test uses {Sentences.Length}.");
        }
        for (int i = 0; i < Sentences.Length; i++)
        {
            if (!string.Equals(reference.Sentences[i], Sentences[i], StringComparison.Ordinal))
            {
                throw new InvalidOperationException(
                    $"Reference '{fileName}' sentence[{i}] does not match the test sentence. Regenerate with scripts/generate_harrier_reference.py.");
            }
        }
        return reference;
    }

    private sealed record Reference(string Model, int Dim, string[] Sentences, float[][] Embeddings);

    /// <summary>Cosine similarity. The encoders L2-normalize their output, but divide by the norms
    /// anyway so the comparison is correct even if a variant's output drifts off the unit sphere.</summary>
    private static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        var denom = Math.Sqrt(na) * Math.Sqrt(nb);
        return denom > 0 ? dot / denom : 0;
    }

    private static string Trim(string s) => s.Length > 30 ? s.Substring(0, 27) + "..." : s;
}
