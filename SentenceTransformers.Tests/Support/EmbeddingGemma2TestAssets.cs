using System.Collections.Concurrent;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics.X86;
using System.Text.Json;
using SentenceTransformers.EmbeddingGemma2;

namespace SentenceTransformers.Tests.Support;

/// <summary>
/// Locates the EmbeddingGemma 2 assets the opt-in tests need. Nothing is downloaded by default: the
/// model tests run when the <c>.litertlm</c> bundles are already present in
/// <c>EMBEDDINGGEMMA2_MODELS_DIR</c> (or the package's default cache path) and silently pass otherwise.
/// Set <c>EMBEDDINGGEMMA2_DOWNLOAD=1</c> to let them download missing bundles from Hugging Face.
/// The per-layer reference tests additionally need dumps written by
/// <c>scripts/generate_embeddinggemma2_reference.py</c>; point <c>EMBEDDINGGEMMA2_REFERENCE_DIR</c> at one
/// or more dump directories (separated by the platform path separator).
/// </summary>
internal static class EmbeddingGemma2TestAssets
{
    public static string FixturesDir => Path.Combine(AppContext.BaseDirectory, "Resources", "embeddinggemma2");

    public static string Fixture(string name) => Path.Combine(FixturesDir, name);

    private static readonly ConcurrentDictionary<EmbeddingGemma2Model, Lazy<string>> _paths = new();
    private static readonly ConcurrentDictionary<EmbeddingGemma2Model, Lazy<SentenceEncoder>> _encoders = new();

    /// <summary>The bundle path, or null when it is not available (the calling test should return early).</summary>
    public static string ModelPath(EmbeddingGemma2Model model) => _paths.GetOrAdd(model, m => new Lazy<string>(() => Resolve(m))).Value;

    /// <summary>A shared encoder per bundle (loading the 740M bundle takes seconds), or null when unavailable.</summary>
    public static SentenceEncoder Encoder(EmbeddingGemma2Model model)
    {
        if (ModelPath(model) is null)
        {
            return null;
        }
        return _encoders.GetOrAdd(model, m => new Lazy<SentenceEncoder>(() => SentenceEncoder.LoadAsync(ModelPath(m), model: m).GetAwaiter().GetResult())).Value;
    }

    private static string Resolve(EmbeddingGemma2Model model)
    {
        var file = EmbeddingGemma2Models.GetFileName(model);
        var dir = Environment.GetEnvironmentVariable("EMBEDDINGGEMMA2_MODELS_DIR");
        if (!string.IsNullOrEmpty(dir) && File.Exists(Path.Combine(dir, file)))
        {
            return Path.Combine(dir, file);
        }
        var cached = EmbeddingGemma2Models.GetDefaultCachePath(model);
        if (File.Exists(cached) && new FileInfo(cached).Length == EmbeddingGemma2Models.GetFileSize(model))
        {
            return cached;
        }
        if (Environment.GetEnvironmentVariable("EMBEDDINGGEMMA2_DOWNLOAD") == "1")
        {
            SentenceEncoder.DownloadModelAsync(model, cached).GetAwaiter().GetResult();
            return cached;
        }
        return null;
    }

    public static double Cosine(ReadOnlySpan<float> a, ReadOnlySpan<float> b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            d += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return d / Math.Sqrt(na * nb);
    }

    /// <summary>max |a − b| / max |b|.</summary>
    public static double MaxRelative(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double m = 0, r = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            m = Math.Max(m, Math.Abs((double)actual[i] - expected[i]));
            r = Math.Max(r, Math.Abs(expected[i]));
        }
        return m / Math.Max(r, 1e-30);
    }

    /// <summary>‖a − b‖ / ‖b‖.</summary>
    public static double RelativeRms(ReadOnlySpan<float> actual, ReadOnlySpan<float> expected)
    {
        Assert.Equal(expected.Length, actual.Length);
        double se = 0, sr = 0;
        for (int i = 0; i < actual.Length; i++)
        {
            double d = (double)actual[i] - expected[i];
            se += d * d;
            sr += (double)expected[i] * expected[i];
        }
        return Math.Sqrt(se / Math.Max(sr, 1e-30));
    }

    /// <summary>
    /// The port reproduces the LiteRT-LM engine bit for bit where it executes the same instructions as the
    /// engine's XNNPACK AVX-512 kernels (x64 with AVX-512VL and FMA). Elsewhere the reciprocal square root
    /// seed and fused multiply-adds can differ in the last bit, which int8 activation quantization can
    /// amplify, so only cosine agreement is checked.
    /// </summary>
    public static bool ExpectBitExactEngineParity => Avx512F.VL.IsSupported && Fma.IsSupported;

    /// <summary>Asserts <paramref name="actual"/> equals the engine's embedding exactly on
    /// <see cref="ExpectBitExactEngineParity"/> hardware, and has cosine ≥ <paramref name="minCosine"/> elsewhere.</summary>
    public static void AssertMatchesEngine(float[] actual, float[] expected, string label, double minCosine = 0.995)
    {
        Assert.Equal(expected.Length, actual.Length);
        if (ExpectBitExactEngineParity)
        {
            double maxAbs = 0;
            int differing = 0;
            for (int i = 0; i < actual.Length; i++)
            {
                if (BitConverter.SingleToInt32Bits(actual[i]) != BitConverter.SingleToInt32Bits(expected[i]))
                {
                    differing++;
                    maxAbs = Math.Max(maxAbs, Math.Abs((double)actual[i] - expected[i]));
                }
            }
            Assert.True(differing == 0, $"{label}: {differing} of {actual.Length} values differ from the engine (max abs {maxAbs:E2}, cosine {Cosine(actual, expected):F7})");
        }
        else
        {
            double cos = Cosine(actual, expected);
            Assert.True(cos >= minCosine, $"{label}: cosine {cos:F5}");
        }
    }

    public static float[] ReadFloats(JsonElement array) => array.EnumerateArray().Select(e => e.GetSingle()).ToArray();
}

/// <summary>Reads the tensors dumped by <c>scripts/generate_embeddinggemma2_reference.py</c> (manifest.json + raw files).</summary>
internal sealed class EmbeddingGemma2Reference
{
    private readonly Dictionary<string, string> _files = new(StringComparer.Ordinal);

    private static readonly Lazy<EmbeddingGemma2Reference> _instance = new(Load);

    /// <summary>The merged dumps, or null when <c>EMBEDDINGGEMMA2_REFERENCE_DIR</c> is not set.</summary>
    public static EmbeddingGemma2Reference Instance => _instance.Value;

    private static EmbeddingGemma2Reference Load()
    {
        var dirs = Environment.GetEnvironmentVariable("EMBEDDINGGEMMA2_REFERENCE_DIR");
        if (string.IsNullOrEmpty(dirs))
        {
            return null;
        }
        var r = new EmbeddingGemma2Reference();
        foreach (var dir in dirs.Split(Path.PathSeparator, StringSplitOptions.RemoveEmptyEntries))
        {
            var manifest = Path.Combine(dir, "manifest.json");
            if (!File.Exists(manifest))
            {
                continue;
            }
            var root = JsonDocument.Parse(File.ReadAllText(manifest)).RootElement;
            foreach (var t in root.GetProperty("tensors").EnumerateObject())
            {
                r._files[t.Name] = Path.Combine(dir, t.Value.GetProperty("file").GetString());
            }
        }
        return r;
    }

    public bool Has(string name) => _files.ContainsKey(name);

    public float[] F(string name) => MemoryMarshal.Cast<byte, float>(File.ReadAllBytes(_files[name])).ToArray();

    public int[] I(string name) => MemoryMarshal.Cast<byte, int>(File.ReadAllBytes(_files[name])).ToArray();
}
