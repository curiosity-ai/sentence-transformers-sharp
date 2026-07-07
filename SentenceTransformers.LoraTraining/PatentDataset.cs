using System.Globalization;
using System.Reflection;
using SentenceTransformers.Training;

namespace SentenceTransformers.LoraTraining;

/// <summary>
/// Loader for the <b>Google Patent Phrase Similarity</b> dataset, embedded in this assembly so it can be
/// used with no download. Each row pairs two short patent phrases (<c>anchor</c>, <c>target</c>) with a
/// human similarity <c>score</c> already in <c>[0,1]</c> (0.00, 0.25, 0.50, 0.75, 1.00) plus the CPC
/// technical <c>context</c> the phrases appear in.
///
/// <para>It is a good complement to STS-B: the phrases are terse, domain-specific technical terms, so a
/// general-purpose encoder has real headroom and adapter fine-tuning shows a clearer lift than it does on
/// the (nearly saturated) general-English STS-B.</para>
///
/// <para>Source: <see href="https://www.kaggle.com/datasets/google/google-patent-phrase-similarity-dataset"/>
/// (© Google, licensed CC BY 4.0).</para>
/// </summary>
internal static class PatentDataset
{
    public static readonly string[] Splits = { "train", "dev", "test" };

    /// <summary>Loads a patent split ("train", "dev"/"validation", or "test") from the embedded CSV.</summary>
    public static SentencePairDataset Load(string split)
    {
        string resourceSuffix = split.ToLowerInvariant() switch
        {
            "train"                => "patent-train.csv",
            "dev" or "validation"  => "patent-validation.csv",
            "test"                 => "patent-test.csv",
            _                      => throw new ArgumentException($"Unknown patent split '{split}'. Use train, dev or test."),
        };

        var pairs = new List<SentencePair>();

        using (var reader = OpenResource(resourceSuffix))
        {
            string line;
            bool   first = true;
            while ((line = reader.ReadLine()) is not null)
            {
                if (first) { first = false; continue; } // header: anchor,target,rating,score,context
                if (string.IsNullOrWhiteSpace(line)) continue;

                // The dataset is clean (no quoted/embedded commas), so a plain split is safe.
                var f = line.Split(',');
                if (f.Length != 5) continue;

                if (!float.TryParse(f[3], NumberStyles.Float, CultureInfo.InvariantCulture, out float score))
                {
                    continue;
                }

                pairs.Add(new SentencePair(f[0], f[1], Math.Clamp(score, 0f, 1f)));
            }
        }

        if (pairs.Count == 0)
        {
            throw new InvalidDataException($"No pairs parsed from embedded resource '{resourceSuffix}'.");
        }

        return new SentencePairDataset(pairs);
    }

    private static StreamReader OpenResource(string suffix)
    {
        var asm  = Assembly.GetExecutingAssembly();
        var name = asm.GetManifestResourceNames().FirstOrDefault(n => n.EndsWith(suffix, StringComparison.Ordinal))
                   ?? throw new InvalidOperationException($"Embedded resource ending in '{suffix}' not found.");
        return new StreamReader(asm.GetManifestResourceStream(name)!);
    }
}
