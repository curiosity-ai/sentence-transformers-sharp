namespace SentenceTransformers.Training;

/// <summary>
/// A related pair of texts used to fine-tune an encoder. <paramref name="Anchor"/> and
/// <paramref name="Positive"/> are two inputs that should map to nearby vectors (a query and a
/// relevant passage, a paraphrase pair, a question and its duplicate, and so on).
/// </summary>
/// <param name="Anchor">The first text of the pair (for retrieval use-cases, the query).</param>
/// <param name="Positive">The second, related text (for retrieval use-cases, the relevant passage).</param>
/// <param name="Score">
/// Optional graded similarity in <c>[0,1]</c> (1 = most similar). When present it is used for
/// STS-style Spearman evaluation; pairs are still treated as positives for contrastive training when
/// their score is at or above the configured threshold. Pass <c>null</c> for datasets that only
/// contain positive pairs.
/// </param>
public readonly record struct SentencePair(string Anchor, string Positive, float? Score = null);

/// <summary>
/// A collection of <see cref="SentencePair"/> plus a deterministic train / validation splitter.
/// </summary>
public sealed class SentencePairDataset
{
    private readonly List<SentencePair> _pairs;

    /// <summary>The pairs backing this dataset, in their original order.</summary>
    public IReadOnlyList<SentencePair> Pairs => _pairs;

    /// <summary>Number of pairs in the dataset.</summary>
    public int Count => _pairs.Count;

    /// <summary>True when every pair carries a similarity <see cref="SentencePair.Score"/>.</summary>
    public bool HasScores => _pairs.Count > 0 && _pairs.All(p => p.Score.HasValue);

    /// <summary>Creates a dataset from an existing sequence of pairs.</summary>
    public SentencePairDataset(IEnumerable<SentencePair> pairs)
    {
        _pairs = pairs?.ToList() ?? throw new ArgumentNullException(nameof(pairs));
    }

    /// <summary>
    /// Shuffles the pairs with a seeded RNG and splits them into a training set and a validation set.
    /// </summary>
    /// <param name="validationFraction">Fraction of pairs (0..1) to hold out for validation.</param>
    /// <param name="seed">Seed for the shuffle, so the split is reproducible.</param>
    public (SentencePairDataset Train, SentencePairDataset Validation) Split(float validationFraction = 0.1f, int seed = 12345)
    {
        if (validationFraction < 0f || validationFraction >= 1f)
        {
            throw new ArgumentOutOfRangeException(nameof(validationFraction), "Validation fraction must be in [0, 1).");
        }

        var shuffled = _pairs.ToArray();
        var rng      = new Random(seed);

        // Fisher-Yates shuffle for a reproducible, unbiased permutation.
        for (int i = shuffled.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (shuffled[i], shuffled[j]) = (shuffled[j], shuffled[i]);
        }

        int valCount = (int)Math.Round(shuffled.Length * validationFraction);

        // Keep at least one training pair when the dataset is non-empty.
        valCount = Math.Clamp(valCount, 0, Math.Max(0, shuffled.Length - 1));

        var validation = shuffled.Take(valCount).ToArray();
        var train      = shuffled.Skip(valCount).ToArray();

        return (new SentencePairDataset(train), new SentencePairDataset(validation));
    }
}
