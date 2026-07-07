namespace SentenceTransformers.Training;

/// <summary>Retrieval quality metrics for a set of positive pairs.</summary>
/// <param name="Count">Number of pairs evaluated.</param>
/// <param name="Accuracy">Top-1 accuracy: fraction of anchors whose nearest candidate is their own positive.</param>
/// <param name="MRR">Mean reciprocal rank of the true positive among all candidates.</param>
public readonly record struct RetrievalMetrics(int Count, float Accuracy, float MRR);

/// <summary>
/// Model-agnostic evaluation utilities for <see cref="ISentenceEncoder"/>s (base or
/// <see cref="AdaptedSentenceEncoder">adapted</see>). Used both to report training progress and to
/// compare a fine-tuned adapter against its frozen baseline on a held-out split.
/// </summary>
public static class EmbeddingEvaluation
{
    /// <summary>
    /// STS-style Spearman correlation between each pair's cosine similarity (as produced by
    /// <paramref name="encoder"/>) and its gold <see cref="SentencePair.Score"/>. Pairs without a score
    /// are ignored; returns NaN when fewer than two scored pairs are available.
    /// </summary>
    public static async Task<float> SpearmanAsync(ISentenceEncoder encoder, SentencePairDataset data, CancellationToken cancellationToken = default)
    {
        var scored = data.Pairs.Where(p => p.Score.HasValue).ToArray();
        if (scored.Length < 2) return float.NaN;

        var cache = await EmbedAsync(encoder, scored, cancellationToken);

        var sims   = new float[scored.Length];
        var scores = new float[scored.Length];
        for (int i = 0; i < scored.Length; i++)
        {
            sims[i]   = EmbeddingMath.Dot(cache[scored[i].Anchor], cache[scored[i].Positive]);
            scores[i] = scored[i].Score.Value;
        }

        return EmbeddingMath.Spearman(sims, scores);
    }

    /// <summary>
    /// In-batch retrieval metrics over <paramref name="data"/>: every pair's positive is scored against
    /// every other pair's positive, and we measure how often (and how highly) each anchor ranks its own
    /// positive first. When <paramref name="minScore"/> is given, only pairs at or above it are used.
    /// </summary>
    public static async Task<RetrievalMetrics> RetrievalAsync(ISentenceEncoder encoder, SentencePairDataset data, float? minScore = null, CancellationToken cancellationToken = default)
    {
        var pairs = data.Pairs.Where(p => !minScore.HasValue || !p.Score.HasValue || p.Score.Value >= minScore.Value).ToArray();
        int n     = pairs.Length;
        if (n < 2) return new RetrievalMetrics(n, float.NaN, float.NaN);

        var cache = await EmbedAsync(encoder, pairs, cancellationToken);

        var anchors    = new float[n][];
        var candidates = new float[n][];
        for (int i = 0; i < n; i++)
        {
            anchors[i]    = EmbeddingMath.Normalized(cache[pairs[i].Anchor]);
            candidates[i] = EmbeddingMath.Normalized(cache[pairs[i].Positive]);
        }

        int    correct = 0;
        double mrrSum  = 0;
        for (int i = 0; i < n; i++)
        {
            float self  = EmbeddingMath.Dot(anchors[i], candidates[i]);
            int   rank   = 1; // rank of the true positive (1 = best)
            int   argmax = i;
            float best   = self;

            for (int j = 0; j < n; j++)
            {
                if (j == i) continue;
                float sim = EmbeddingMath.Dot(anchors[i], candidates[j]);
                if (sim > self) rank++;
                if (sim > best) { best = sim; argmax = j; }
            }

            if (argmax == i) correct++;
            mrrSum += 1.0 / rank;
        }

        return new RetrievalMetrics(n, (float)correct / n, (float)(mrrSum / n));
    }

    private static async Task<Dictionary<string, float[]>> EmbedAsync(ISentenceEncoder encoder, SentencePair[] pairs, CancellationToken cancellationToken)
    {
        var unique = new List<string>();
        var index  = new Dictionary<string, float[]>(StringComparer.Ordinal);

        void Register(string s)
        {
            s ??= string.Empty;
            if (!index.ContainsKey(s)) { index[s] = null; unique.Add(s); }
        }

        foreach (var p in pairs) { Register(p.Anchor); Register(p.Positive); }

        const int batch = 64;
        for (int start = 0; start < unique.Count; start += batch)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int end   = Math.Min(start + batch, unique.Count);
            var slice = unique.GetRange(start, end - start).ToArray();
            var vecs  = await encoder.EncodeAsync(slice, cancellationToken);
            for (int i = 0; i < slice.Length; i++) index[slice[i]] = vecs[i];
        }

        return index;
    }
}
