using SentenceTransformers.Bert.Pure.Model;
using SentenceTransformers.Training;

namespace SentenceTransformers.Bert.Pure.Training;

/// <summary>
/// Fits a ZCA whitening transform from the tuned training embeddings and stores it on the adapter, so at
/// inference the pooled vector becomes <c>u = normalize(W·(z − μ))</c>. Whitening counteracts the strong
/// anisotropy of pooled transformer embeddings (they occupy a narrow cone, which compresses all cosine
/// similarities) and is a classic, cheap boost for cosine-based STS. Uses a symmetric Jacobi eigensolver
/// (no native BLAS/LAPACK) — the covariance is only <c>dim×dim</c>.
/// </summary>
internal static class WhiteningFitter
{
    public static void Fit(SentenceEncoder tuned, SentencePairDataset train, LoraAdapter adapter, CancellationToken ct, float eps = 1e-3f)
    {
        // Collect the (pre-normalization) pooled embeddings of every unique training text.
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var rows = new List<float[]>();
        void Add(string s) { if (s != null && seen.Add(s)) rows.Add(tuned.PooledPreNormalize(s)); }
        foreach (var p in train.Pairs) { Add(p.Anchor); Add(p.Positive); }

        var (mean, matrix) = Whitening.Fit(rows, adapter.Dimension, ct, eps);
        if (mean == null) return;
        adapter.WhiteningMean = mean;
        adapter.WhiteningMatrix = matrix;
    }
}
