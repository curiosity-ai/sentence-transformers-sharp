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
        int d = adapter.Dimension;

        // Collect the (pre-normalization) pooled embeddings of every unique training text.
        var seen = new HashSet<string>(StringComparer.Ordinal);
        var rows = new List<float[]>();
        void Add(string s) { if (s != null && seen.Add(s)) rows.Add(tuned.PooledPreNormalize(s)); }
        foreach (var p in train.Pairs) { Add(p.Anchor); Add(p.Positive); }
        if (rows.Count < 2) return;

        var mean = new float[d];
        foreach (var r in rows) for (int i = 0; i < d; i++) mean[i] += r[i];
        for (int i = 0; i < d; i++) mean[i] /= rows.Count;

        // Covariance (double for stability).
        var cov = new double[d * d];
        foreach (var r in rows)
        {
            ct.ThrowIfCancellationRequested();
            for (int i = 0; i < d; i++)
            {
                double di = r[i] - mean[i];
                if (di == 0) continue;
                int b = i * d;
                for (int j = i; j < d; j++) cov[b + j] += di * (r[j] - mean[j]);
            }
        }
        for (int i = 0; i < d; i++)
            for (int j = i; j < d; j++) { double v = cov[i * d + j] / rows.Count; cov[i * d + j] = v; cov[j * d + i] = v; }

        var (eigVals, eigVecs) = JacobiEigen(cov, d, ct);

        // W = V diag(1/sqrt(lambda+eps)) Vᵀ   (ZCA whitening; symmetric).
        var scale = new double[d];
        for (int i = 0; i < d; i++) scale[i] = 1.0 / Math.Sqrt(Math.Max(eigVals[i], 0) + eps);

        var W = new float[d * d];
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++)
            {
                double sum = 0;
                for (int k = 0; k < d; k++) sum += eigVecs[i * d + k] * scale[k] * eigVecs[j * d + k];
                W[i * d + j] = (float)sum;
            }

        adapter.WhiteningMean = mean;
        adapter.WhiteningMatrix = W;
    }

    // Classic cyclic Jacobi eigenvalue algorithm for a symmetric matrix. Returns (eigenvalues,
    // eigenvectors) with eigenvectors stored column-wise: V[i*d + k] is component i of eigenvector k.
    private static (double[] vals, double[] vecs) JacobiEigen(double[] aIn, int d, CancellationToken ct)
    {
        var a = (double[])aIn.Clone();
        var v = new double[d * d];
        for (int i = 0; i < d; i++) v[i * d + i] = 1.0;

        for (int sweep = 0; sweep < 100; sweep++)
        {
            ct.ThrowIfCancellationRequested();
            double off = 0;
            for (int p = 0; p < d; p++) for (int q = p + 1; q < d; q++) off += a[p * d + q] * a[p * d + q];
            if (off < 1e-18) break;

            for (int p = 0; p < d; p++)
            {
                for (int q = p + 1; q < d; q++)
                {
                    double apq = a[p * d + q];
                    if (Math.Abs(apq) < 1e-300) continue;
                    double app = a[p * d + p], aqq = a[q * d + q];
                    double phi = 0.5 * Math.Atan2(2 * apq, aqq - app);
                    double c = Math.Cos(phi), s = Math.Sin(phi);

                    for (int k = 0; k < d; k++)
                    {
                        double akp = a[k * d + p], akq = a[k * d + q];
                        a[k * d + p] = c * akp - s * akq;
                        a[k * d + q] = s * akp + c * akq;
                    }
                    for (int k = 0; k < d; k++)
                    {
                        double apk = a[p * d + k], aqk = a[q * d + k];
                        a[p * d + k] = c * apk - s * aqk;
                        a[q * d + k] = s * apk + c * aqk;
                    }
                    for (int k = 0; k < d; k++)
                    {
                        double vkp = v[k * d + p], vkq = v[k * d + q];
                        v[k * d + p] = c * vkp - s * vkq;
                        v[k * d + q] = s * vkp + c * vkq;
                    }
                }
            }
        }

        var vals = new double[d];
        for (int i = 0; i < d; i++) vals[i] = a[i * d + i];
        return (vals, v);
    }
}
