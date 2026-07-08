namespace SentenceTransformers.Training;

/// <summary>
/// Fits a ZCA whitening transform from a set of embedding vectors: <c>W = V·diag(1/√(λ+eps))·Vᵀ</c> from
/// the eigendecomposition of the covariance, so a whitened vector is <c>W·(x − μ)</c>. Whitening counters
/// the strong anisotropy of pooled transformer embeddings (they occupy a narrow cone, compressing cosine
/// similarities) and is a classic cheap boost for cosine-based STS. Pure-managed (symmetric Jacobi
/// eigensolver — the covariance is only <c>dim×dim</c>); shared by the BERT and Gemma3 LoRA trainers.
/// </summary>
public static class Whitening
{
    /// <summary>Returns (mean, W row-major [dim,dim]) or (null, null) when there are &lt; 2 rows.</summary>
    public static (float[] mean, float[] matrix) Fit(IReadOnlyList<float[]> rows, int dim, CancellationToken ct = default, float eps = 1e-3f)
    {
        if (rows.Count < 2) return (null, null);

        var mean = new float[dim];
        foreach (var r in rows) for (int i = 0; i < dim; i++) mean[i] += r[i];
        for (int i = 0; i < dim; i++) mean[i] /= rows.Count;

        var cov = new double[dim * dim];
        foreach (var r in rows)
        {
            ct.ThrowIfCancellationRequested();
            for (int i = 0; i < dim; i++)
            {
                double di = r[i] - mean[i];
                if (di == 0) continue;
                int b = i * dim;
                for (int j = i; j < dim; j++) cov[b + j] += di * (r[j] - mean[j]);
            }
        }
        for (int i = 0; i < dim; i++)
            for (int j = i; j < dim; j++) { double v = cov[i * dim + j] / rows.Count; cov[i * dim + j] = v; cov[j * dim + i] = v; }

        var (eigVals, eigVecs) = JacobiEigen(cov, dim, ct);

        var scale = new double[dim];
        for (int i = 0; i < dim; i++) scale[i] = 1.0 / Math.Sqrt(Math.Max(eigVals[i], 0) + eps);

        var W = new float[dim * dim];
        for (int i = 0; i < dim; i++)
            for (int j = 0; j < dim; j++)
            {
                double sum = 0;
                for (int k = 0; k < dim; k++) sum += eigVecs[i * dim + k] * scale[k] * eigVecs[j * dim + k];
                W[i * dim + j] = (float)sum;
            }
        return (mean, W);
    }

    // Cyclic Jacobi eigen-decomposition of a symmetric matrix. Eigenvectors stored column-wise:
    // V[i*d + k] is component i of eigenvector k.
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
                for (int q = p + 1; q < d; q++)
                {
                    double apq = a[p * d + q];
                    if (Math.Abs(apq) < 1e-300) continue;
                    double app = a[p * d + p], aqq = a[q * d + q];
                    double phi = 0.5 * Math.Atan2(2 * apq, aqq - app);
                    double c = Math.Cos(phi), s = Math.Sin(phi);

                    for (int k = 0; k < d; k++) { double akp = a[k * d + p], akq = a[k * d + q]; a[k * d + p] = c * akp - s * akq; a[k * d + q] = s * akp + c * akq; }
                    for (int k = 0; k < d; k++) { double apk = a[p * d + k], aqk = a[q * d + k]; a[p * d + k] = c * apk - s * aqk; a[q * d + k] = s * apk + c * aqk; }
                    for (int k = 0; k < d; k++) { double vkp = v[k * d + p], vkq = v[k * d + q]; v[k * d + p] = c * vkp - s * vkq; v[k * d + q] = s * vkp + c * vkq; }
                }
        }

        var vals = new double[d];
        for (int i = 0; i < d; i++) vals[i] = a[i * d + i];
        return (vals, v);
    }
}
