namespace SentenceTransformers.Training;

/// <summary>Small numeric helpers shared by the trainer and the evaluation utilities.</summary>
internal static class EmbeddingMath
{
    /// <summary>Dot product of two equal-length vectors (== cosine similarity when both are L2-normalized).</summary>
    public static float Dot(float[] a, float[] b)
    {
        float acc = 0f;
        for (int i = 0; i < a.Length; i++) acc += a[i] * b[i];
        return acc;
    }

    /// <summary>L2-normalizes a copy of <paramref name="v"/>.</summary>
    public static float[] Normalized(float[] v, float eps = 1e-12f)
    {
        var u    = (float[])v.Clone();
        float ss = 0f;
        for (int k = 0; k < u.Length; k++) ss += u[k] * u[k];
        float inv = 1f / MathF.Max(MathF.Sqrt(ss), eps);
        for (int k = 0; k < u.Length; k++) u[k] *= inv;
        return u;
    }

    /// <summary>Spearman rank correlation between two series.</summary>
    public static float Spearman(float[] x, float[] y) => Pearson(Rank(x), Rank(y));

    /// <summary>Fractional (average) ranks, so tied values share the mean of their rank positions.</summary>
    public static float[] Rank(float[] values)
    {
        int n   = values.Length;
        var idx = Enumerable.Range(0, n).ToArray();
        Array.Sort(idx, (i, j) => values[i].CompareTo(values[j]));

        var ranks = new float[n];
        int p     = 0;
        while (p < n)
        {
            int q = p;
            while (q + 1 < n && values[idx[q + 1]] == values[idx[p]]) q++;
            float avgRank = (p + q) / 2f + 1f; // 1-based average rank of the tied group
            for (int k = p; k <= q; k++) ranks[idx[k]] = avgRank;
            p = q + 1;
        }
        return ranks;
    }

    /// <summary>Pearson correlation coefficient.</summary>
    public static float Pearson(float[] a, float[] b)
    {
        int n = a.Length;
        if (n == 0) return float.NaN;

        double ma = 0, mb = 0;
        for (int i = 0; i < n; i++) { ma += a[i]; mb += b[i]; }
        ma /= n; mb /= n;

        double num = 0, da = 0, db = 0;
        for (int i = 0; i < n; i++)
        {
            double xa = a[i] - ma;
            double xb = b[i] - mb;
            num += xa * xb;
            da  += xa * xa;
            db  += xb * xb;
        }

        double denom = Math.Sqrt(da * db);
        return denom < 1e-20 ? 0f : (float)(num / denom);
    }
}
