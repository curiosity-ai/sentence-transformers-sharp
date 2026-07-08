using System.Numerics.Tensors;

namespace SentenceTransformers.Training;

/// <summary>
/// Analytic sentence-embedding losses that operate on already-L2-normalized vectors and accumulate the
/// gradient w.r.t. each vector. A LoRA trainer maps these vector-gradients back through normalization and
/// the autograd graph into the LoRA parameters, so these functions are the only place the objective math
/// lives. Shared by the BERT and Gemma3 trainers.
/// </summary>
public static class LoraLosses
{
    // ----- symmetric InfoNCE (MultipleNegativesRanking) with in-batch + hard negatives ------------

    /// <param name="vecs">All unique normalized vectors for the batch.</param>
    /// <param name="anchors">Per-pair anchor index into <paramref name="vecs"/>.</param>
    /// <param name="positives">Per-pair positive index into <paramref name="vecs"/>.</param>
    /// <param name="extraNeg">Per-pair extra (hard) negative indices, used only as negatives on the anchor→doc side.</param>
    /// <param name="allowNegP">allowNegP[i][j]: may positive j be a negative for anchor i? (false-negative mask).</param>
    /// <param name="allowNegQ">allowNegQ[i][j]: may anchor j be a negative for positive i?</param>
    /// <param name="gU">Gradient accumulator, same layout as <paramref name="vecs"/> (caller pre-zeroes).</param>
    public static double Contrastive(
        float[][] vecs, int[] anchors, int[] positives, List<int>[] extraNeg,
        bool[][] allowNegP, bool[][] allowNegQ, float temp, float[][] gU, out double dTemp)
    {
        int n = anchors.Length;
        double loss = 0, dt = 0;
        float invTemp = 1f / temp;
        float w = 0.5f / n; // 0.5 per direction, averaged over rows

        var cand = new List<(int idx, bool pos)>(2 * n + 8);

        // direction A: anchor -> positive (+ hard negatives)
        for (int i = 0; i < n; i++)
        {
            cand.Clear();
            cand.Add((positives[i], true));
            for (int j = 0; j < n; j++) if (j != i && allowNegP[i][j]) cand.Add((positives[j], false));
            if (extraNeg != null && extraNeg[i] != null) foreach (int e in extraNeg[i]) cand.Add((e, false));
            loss += RowSoftmaxGrad(vecs, anchors[i], cand, invTemp, w, gU, ref dt);
        }
        // direction B: positive -> anchor
        for (int i = 0; i < n; i++)
        {
            cand.Clear();
            cand.Add((anchors[i], true));
            for (int j = 0; j < n; j++) if (j != i && allowNegQ[i][j]) cand.Add((anchors[j], false));
            loss += RowSoftmaxGrad(vecs, positives[i], cand, invTemp, w, gU, ref dt);
        }

        dTemp = dt;
        return loss;
    }

    private static double RowSoftmaxGrad(float[][] vecs, int q, List<(int idx, bool pos)> cand, float invTemp, float w, float[][] gU, ref double dTemp)
    {
        int m = cand.Count;
        if (m <= 1) return 0; // only the positive; softmax is 1, loss 0
        var qv = vecs[q];
        Span<float> logit = m <= 256 ? stackalloc float[m] : new float[m];
        Span<float> sim   = m <= 256 ? stackalloc float[m] : new float[m];
        int posPos = -1;
        float max = float.NegativeInfinity;
        for (int c = 0; c < m; c++)
        {
            float s = TensorPrimitives.Dot(qv, vecs[cand[c].idx]);
            sim[c] = s;
            logit[c] = s * invTemp;
            if (cand[c].pos) posPos = c;
            if (logit[c] > max) max = logit[c];
        }
        float sum = 0;
        for (int c = 0; c < m; c++) { logit[c] = MathF.Exp(logit[c] - max); sum += logit[c]; }
        float inv = 1f / sum;
        double rowLoss = -Math.Log(Math.Max(logit[posPos] * inv, 1e-20f)) * w;

        for (int c = 0; c < m; c++)
        {
            float sm = logit[c] * inv;
            float d  = w * (sm - (c == posPos ? 1f : 0f)); // dLoss/dlogit
            float coef = d * invTemp;                       // dLoss/dsim
            var cv = vecs[cand[c].idx];
            var gq = gU[q];
            var gc = gU[cand[c].idx];
            for (int k = 0; k < qv.Length; k++)
            {
                gq[k] += coef * cv[k];
                gc[k] += coef * qv[k];
            }
            dTemp += d * (-sim[c] * invTemp * invTemp); // dlogit/dtemp = -sim/temp^2
        }
        return rowLoss;
    }

    // ----- CoSENT (pairwise ranking; directly targets Spearman) -----------------------------------

    public static double CoSent(float[][] vecs, int[] aIdx, int[] pIdx, float[] scores, float temp, float[][] gU, out double dTemp)
    {
        int p = aIdx.Length;
        var s = new float[p];
        for (int k = 0; k < p; k++) s[k] = TensorPrimitives.Dot(vecs[aIdx[k]], vecs[pIdx[k]]);

        float invTemp = 1f / temp;
        // Z = 1 + sum_{score_k > score_l} exp((s_l - s_k)/temp)
        double z = 1.0;
        var pairs = new List<(int hi, int lo, double term)>();
        for (int k = 0; k < p; k++)
            for (int l = 0; l < p; l++)
                if (scores[k] > scores[l])
                {
                    double t = Math.Exp((s[l] - s[k]) * invTemp);
                    z += t;
                    pairs.Add((k, l, t));
                }

        var ds = new double[p];
        double dt = 0;
        foreach (var (hi, lo, term) in pairs)
        {
            double tn = term / z;              // dLoss/d(exponent)
            ds[hi] += -tn * invTemp;           // d exponent/ds_hi = -1/temp
            ds[lo] += tn * invTemp;            // d exponent/ds_lo = +1/temp
            dt     += tn * (-(s[lo] - s[hi]) * invTemp * invTemp);
        }
        for (int k = 0; k < p; k++)
        {
            if (ds[k] == 0) continue;
            float g = (float)ds[k];
            var av = vecs[aIdx[k]]; var pv = vecs[pIdx[k]];
            var ga = gU[aIdx[k]]; var gp = gU[pIdx[k]];
            for (int i = 0; i < av.Length; i++) { ga[i] += g * pv[i]; gp[i] += g * av[i]; }
        }
        dTemp = dt;
        return Math.Log(z);
    }

    // ----- cosine-similarity regression (MSE) -----------------------------------------------------

    public static double Regression(float[][] vecs, int[] aIdx, int[] pIdx, float[] scores, float[][] gU)
    {
        int p = aIdx.Length;
        if (p == 0) return 0;
        double loss = 0;
        float invN = 1f / p;
        for (int k = 0; k < p; k++)
        {
            var av = vecs[aIdx[k]]; var pv = vecs[pIdx[k]];
            float s = TensorPrimitives.Dot(av, pv);
            float diff = s - scores[k];
            loss += diff * diff;
            float g = 2f * diff * invN;
            var ga = gU[aIdx[k]]; var gp = gU[pIdx[k]];
            for (int i = 0; i < av.Length; i++) { ga[i] += g * pv[i]; gp[i] += g * av[i]; }
        }
        return loss * invN;
    }
}
