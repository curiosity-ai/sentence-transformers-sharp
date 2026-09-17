#nullable enable

namespace SentenceTransformers.Ternary;

/// <summary>How a group of weights is reduced to trits and one shared scale.</summary>
public enum TernaryMethod
{
    /// <summary>
    /// The exact L2-optimal ternarization of the group (default). See
    /// <see cref="TernaryQuantizer.QuantizeGroup"/> for why a sort plus a linear scan finds the true
    /// optimum rather than approximating it.
    /// </summary>
    Optimal = 0,

    /// <summary>BitNet b1.58's rule: <c>s = mean(|w|)</c>, then round-to-nearest and clamp to
    /// <c>[-1, 1]</c>. Cheapest, and the baseline most ternary papers quote.</summary>
    AbsMean = 1,

    /// <summary>Ternary Weight Networks (Li &amp; Liu): keep weights above <c>0.7 * mean(|w|)</c>,
    /// scale by the mean of the kept magnitudes. A closed-form approximation of
    /// <see cref="Optimal"/> derived under a Gaussian assumption.</summary>
    Twn = 2,
}

/// <summary>
/// Reduces a group of float weights to <c>{-1, 0, +1}</c> codes plus one shared scale, minimizing
/// <c>|| w - s * t ||^2</c>.
/// </summary>
public static class TernaryQuantizer
{
    /// <summary>
    /// Quantizes one group in place: fills <paramref name="codes"/> with values in
    /// <c>{-1, 0, +1}</c> and returns the shared scale <c>s</c> such that <c>w_i ~= s * codes[i]</c>.
    ///
    /// <para><b>The optimal method.</b> Fix the number of non-zero trits <c>k</c>. For any support set
    /// <c>S</c> of size <c>k</c> the signs must be <c>sign(w_i)</c>, and the error is
    /// <code>
    ///   sum_i w_i^2  -  2 s * sum_{i in S} |w_i|  +  k s^2
    /// </code>
    /// which is minimized at <c>s = (sum_{i in S} |w_i|) / k</c>, leaving
    /// <c>sum w^2 - (sum_{S} |w|)^2 / k</c>. For a fixed <c>k</c> that is smallest when <c>S</c> holds
    /// the <c>k</c> largest magnitudes, so sorting <c>|w|</c> descending and scanning
    /// <c>k = 1..n</c> for the largest <c>prefix(k)^2 / k</c> visits every candidate optimum. The
    /// result is exactly optimal for the group, not a heuristic, and costs one sort.</para>
    /// </summary>
    /// <param name="w">The group's weights.</param>
    /// <param name="codes">Receives one trit per weight; must be the same length as <paramref name="w"/>.</param>
    /// <param name="method">Quantization rule.</param>
    /// <param name="scratch">Optional scratch buffer of at least <c>w.Length</c> floats, used by
    /// <see cref="TernaryMethod.Optimal"/> to avoid allocating per group.</param>
    /// <returns>The group scale. Zero only when every weight in the group is zero.</returns>
    public static float QuantizeGroup(ReadOnlySpan<float> w, Span<sbyte> codes, TernaryMethod method = TernaryMethod.Optimal, Span<float> scratch = default)
    {
        if (codes.Length != w.Length)
        {
            throw new ArgumentException($"codes ({codes.Length}) must match w ({w.Length}).", nameof(codes));
        }

        return method switch
        {
            TernaryMethod.Optimal => QuantizeOptimal(w, codes, scratch),
            TernaryMethod.AbsMean => QuantizeAbsMean(w, codes),
            TernaryMethod.Twn     => QuantizeTwn(w, codes),
            _ => throw new ArgumentOutOfRangeException(nameof(method), method, null),
        };
    }

    private static float QuantizeOptimal(ReadOnlySpan<float> w, Span<sbyte> codes, Span<float> scratch)
    {
        int n = w.Length;
        Span<float> mag = scratch.Length >= n ? scratch[..n] : new float[n];
        for (int i = 0; i < n; i++)
        {
            mag[i] = MathF.Abs(w[i]);
        }

        // Sort a copy of the magnitudes descending; we only need the threshold, not the permutation.
        mag.Sort();                     // ascending
        float best = 0f;
        int bestK = 0;
        float prefix = 0f;
        for (int k = 1; k <= n; k++)
        {
            prefix += mag[n - k];       // walk the sorted magnitudes from the largest down
            float gain = prefix * prefix / k;
            if (gain > best)
            {
                best = gain;
                bestK = k;
            }
        }

        if (bestK == 0)
        {
            codes.Clear();
            return 0f;
        }

        // Keep exactly bestK weights: everything strictly above the k-th largest magnitude, plus
        // however many ties at that magnitude are needed to reach bestK. Getting the count wrong
        // here would divide the scale by the wrong k. bestK never selects a zero-magnitude weight
        // (that would leave the prefix unchanged while raising k, lowering the gain), so the
        // threshold is always positive and no zero weight is given a sign.
        float threshold = mag[n - bestK];
        int strictlyAbove = 0;
        for (int i = 0; i < n; i++)
        {
            if (MathF.Abs(w[i]) > threshold) strictlyAbove++;
        }
        int tiesToKeep = bestK - strictlyAbove;

        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            float a = MathF.Abs(w[i]);
            bool keep;
            if (a > threshold)
            {
                keep = true;
            }
            else if (a == threshold && tiesToKeep > 0)
            {
                tiesToKeep--;
                keep = true;
            }
            else
            {
                keep = false;
            }

            if (keep)
            {
                codes[i] = w[i] < 0 ? (sbyte)-1 : (sbyte)1;
                sum += a;
            }
            else
            {
                codes[i] = 0;
            }
        }
        return sum / bestK;
    }

    private static float QuantizeAbsMean(ReadOnlySpan<float> w, Span<sbyte> codes)
    {
        int n = w.Length;
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            sum += MathF.Abs(w[i]);
        }
        float s = sum / n;
        if (s <= 0f)
        {
            codes.Clear();
            return 0f;
        }
        float inv = 1f / s;
        for (int i = 0; i < n; i++)
        {
            int q = (int)MathF.Round(w[i] * inv);
            codes[i] = (sbyte)Math.Clamp(q, -1, 1);
        }
        return s;
    }

    private static float QuantizeTwn(ReadOnlySpan<float> w, Span<sbyte> codes)
    {
        int n = w.Length;
        float sum = 0f;
        for (int i = 0; i < n; i++)
        {
            sum += MathF.Abs(w[i]);
        }
        float threshold = 0.7f * sum / n;
        float kept = 0f;
        int support = 0;
        for (int i = 0; i < n; i++)
        {
            float a = MathF.Abs(w[i]);
            if (a > threshold)
            {
                codes[i] = w[i] < 0 ? (sbyte)-1 : (sbyte)1;
                kept += a;
                support++;
            }
            else
            {
                codes[i] = 0;
            }
        }
        return support > 0 ? kept / support : 0f;
    }

    /// <summary>Relative L2 error <c>||w - s*t|| / ||w||</c> for an already-quantized group - the
    /// number the converter's per-tensor report aggregates.</summary>
    public static double RelativeError(ReadOnlySpan<float> w, ReadOnlySpan<sbyte> codes, float scale)
    {
        double err = 0, mag = 0;
        for (int i = 0; i < w.Length; i++)
        {
            double d = w[i] - scale * codes[i];
            err += d * d;
            mag += (double)w[i] * w[i];
        }
        return mag > 0 ? Math.Sqrt(err / mag) : 0.0;
    }
}
