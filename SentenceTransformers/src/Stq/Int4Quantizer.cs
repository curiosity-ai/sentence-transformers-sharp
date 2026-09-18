#nullable enable

namespace SentenceTransformers.Stq;

/// <summary>
/// Reduces a group of float weights to symmetric 4-bit codes in <c>[-8, +7]</c> plus one shared
/// scale, minimizing <c>|| w - s * q ||^2</c>.
///
/// <para>Unlike the ternary case there is no cheap exact optimum: the codes depend on the scale
/// through a rounding, so the error is piecewise-quadratic in <c>s</c> with a kink at every point
/// where some weight changes code. Round-to-nearest at <c>s = max|w| / 7</c> is the usual choice and
/// is what this repository's load-time <c>Int4Matrix</c> does. Searching a small grid of scales
/// around it and keeping the best costs a handful of passes over 32 weights at conversion time - it
/// is free at inference - and recovers most of the gap to the true optimum, mainly by trading a
/// little clipping of the single largest weight for finer resolution on all the others.</para>
/// </summary>
public static class Int4Quantizer
{
    /// <summary>Lowest code. The range is asymmetric by one because four bits give sixteen values;
    /// leaving -8 unused to keep it symmetric would throw away 1/16 of the code space.</summary>
    public const int QMin = -8;

    /// <summary>Highest code.</summary>
    public const int QMax = 7;

    /// <summary>Scale-grid resolution used when searching. Each step narrows the scale, trading
    /// clipping against resolution.</summary>
    private const int SearchSteps = 24;

    /// <summary>Narrowest scale tried, as a fraction of the round-to-nearest scale.</summary>
    private const float SearchFloor = 0.60f;

    /// <summary>
    /// Quantizes one group in place and returns the shared scale, such that
    /// <c>w_i ~= scale * codes[i]</c>.
    /// </summary>
    /// <param name="search">When true (the default) the scale grid is searched for the lowest
    /// squared error; when false the plain round-to-nearest scale is used, which reproduces the
    /// load-time <c>Int4Matrix</c> exactly.</param>
    public static float QuantizeGroup(ReadOnlySpan<float> w, Span<sbyte> codes, bool search = true)
    {
        if (codes.Length != w.Length)
        {
            throw new ArgumentException($"codes ({codes.Length}) must match w ({w.Length}).", nameof(codes));
        }

        float amax = 0f;
        for (int i = 0; i < w.Length; i++)
        {
            amax = MathF.Max(amax, MathF.Abs(w[i]));
        }
        if (amax <= 0f)
        {
            codes.Clear();
            return 0f;
        }

        float baseScale = amax / QMax;
        if (!search)
        {
            Apply(w, codes, baseScale);
            return baseScale;
        }

        float bestScale = baseScale;
        double bestError = Error(w, baseScale);
        for (int step = 1; step <= SearchSteps; step++)
        {
            float candidate = baseScale * (1f - (1f - SearchFloor) * step / SearchSteps);
            double error = Error(w, candidate);
            if (error < bestError)
            {
                bestError = error;
                bestScale = candidate;
            }
        }

        Apply(w, codes, bestScale);
        return bestScale;
    }

    /// <summary>Squared error of quantizing at <paramref name="scale"/>, without writing codes.</summary>
    private static double Error(ReadOnlySpan<float> w, float scale)
    {
        float inv = 1f / scale;
        double error = 0;
        for (int i = 0; i < w.Length; i++)
        {
            int q = Math.Clamp((int)MathF.Round(w[i] * inv), QMin, QMax);
            double d = w[i] - scale * q;
            error += d * d;
        }
        return error;
    }

    private static void Apply(ReadOnlySpan<float> w, Span<sbyte> codes, float scale)
    {
        float inv = 1f / scale;
        for (int i = 0; i < w.Length; i++)
        {
            codes[i] = (sbyte)Math.Clamp((int)MathF.Round(w[i] * inv), QMin, QMax);
        }
    }
}

/// <summary>Dispatches a group to the right quantizer for its band, so the tensor builder does not
/// have to know which bands are ternary.</summary>
public static class StqQuantizer
{
    /// <summary>Quantizes one group into <paramref name="codes"/> and returns the shared scale.</summary>
    /// <param name="scratch">Optional scratch of at least <c>w.Length</c> floats, used by the
    /// ternary optimal method to avoid allocating per group.</param>
    public static float QuantizeGroup(StqBand band, ReadOnlySpan<float> w, Span<sbyte> codes,
                                      TernaryMethod method = TernaryMethod.Optimal, Span<float> scratch = default)
        => band switch
        {
            StqBand.TQ1_0 or StqBand.TQ2_0 => TernaryQuantizer.QuantizeGroup(w, codes, method, scratch),
            StqBand.Q4_0 => Int4Quantizer.QuantizeGroup(w, codes),
            _ => throw new ArgumentOutOfRangeException(nameof(band), band, "Not a packed band."),
        };
}
