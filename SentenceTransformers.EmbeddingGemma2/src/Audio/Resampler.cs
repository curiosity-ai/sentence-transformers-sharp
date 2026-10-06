namespace SentenceTransformers.EmbeddingGemma2.Audio;

/// <summary>Band-limited (windowed-sinc) sample-rate conversion to the model's 16 kHz. The reference runtime
/// resamples with miniaudio inside its decoder, so non-16 kHz inputs can differ marginally from it; feed
/// 16 kHz audio for exact parity.</summary>
internal static class Resampler
{
    private const int HalfTaps = 16;

    public static float[] Resample(ReadOnlySpan<float> input, int fromRate, int toRate)
    {
        if (fromRate == toRate)
        {
            return input.ToArray();
        }
        long outLength = (long)input.Length * toRate / fromRate;
        var output = new float[outLength];
        double ratio = (double)fromRate / toRate;
        // Cut-off at the lower Nyquist frequency.
        double cutoff = Math.Min(1.0, (double)toRate / fromRate);
        for (long i = 0; i < outLength; i++)
        {
            double center = i * ratio;
            int c = (int)Math.Floor(center);
            double sum = 0, wsum = 0;
            int half = (int)Math.Ceiling(HalfTaps / cutoff);
            for (int k = c - half + 1; k <= c + half; k++)
            {
                double t = (center - k) * cutoff;
                double sinc = t == 0 ? 1.0 : Math.Sin(Math.PI * t) / (Math.PI * t);
                double x = (center - k) / (half + 1);
                double window = Math.Abs(x) >= 1 ? 0 : 0.5 * (1 + Math.Cos(Math.PI * x));
                double w = sinc * window;
                float v = k < 0 || k >= input.Length ? 0f : input[k];
                sum += v * w;
                wsum += w;
            }
            output[i] = (float)(wsum != 0 ? sum / wsum : 0);
        }
        return output;
    }
}
