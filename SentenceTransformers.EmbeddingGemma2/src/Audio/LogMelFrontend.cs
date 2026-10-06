using SentenceTransformers.EmbeddingGemma2.LiteRt;

namespace SentenceTransformers.EmbeddingGemma2.Audio;

/// <summary>
/// Port of LiteRT-LM's audio front-end (<c>audio_preprocessor_miniaudio.cc</c>, <c>audio_preprocessor_utils.cc</c>,
/// <c>mel_filterbank.cc</c>) for 16 kHz mono PCM: semicausal framing (the first window is preceded by
/// <c>frame - hop</c> zeros), optional pre-emphasis, Hann window, zero padding to the FFT size, power
/// spectrum, HTK-style mel filterbank over magnitudes (double precision, like the reference) and
/// <c>log(mel + floor)</c>.
/// </summary>
internal sealed class LogMelFrontend
{
    /// <summary><c>FFT_PADDING_TYPE_CENTER</c>; every other value means right padding (as in the engine).</summary>
    private const int CenterPadding = 2;

    private readonly AudioPreprocessorConfig _c;
    private readonly float[] _window;
    private readonly int _fftBins;
    private readonly int _melStart, _melEnd;
    private readonly int[] _bandMapper;
    private readonly double[] _weights;
    private readonly double[] _cos, _sin;   // FFT twiddles
    private readonly int[] _bitReverse;

    public int NumMelBins => _c.NumMelBins;
    public int SampleRate => _c.SampleRateHz;

    public LogMelFrontend(AudioPreprocessorConfig config)
    {
        _c = config ?? throw new ArgumentNullException(nameof(config));
        if (_c.FftLength <= 0 || (_c.FftLength & (_c.FftLength - 1)) != 0)
        {
            throw new NotSupportedException($"FFT length {_c.FftLength} must be a power of two.");
        }
        if (_c.NormalizeMel)
        {
            throw new NotSupportedException("normalize_mel is not supported (not used by EmbeddingGemma 2).");
        }
        _window = HanningWindow(_c.FrameLength, _c.PeriodicHanning, _c.NonZeroHanning);
        _fftBins = _c.FftLength / 2 + 1;

        // MelFilterbank::Initialize(fft_bins, sample_rate, mel_bins, low, high).
        double FreqToMel(double f) => 1127.0 * Math.Log(1.0 + f / 700.0);
        double melLow = FreqToMel(_c.MelLowHz), melHigh = FreqToMel(_c.MelHighHz);
        double spacing = (melHigh - melLow) / (_c.NumMelBins + 1);
        double hzPerBin = _c.SampleRateHz / (2.0 * (_fftBins - 1));
        _melStart = (int)(1.5 + _c.MelLowHz / hzPerBin);
        _melEnd = (int)(_c.MelHighHz / hzPerBin);
        _bandMapper = new int[_fftBins];
        _weights = new double[_fftBins];
        for (int i = 0; i < _fftBins; i++)
        {
            if (i < _melStart || i > _melEnd)
            {
                _bandMapper[i] = -2;
                continue;
            }
            double pos = (FreqToMel(i * hzPerBin) - melLow) / spacing - 1;
            int channel = (int)Math.Ceiling(pos) - 1;
            _bandMapper[i] = channel;
            _weights[i] = 1.0 - (pos - channel);
        }

        int n = _c.FftLength;
        _cos = new double[n / 2];
        _sin = new double[n / 2];
        for (int i = 0; i < n / 2; i++)
        {
            _cos[i] = Math.Cos(-2 * Math.PI * i / n);
            _sin[i] = Math.Sin(-2 * Math.PI * i / n);
        }
        _bitReverse = new int[n];
        int bits = System.Numerics.BitOperations.Log2((uint)n);
        for (int i = 0; i < n; i++)
        {
            int r = 0;
            for (int b = 0; b < bits; b++)
            {
                r |= ((i >> b) & 1) << (bits - 1 - b);
            }
            _bitReverse[i] = r;
        }
    }

    /// <summary><c>GetHanningWindow</c>: 0.5 − 0.5·cos(2π(i + shift)/N) with N = L (periodic, even L) or L − 1.</summary>
    private static float[] HanningWindow(int length, bool periodic, bool nonZero)
    {
        int n = periodic && length % 2 == 0 ? length : length - 1;
        float arg = (float)(Math.PI * 2.0 / n);
        float shift = nonZero ? 0.5f : 0f;
        var w = new float[length];
        for (int i = 0; i < length; i++)
        {
            w[i] = (float)(0.5 - 0.5 * Math.Cos(arg * (i + shift)));
        }
        return w;
    }

    /// <summary>Converts a whole clip (one engine <c>Preprocess</c> call from a fresh state) into
    /// [frames, mel] log-mel features.</summary>
    public float[] Compute(ReadOnlySpan<float> pcm, out int frames)
    {
        var windows = Frame(pcm);
        frames = windows.Count;
        int mels = _c.NumMelBins;
        var result = new float[frames * mels];
        var re = new double[_c.FftLength];
        var im = new double[_c.FftLength];
        var power = new double[_fftBins];
        var mel = new double[mels];
        for (int f = 0; f < frames; f++)
        {
            var frame = windows[f];
            // Pre-emphasis then window (GetWindowedSignalsForFft).
            float pe = _c.PreEmphasisFactor;
            var x = new float[_c.FrameLength];
            x[0] = frame[0] * (1 - pe);
            for (int j = 1; j < x.Length; j++)
            {
                x[j] = frame[j] - pe * frame[j - 1];
            }
            for (int j = 0; j < x.Length; j++)
            {
                x[j] *= _window[j];
            }
            // PadOrTruncateForFft (center or right).
            Array.Clear(re);
            Array.Clear(im);
            int fl = _c.FrameLength, n = _c.FftLength;
            if (fl <= n)
            {
                int padLeft = _c.FftPaddingType == CenterPadding ? (n - fl) / 2 : 0;
                for (int j = 0; j < fl; j++)
                {
                    re[padLeft + j] = x[j];
                }
            }
            else
            {
                int trim = _c.FftPaddingType == CenterPadding ? (fl - n) / 2 : 0;
                for (int j = 0; j < n; j++)
                {
                    re[j] = x[trim + j];
                }
            }
            Fft(re, im);
            for (int k = 0; k < _fftBins; k++)
            {
                // kiss_fftr output is float; the power is formed in float.
                float r = (float)re[k], i = (float)im[k];
                power[k] = r * r + i * i;
            }
            // MelFilterbank::ToMelSpectrum (magnitudes, double accumulation).
            Array.Clear(mel);
            for (int k = _melStart; k <= _melEnd; k++)
            {
                double spec = Math.Sqrt(power[k]);
                double weighted = spec * _weights[k];
                int channel = _bandMapper[k];
                if (channel >= 0)
                {
                    mel[channel] += weighted;
                }
                channel++;
                if (channel < mels)
                {
                    mel[channel] += spec - weighted;
                }
            }
            var row = result.AsSpan(f * mels, mels);
            for (int j = 0; j < mels; j++)
            {
                row[j] = _c.AddFloorToMelBeforeLog
                    ? MathF.Log((float)mel[j] + _c.MelFloor)
                    : MathF.Max(MathF.Log((float)mel[j]), _c.MelFloor);
            }
        }
        return result;
    }

    /// <summary><c>GetFramedSegments</c> from a fresh preprocessor state (semicausal padding included).</summary>
    private List<float[]> Frame(ReadOnlySpan<float> pcmIn)
    {
        int frameLength = _c.FrameLength, hop = _c.HopLength;
        var pcm = new float[pcmIn.Length];
        for (int i = 0; i < pcm.Length; i++)
        {
            pcm[i] = pcmIn[i] * _c.InputScale;
        }
        var queue = new List<float>();
        int samplesToNext;
        if (_c.SemicausalPadding)
        {
            samplesToNext = frameLength - hop;
            queue.AddRange(new float[hop]);
        }
        else
        {
            samplesToNext = frameLength;
        }
        var windows = new List<float[]>();
        int start = 0;
        while (true)
        {
            int remaining = pcm.Length - start;
            if (samplesToNext > remaining)
            {
                for (int i = start; i < pcm.Length; i++)
                {
                    queue.Add(pcm[i]);
                }
                start = pcm.Length;
                break;
            }
            if (samplesToNext < frameLength)
            {
                queue.RemoveRange(0, queue.Count - (frameLength - samplesToNext));
                for (int i = 0; i < samplesToNext; i++)
                {
                    queue.Add(pcm[start + i]);
                }
            }
            else
            {
                queue.Clear();
                for (int i = start + samplesToNext - frameLength; i < start + samplesToNext; i++)
                {
                    queue.Add(pcm[i]);
                }
            }
            start += samplesToNext;
            samplesToNext = hop;
            windows.Add(queue.ToArray());
        }
        // Not buffering the last frame: a partial window is zero-padded and emitted.
        if (queue.Count > 0 && queue.Count < frameLength)
        {
            var last = new float[frameLength];
            queue.CopyTo(last);
            windows.Add(last);
        }
        return windows;
    }

    /// <summary>In-place iterative radix-2 complex FFT (double precision).</summary>
    private void Fft(double[] re, double[] im)
    {
        int n = re.Length;
        for (int i = 0; i < n; i++)
        {
            int j = _bitReverse[i];
            if (j > i)
            {
                (re[i], re[j]) = (re[j], re[i]);
                (im[i], im[j]) = (im[j], im[i]);
            }
        }
        for (int size = 2; size <= n; size <<= 1)
        {
            int half = size >> 1, step = n / size;
            for (int s = 0; s < n; s += size)
            {
                for (int k = 0; k < half; k++)
                {
                    double wr = _cos[k * step], wi = _sin[k * step];
                    int a = s + k, b = a + half;
                    double tr = re[b] * wr - im[b] * wi;
                    double ti = re[b] * wi + im[b] * wr;
                    re[b] = re[a] - tr;
                    im[b] = im[a] - ti;
                    re[a] += tr;
                    im[a] += ti;
                }
            }
        }
    }
}
