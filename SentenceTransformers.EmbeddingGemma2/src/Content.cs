using SentenceTransformers.EmbeddingGemma2.Vision;

namespace SentenceTransformers.EmbeddingGemma2;

/// <summary>
/// One item of a (possibly interleaved) multimodal input: text, an image or an audio clip. A sequence of
/// items is embedded into a single vector with
/// <see cref="SentenceEncoder.EncodeContentAsync(IReadOnlyList{EmbeddingGemma2Content}, CancellationToken)"/>,
/// exactly like the LiteRT-LM runtime's <c>compute_embedding([...])</c>: text is tokenized, images and audio
/// become soft tokens wrapped in their start/end markers, and the whole sequence is wrapped in BOS/EOS.
/// Strings convert implicitly to text items.
/// </summary>
public abstract class EmbeddingGemma2Content
{
    private protected EmbeddingGemma2Content() { }

    public static EmbeddingGemma2Content Text(string text) => new EmbeddingGemma2Text(text);

    public static implicit operator EmbeddingGemma2Content(string text) => new EmbeddingGemma2Text(text);
}

/// <summary>A text item.</summary>
public sealed class EmbeddingGemma2Text : EmbeddingGemma2Content
{
    public string Value { get; }

    public EmbeddingGemma2Text(string value)
    {
        Value = value ?? string.Empty;
    }

    public override string ToString() => Value;
}

/// <summary>
/// An image as packed 8-bit RGB pixels (row-major, top row first). Build one from raw pixels
/// (<see cref="FromRgb"/>, <see cref="FromRgba"/>, <see cref="FromBgra"/>) or from an encoded PNG / JPEG /
/// BMP file (<see cref="FromFile"/>, <see cref="FromBytes"/>), which is decoded by a pure C# port of the
/// stb_image decoders the reference runtime uses, so pixels match it exactly.
/// </summary>
public sealed class EmbeddingGemma2Image : EmbeddingGemma2Content
{
    public int Width { get; }
    public int Height { get; }
    /// <summary>Packed RGB bytes, <c>Width · Height · 3</c>.</summary>
    public byte[] Rgb { get; }

    private EmbeddingGemma2Image(byte[] rgb, int width, int height)
    {
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentException($"Image dimensions must be positive, got {width}x{height}.");
        }
        if (rgb is null || rgb.Length != checked(width * height * 3))
        {
            throw new ArgumentException($"Expected {width * height * 3} RGB bytes for a {width}x{height} image.", nameof(rgb));
        }
        Rgb = rgb;
        Width = width;
        Height = height;
    }

    /// <summary>Wraps packed RGB pixels (not copied).</summary>
    public static EmbeddingGemma2Image FromRgb(byte[] rgb, int width, int height) => new(rgb, width, height);

    /// <summary>Converts packed RGBA pixels (alpha is dropped, as the reference decoder does).</summary>
    public static EmbeddingGemma2Image FromRgba(ReadOnlySpan<byte> rgba, int width, int height) => FromFourChannel(rgba, width, height, 0, 1, 2);

    /// <summary>Converts packed BGRA pixels (e.g. System.Drawing / SkiaSharp native layout; alpha dropped).</summary>
    public static EmbeddingGemma2Image FromBgra(ReadOnlySpan<byte> bgra, int width, int height) => FromFourChannel(bgra, width, height, 2, 1, 0);

    private static EmbeddingGemma2Image FromFourChannel(ReadOnlySpan<byte> src, int width, int height, int r, int g, int b)
    {
        int n = checked(width * height);
        if (src.Length != n * 4)
        {
            throw new ArgumentException($"Expected {n * 4} bytes for a {width}x{height} 4-channel image.");
        }
        var rgb = new byte[n * 3];
        for (int i = 0; i < n; i++)
        {
            rgb[3 * i] = src[4 * i + r];
            rgb[3 * i + 1] = src[4 * i + g];
            rgb[3 * i + 2] = src[4 * i + b];
        }
        return new EmbeddingGemma2Image(rgb, width, height);
    }

    /// <summary>Decodes an encoded image (PNG, JPEG or BMP).</summary>
    public static EmbeddingGemma2Image FromBytes(ReadOnlySpan<byte> encoded)
    {
        if (!ImageDecoder.TryDecode(encoded, out var rgb, out int w, out int h))
        {
            throw new InvalidDataException("Unsupported or corrupt image (supported: PNG, JPEG, BMP). Decode it yourself and use FromRgb/FromRgba.");
        }
        return new EmbeddingGemma2Image(rgb, w, h);
    }

    /// <summary>Loads and decodes an image file (PNG, JPEG or BMP).</summary>
    public static EmbeddingGemma2Image FromFile(string path) => FromBytes(File.ReadAllBytes(path));

    public override string ToString() => $"Image {Width}x{Height}";
}

/// <summary>
/// An audio clip as 16 kHz mono float samples in [-1, 1] - the format the EmbeddingGemma 2 audio
/// front-end consumes. Build one from samples (<see cref="FromSamples"/>, resampled when needed), 16-bit
/// PCM (<see cref="FromPcm16"/>) or a WAV file (<see cref="FromWav"/> / <see cref="FromFile"/>: PCM 8/16/24/32-bit
/// or IEEE float, any channel count; channels are averaged to mono and sample values are converted exactly
/// like the reference runtime's miniaudio decoder).
/// </summary>
public sealed class EmbeddingGemma2Audio : EmbeddingGemma2Content
{
    /// <summary>The model's sample rate.</summary>
    public const int SampleRate = 16000;

    /// <summary>Mono samples at <see cref="SampleRate"/>.</summary>
    public float[] Samples { get; }

    public TimeSpan Duration => TimeSpan.FromSeconds(Samples.Length / (double)SampleRate);

    private EmbeddingGemma2Audio(float[] samples)
    {
        Samples = samples;
    }

    /// <summary>Mono samples at <paramref name="sampleRate"/> (resampled to 16 kHz when different).</summary>
    public static EmbeddingGemma2Audio FromSamples(ReadOnlySpan<float> mono, int sampleRate = SampleRate)
    {
        if (sampleRate <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        }
        return new EmbeddingGemma2Audio(sampleRate == SampleRate ? mono.ToArray() : Audio.Resampler.Resample(mono, sampleRate, SampleRate));
    }

    /// <summary>Interleaved signed 16-bit PCM (any channel count / sample rate).</summary>
    public static EmbeddingGemma2Audio FromPcm16(ReadOnlySpan<short> interleaved, int channels = 1, int sampleRate = SampleRate)
    {
        if (channels <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(channels));
        }
        var pcm = interleaved.ToArray();
        var mono = Audio.WavReader.MixToMono(pcm.Length / channels, channels, (frame, ch) => pcm[frame * channels + ch] / 32768f);
        return FromSamples(mono, sampleRate);
    }

    /// <summary>Decodes a RIFF/WAVE file.</summary>
    public static EmbeddingGemma2Audio FromWav(ReadOnlySpan<byte> wav)
    {
        var mono = Audio.WavReader.ReadMono(wav, out int sampleRate);
        return FromSamples(mono, sampleRate);
    }

    /// <summary>Loads a WAV file.</summary>
    public static EmbeddingGemma2Audio FromFile(string path) => FromWav(File.ReadAllBytes(path));

    public override string ToString() => $"Audio {Duration.TotalSeconds:F2}s";
}
