using System.Buffers.Binary;
using SentenceTransformers.EmbeddingGemma2;
using SentenceTransformers.EmbeddingGemma2.Audio;
using SentenceTransformers.EmbeddingGemma2.LiteRt;
using SentenceTransformers.EmbeddingGemma2.Vision;
using SentenceTransformers.Tests.Support;

namespace SentenceTransformers.Tests;

/// <summary>
/// Model-free checks of the EmbeddingGemma 2 input pipelines: the image resize-target arithmetic and
/// patchification (expected values from LiteRT-LM's <c>GetAspectRatioPreservingSize</c>), image decoding
/// and resampling, WAV decoding, the log-mel front-end framing, and the download catalogue.
/// </summary>
public class EmbeddingGemma2InputTests
{
    [Theory]
    // 140 tokens = 1260 patches (16 px, 3×3 pooling): the values the reference runtime produced.
    [InlineData(528, 528, 140, 528, 528)]
    [InlineData(640, 480, 140, 480, 624)]
    [InlineData(300, 900, 140, 960, 288)]
    [InlineData(528, 528, 70, 384, 384)]
    [InlineData(640, 480, 70, 336, 432)]
    [InlineData(300, 900, 70, 672, 192)]
    [InlineData(4000, 30, 140, 48, 6528)]
    // Extreme aspect ratios fall back to one 48-px strip, capped at the maximum side length.
    [InlineData(8000, 10, 140, 48, 6720)]
    [InlineData(10, 8000, 140, 6720, 48)]
    public void AspectRatioPreservingSize_MatchesReference(int width, int height, int tokens, int expectedHeight, int expectedWidth)
    {
        var (h, w) = ImagePreprocessor.GetAspectRatioPreservingSize(width, height, tokens * 9, 16, 3);
        Assert.Equal((expectedHeight, expectedWidth), (h, w));
        Assert.Equal(0, h % 48);
        Assert.Equal(0, w % 48);
        Assert.True(h / 16 * (w / 16) <= tokens * 9);
    }

    [Fact]
    public void Patchify_IsRowMajorWithColumnRowPositionsAndPadding()
    {
        const int H = 32, W = 48, P = 16;
        var rgb = new byte[H * W * 3];
        for (int y = 0; y < H; y++)
        {
            for (int x = 0; x < W; x++)
            {
                rgb[(y * W + x) * 3] = (byte)x;
                rgb[(y * W + x) * 3 + 1] = (byte)y;
                rgb[(y * W + x) * 3 + 2] = 255;
            }
        }
        var p = ImagePreprocessor.Patchify(rgb, H, W, maxPatches: 9, P);
        Assert.Equal(6, p.ValidPatches);
        Assert.Equal(9, p.NumPatches);
        Assert.Equal(new[] { 0, 0, 1, 0, 2, 0, 0, 1, 1, 1, 2, 1, -1, -1, -1, -1, -1, -1 }, p.Positions);
        const int Dim = P * P * 3;
        // Patch 4 = column 1, row 1; its pixel (y=2, x=3) is image pixel (18, 19).
        int o = 4 * Dim + (2 * P + 3) * 3;
        Assert.Equal(19 / 255f, p.Patches[o]);
        Assert.Equal(18 / 255f, p.Patches[o + 1]);
        Assert.Equal(1f, p.Patches[o + 2]);
        Assert.All(p.Patches.AsSpan(6 * Dim).ToArray(), v => Assert.Equal(0f, v));
    }

    [Fact]
    public void Patchify_RejectsTooManyPatches()
    {
        Assert.Throws<ArgumentException>(() => ImagePreprocessor.Patchify(new byte[48 * 48 * 3], 48, 48, 8, 16));
    }

    [Theory]
    [InlineData("synthetic_528x528.png", 528, 528)]
    [InlineData("synthetic_640x480.png", 640, 480)]
    [InlineData("synthetic_300x900.png", 300, 900)]
    public void PngFixtures_Decode(string file, int width, int height)
    {
        var image = EmbeddingGemma2Image.FromFile(EmbeddingGemma2TestAssets.Fixture(file));
        Assert.Equal(width, image.Width);
        Assert.Equal(height, image.Height);
        Assert.Equal(width * height * 3, image.Rgb.Length);
        Assert.True(image.Rgb.Distinct().Count() > 16, "decoded image should not be flat");
    }

    [Fact]
    public void Image_RejectsGarbage()
    {
        Assert.Throws<InvalidDataException>(() => EmbeddingGemma2Image.FromBytes(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8 }));
    }

    [Fact]
    public void Image_FromRgbaAndBgraDropAlpha()
    {
        var rgba = new byte[] { 1, 2, 3, 255, 4, 5, 6, 0 };
        Assert.Equal(new byte[] { 1, 2, 3, 4, 5, 6 }, EmbeddingGemma2Image.FromRgba(rgba, 2, 1).Rgb);
        Assert.Equal(new byte[] { 3, 2, 1, 6, 5, 4 }, EmbeddingGemma2Image.FromBgra(rgba, 2, 1).Rgb);
        Assert.Throws<ArgumentException>(() => EmbeddingGemma2Image.FromRgb(new byte[5], 1, 2));
    }

    [Theory]
    [InlineData(100, 80, 624, 480)]
    [InlineData(1920, 1080, 960, 528)]
    [InlineData(17, 23, 48, 96)]
    public void Resize_PreservesFlatColors(int width, int height, int newWidth, int newHeight)
    {
        var rgb = new byte[width * height * 3];
        for (int i = 0; i < width * height; i++)
        {
            rgb[3 * i] = 12;
            rgb[3 * i + 1] = 128;
            rgb[3 * i + 2] = 250;
        }
        var resized = StbImageResize.ResizeRgb(rgb, width, height, newWidth, newHeight);
        Assert.Equal(newWidth * newHeight * 3, resized.Length);
        for (int i = 0; i < newWidth * newHeight; i++)
        {
            Assert.Equal(12, resized[3 * i]);
            Assert.Equal(128, resized[3 * i + 1]);
            Assert.Equal(250, resized[3 * i + 2]);
        }
    }

    [Fact]
    public void Preprocess_ProducesSignatureSizedInput()
    {
        var image = EmbeddingGemma2Image.FromFile(EmbeddingGemma2TestAssets.Fixture("synthetic_640x480.png"));
        var p = ImagePreprocessor.Preprocess(image, 140, 16, 3);
        Assert.Equal((480, 624), (p.Height, p.Width));
        Assert.Equal(1260, p.NumPatches);
        Assert.Equal(30 * 39, p.ValidPatches);
        Assert.Equal(1260 * 768, p.Patches.Length);
    }

    private static byte[] Wav(short[] interleaved, int channels, int sampleRate)
    {
        var data = new byte[44 + interleaved.Length * 2];
        "RIFF"u8.CopyTo(data);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(4), data.Length - 8);
        "WAVEfmt "u8.CopyTo(data.AsSpan(8));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(16), 16);
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(20), 1);
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(22), (short)channels);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(24), sampleRate);
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(28), sampleRate * channels * 2);
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(32), (short)(channels * 2));
        BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(34), 16);
        "data"u8.CopyTo(data.AsSpan(36));
        BinaryPrimitives.WriteInt32LittleEndian(data.AsSpan(40), interleaved.Length * 2);
        for (int i = 0; i < interleaved.Length; i++)
        {
            BinaryPrimitives.WriteInt16LittleEndian(data.AsSpan(44 + 2 * i), interleaved[i]);
        }
        return data;
    }

    [Fact]
    public void Wav_Pcm16StereoIsAveragedToMono()
    {
        var wav = Wav(new short[] { 16384, 0, -32768, -32768, 100, 300 }, 2, 16000);
        var mono = WavReader.ReadMono(wav, out int rate);
        Assert.Equal(16000, rate);
        Assert.Equal(new[] { 0.25f, -1f, 200 / 32768f }, mono);

        var audio = EmbeddingGemma2Audio.FromWav(wav);
        Assert.Equal(3, audio.Samples.Length);
        Assert.Throws<InvalidDataException>(() => EmbeddingGemma2Audio.FromWav(new byte[16]));
    }

    [Fact]
    public void Audio_ResamplesToModelRate()
    {
        var samples = new float[48000];
        for (int i = 0; i < samples.Length; i++)
        {
            samples[i] = MathF.Sin(2 * MathF.PI * 440 * i / 48000f);
        }
        var audio = EmbeddingGemma2Audio.FromSamples(samples, 48000);
        Assert.InRange(audio.Samples.Length, 15990, 16010);
        Assert.Equal(1.0, audio.Duration.TotalSeconds, 2);
    }

    [Fact]
    public void WavFixtures_Decode()
    {
        var a = EmbeddingGemma2Audio.FromFile(EmbeddingGemma2TestAssets.Fixture("tone_1.0s.wav"));
        Assert.Equal(16000, a.Samples.Length);
        var b = EmbeddingGemma2Audio.FromFile(EmbeddingGemma2TestAssets.Fixture("tone_3.3s.wav"));
        Assert.Equal(52800, b.Samples.Length);
    }

    /// <summary>The front-end configuration stored in the 740M bundle's embedding metadata.</summary>
    internal static AudioPreprocessorConfig GemmaAudioConfig() => new()
    {
        SampleRateHz = 16000,
        NumChannels = 1,
        FrameLength = 320,
        HopLength = 160,
        FftLength = 512,
        NumMelBins = 128,
        MelLowHz = 0,
        MelHighHz = 8000,
        MelFloor = 0.001f,
        AddFloorToMelBeforeLog = true,
        SemicausalPadding = true,
        PeriodicHanning = true,
        FftPaddingType = 2,
    };

    [Theory]
    [InlineData(16000, 100)]
    [InlineData(52800, 330)]
    [InlineData(16080, 100)]   // samples short of a full hop stay buffered (streaming state), as in the engine
    [InlineData(100, 1)]       // a clip shorter than one window yields one zero-padded frame
    public void LogMel_FrameCount(int samples, int expectedFrames)
    {
        var mel = new LogMelFrontend(GemmaAudioConfig()).Compute(new float[samples], out int frames);
        Assert.Equal(expectedFrames, frames);
        Assert.Equal(expectedFrames * 128, mel.Length);
        Assert.All(mel, v => Assert.Equal(MathF.Log(0.001f), v));
    }

    [Fact]
    public void LogMel_ToneEnergyLandsInTheRightBand()
    {
        var frontend = new LogMelFrontend(GemmaAudioConfig());
        var pcm = new float[16000];
        for (int i = 0; i < pcm.Length; i++)
        {
            pcm[i] = 0.5f * MathF.Sin(2 * MathF.PI * 1000 * i / 16000f);
        }
        var mel = frontend.Compute(pcm, out int frames);
        var row = mel.AsSpan(50 * 128, 128);
        int peak = 0;
        for (int i = 1; i < 128; i++)
        {
            if (row[i] > row[peak])
            {
                peak = i;
            }
        }
        // HTK mel scale: 1000 Hz sits at mel 1000 of 2840 (0-8 kHz), i.e. channel ≈ 1000/2840·129 − 1 ≈ 44.
        Assert.InRange(peak, 42, 46);
    }

    [Theory]
    [InlineData(EmbeddingGemma2Model.Text270M, "https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm/resolve/main/embeddinggemma-2-text-270m.litertlm", false, false)]
    [InlineData(EmbeddingGemma2Model.TextVision440M, "https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm/resolve/main/embeddinggemma-2-text-vision-440m.litertlm", true, false)]
    [InlineData(EmbeddingGemma2Model.Multimodal740M, "https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm/resolve/main/embeddinggemma-2-740m.litertlm", true, true)]
    public void ModelCatalogue(EmbeddingGemma2Model model, string url, bool images, bool audio)
    {
        Assert.Equal(url, EmbeddingGemma2Models.GetDownloadUrl(model));
        Assert.Equal(images, EmbeddingGemma2Models.SupportsImages(model));
        Assert.Equal(audio, EmbeddingGemma2Models.SupportsAudio(model));
        Assert.True(EmbeddingGemma2Models.GetFileSize(model) > 100_000_000);
        Assert.EndsWith(EmbeddingGemma2Models.GetFileName(model), EmbeddingGemma2Models.GetDefaultCachePath(model));
    }
}
