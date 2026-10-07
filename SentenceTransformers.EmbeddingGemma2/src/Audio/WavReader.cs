using System.Buffers.Binary;

namespace SentenceTransformers.EmbeddingGemma2.Audio;

/// <summary>Minimal RIFF/WAVE reader (PCM 8/16/24/32-bit, IEEE float 32/64, WAVE_FORMAT_EXTENSIBLE).
/// Sample conversion matches miniaudio's <c>ma_pcm_*_to_f32</c> (e.g. s16 / 32768).</summary>
internal static class WavReader
{
    public static float[] ReadMono(ReadOnlySpan<byte> data, out int sampleRate)
    {
        if (data.Length < 12 || !data.Slice(0, 4).SequenceEqual("RIFF"u8) || !data.Slice(8, 4).SequenceEqual("WAVE"u8))
        {
            throw new InvalidDataException("Not a RIFF/WAVE file.");
        }
        int pos = 12;
        int format = 0, channels = 0, bits = 0;
        sampleRate = 0;
        ReadOnlySpan<byte> samples = default;
        bool haveFmt = false, haveData = false;
        while (pos + 8 <= data.Length)
        {
            var id = data.Slice(pos, 4);
            int size = (int)Math.Min(BinaryPrimitives.ReadUInt32LittleEndian(data.Slice(pos + 4)), (uint)(data.Length - pos - 8));
            var body = data.Slice(pos + 8, size);
            if (id.SequenceEqual("fmt "u8))
            {
                format = BinaryPrimitives.ReadUInt16LittleEndian(body);
                channels = BinaryPrimitives.ReadUInt16LittleEndian(body.Slice(2));
                sampleRate = (int)BinaryPrimitives.ReadUInt32LittleEndian(body.Slice(4));
                bits = BinaryPrimitives.ReadUInt16LittleEndian(body.Slice(14));
                if (format == 0xFFFE && body.Length >= 26)
                {
                    format = BinaryPrimitives.ReadUInt16LittleEndian(body.Slice(24));   // sub-format GUID's first two bytes
                }
                haveFmt = true;
            }
            else if (id.SequenceEqual("data"u8))
            {
                samples = body;
                haveData = true;
            }
            pos += 8 + size + (size & 1);
        }
        if (!haveFmt || !haveData || channels <= 0 || sampleRate <= 0)
        {
            throw new InvalidDataException("WAV file is missing its fmt or data chunk.");
        }
        int bytesPerSample = bits / 8;
        int frames = samples.Length / (bytesPerSample * channels);
        var s = samples.ToArray();
        Func<int, int, float> read = (format, bits) switch
        {
            (1, 8) => (f, c) => (s[f * channels + c] - 128) / 128f,
            (1, 16) => (f, c) => BinaryPrimitives.ReadInt16LittleEndian(s.AsSpan((f * channels + c) * 2)) / 32768f,
            (1, 24) => (f, c) =>
            {
                int o = (f * channels + c) * 3;
                int v = (s[o] | (s[o + 1] << 8) | (s[o + 2] << 16)) << 8 >> 8;
                return v / 8388608f;
            },
            (1, 32) => (f, c) => (float)(BinaryPrimitives.ReadInt32LittleEndian(s.AsSpan((f * channels + c) * 4)) / 2147483648.0),
            (3, 32) => (f, c) => BinaryPrimitives.ReadSingleLittleEndian(s.AsSpan((f * channels + c) * 4)),
            (3, 64) => (f, c) => (float)BinaryPrimitives.ReadDoubleLittleEndian(s.AsSpan((f * channels + c) * 8)),
            _ => throw new NotSupportedException($"Unsupported WAV encoding (format {format}, {bits} bits)."),
        };
        return MixToMono(frames, channels, read);
    }

    /// <summary>Averages channels into mono.</summary>
    public static float[] MixToMono(int frames, int channels, Func<int, int, float> sample)
    {
        var mono = new float[frames];
        for (int f = 0; f < frames; f++)
        {
            if (channels == 1)
            {
                mono[f] = sample(f, 0);
                continue;
            }
            float sum = 0;
            for (int c = 0; c < channels; c++)
            {
                sum += sample(f, c);
            }
            mono[f] = sum / channels;
        }
        return mono;
    }
}
