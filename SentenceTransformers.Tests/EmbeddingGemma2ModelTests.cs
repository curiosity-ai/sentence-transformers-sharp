using System.Text.Json;
using SentenceTransformers.EmbeddingGemma2;
using SentenceTransformers.EmbeddingGemma2.Tokenizer;
using SentenceTransformers.Tests.Support;

namespace SentenceTransformers.Tests;

/// <summary>
/// End-to-end parity of the managed EmbeddingGemma 2 port with Google's LiteRT-LM runtime
/// (<c>litert_lm.EmbeddingEngine</c>, CPU / XNNPACK). The expected tokens and embeddings in
/// <c>Resources/embeddinggemma2</c> were produced by <c>scripts/generate_embeddinggemma2_reference.py</c>.
/// <para>
/// Opt-in: each test returns early unless the bundle it needs is available (see
/// <see cref="EmbeddingGemma2TestAssets"/>). The engine's XNNPACK kernels quantize activations with their
/// own rounding / GELU approximation, so embeddings agree to cosine ≈ 0.997–0.9997 rather than bit-exactly;
/// the per-layer agreement is checked by <see cref="EmbeddingGemma2ReferenceTests"/>.
/// </para>
/// </summary>
public class EmbeddingGemma2ModelTests
{
    private static JsonElement Fixture(string name) => JsonDocument.Parse(File.ReadAllText(EmbeddingGemma2TestAssets.Fixture(name))).RootElement;

    /// <summary>All bundles share the text tower, so text tests use whichever is available (smallest first).</summary>
    private static SentenceEncoder AnyEncoder()
        => EmbeddingGemma2TestAssets.Encoder(EmbeddingGemma2Model.Text270M)
           ?? EmbeddingGemma2TestAssets.Encoder(EmbeddingGemma2Model.TextVision440M)
           ?? EmbeddingGemma2TestAssets.Encoder(EmbeddingGemma2Model.Multimodal740M);

    [Fact]
    public void Tokenizer_MatchesSentencePiece()
    {
        var encoder = AnyEncoder();
        if (encoder is null)
        {
            return;
        }
        var tokenizer = Assert.IsType<EmbeddingGemma2Tokenizer>(encoder.Tokenizer);
        int n = 0;
        foreach (var c in Fixture("tokenizer_cases.json").EnumerateArray())
        {
            var text = c.GetProperty("text").GetString();
            var expected = c.GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()).ToArray();
            var ids = tokenizer.EncodeIdsWithoutSpecialTokens(text);
            Assert.True(expected.SequenceEqual(ids), $"{JsonSerializer.Serialize(text)}: [{string.Join(",", ids)}] != [{string.Join(",", expected)}]");
            Assert.Equal(text, tokenizer.Decode(ids));
            n++;
        }
        Assert.True(n >= 50);
    }

    [Fact]
    public async Task Text_MatchesEngine()
    {
        var encoder = AnyEncoder();
        if (encoder is null)
        {
            return;
        }
        var tokenizer = (EmbeddingGemma2Tokenizer)encoder.Tokenizer;
        var cases = Fixture("engine_text_embeddings.json").EnumerateArray().ToArray();
        var texts = cases.Select(c => c.GetProperty("text").GetString()).ToArray();
        var batch = await encoder.EncodeAsync(texts);
        for (int i = 0; i < cases.Length; i++)
        {
            Assert.Equal(cases[i].GetProperty("ids").EnumerateArray().Select(e => e.GetInt32()), tokenizer.EncodeIds(texts[i]));
            var expected = EmbeddingGemma2TestAssets.ReadFloats(cases[i].GetProperty("embedding"));
            Assert.Equal(768, batch[i].Length);
            double cos = EmbeddingGemma2TestAssets.Cosine(batch[i], expected);
            Assert.True(cos >= 0.995, $"{JsonSerializer.Serialize(texts[i])}: cosine {cos:F5}");
            Assert.Equal(1.0, Math.Sqrt(batch[i].Sum(v => (double)v * v)), 4);
        }
        // Batching must not change results (each sequence attends only to itself).
        var single = await encoder.EncodeAsync(texts[2] + " ");
        var again = await encoder.EncodeAsync(new[] { texts[0] + " ", texts[2] + " " });
        Assert.True(EmbeddingGemma2TestAssets.Cosine(single, again[1]) > 0.99999);
    }

    [Fact]
    public async Task Text_MatryoshkaTruncationAndOverflow()
    {
        var encoder = AnyEncoder();
        if (encoder is null)
        {
            return;
        }
        const string Text = "Matryoshka representation learning keeps prefixes useful.";
        var full = await encoder.EncodeAsync(Text);
        var dimension = encoder.OutputDimension;
        var strategy = encoder.OverflowStrategy;
        try
        {
            encoder.OutputDimension = 256;
            var small = await encoder.EncodeAsync(Text);
            Assert.Equal(256, small.Length);
            var prefix = full.AsSpan(0, 256).ToArray();
            Assert.True(EmbeddingGemma2TestAssets.Cosine(small, prefix) > 0.99999);
            Assert.Equal(1.0, Math.Sqrt(small.Sum(v => (double)v * v)), 4);
            encoder.OutputDimension = 768;

            // ~2000 tokens: the tail beyond the 1024-token window must not matter when truncating.
            var body = string.Concat(Enumerable.Repeat("lorem ipsum dolor ", 700));
            encoder.OverflowStrategy = InputOverflowStrategy.Truncate;
            var a = await encoder.EncodeAsync(body + "alpha beta gamma");
            var b = await encoder.EncodeAsync(body + "completely different ending");
            Assert.True(EmbeddingGemma2TestAssets.Cosine(a, b) > 0.99999);

            encoder.OverflowStrategy = InputOverflowStrategy.ChunkAndAverage;
            var c = await encoder.EncodeAsync(body + "alpha beta gamma");
            Assert.Equal(768, c.Length);
            Assert.True(EmbeddingGemma2TestAssets.Cosine(a, c) < 0.99999);

            encoder.OverflowStrategy = InputOverflowStrategy.Error;
            await Assert.ThrowsAsync<ArgumentException>(() => encoder.EncodeAsync(body + "error"));
        }
        finally
        {
            encoder.OutputDimension = dimension;
            encoder.OverflowStrategy = strategy;
        }
    }

    [Fact]
    public async Task Image_MatchesEngine()
    {
        var encoder = EmbeddingGemma2TestAssets.Encoder(EmbeddingGemma2Model.TextVision440M);
        if (encoder is null)
        {
            return;
        }
        Assert.True(encoder.SupportsImages);
        Assert.False(encoder.SupportsAudio);
        foreach (var c in Fixture("engine_image_embeddings.json").EnumerateArray())
        {
            var file = c.GetProperty("image").GetString();
            var image = EmbeddingGemma2Image.FromFile(EmbeddingGemma2TestAssets.Fixture(file));
            bool interleaved = c.TryGetProperty("text_before", out var before);
            var v = interleaved
                ? await encoder.EncodeContentAsync(new EmbeddingGemma2Content[] { before.GetString(), image, c.GetProperty("text_after").GetString() })
                : await encoder.EncodeImageAsync(image);
            double cos = EmbeddingGemma2TestAssets.Cosine(v, EmbeddingGemma2TestAssets.ReadFloats(c.GetProperty("embedding")));
            Assert.True(cos >= 0.995, $"{file}{(interleaved ? " (interleaved)" : "")}: cosine {cos:F5}");
        }
        await Assert.ThrowsAsync<NotSupportedException>(() => encoder.EncodeAudioAsync(EmbeddingGemma2Audio.FromSamples(new float[16000])));
    }

    [Fact]
    public async Task Audio_MatchesEngine()
    {
        var encoder = EmbeddingGemma2TestAssets.Encoder(EmbeddingGemma2Model.Multimodal740M);
        if (encoder is null)
        {
            return;
        }
        Assert.True(encoder.SupportsAudio);
        foreach (var c in Fixture("engine_audio_embeddings.json").EnumerateArray())
        {
            var file = c.GetProperty("audio").GetString();
            var v = await encoder.EncodeAudioAsync(EmbeddingGemma2Audio.FromFile(EmbeddingGemma2TestAssets.Fixture(file)));
            double cos = EmbeddingGemma2TestAssets.Cosine(v, EmbeddingGemma2TestAssets.ReadFloats(c.GetProperty("embedding")));
            Assert.True(cos >= 0.99, $"{file}: cosine {cos:F5}");
        }
    }

    [Fact]
    public async Task TextOnlyBundle_RejectsImages()
    {
        var encoder = EmbeddingGemma2TestAssets.Encoder(EmbeddingGemma2Model.Text270M);
        if (encoder is null)
        {
            return;
        }
        Assert.False(encoder.SupportsImages);
        var image = EmbeddingGemma2Image.FromRgb(new byte[48 * 48 * 3], 48, 48);
        await Assert.ThrowsAsync<NotSupportedException>(() => encoder.EncodeImageAsync(image));
    }
}
