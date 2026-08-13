using SentenceTransformers.Qwen3;
using SentenceTransformers.Tests.Support;

namespace SentenceTransformers.Tests;

/// <summary>
/// Tests for <see cref="ChunkPrefix"/> and for the chunking helpers reading it. A prefix leads every
/// chunk that goes to the model, so what matters is that it costs the chunk its room (rather than
/// pushing the chunk past the context window), that it can never take over more than its share of
/// that room, and that the chunks reported back stay the plain source text.
/// </summary>
public class ChunkPrefixTests
{
    private const int MaxTokens = 1024;

    private static QwenTokenizer NewQwen() => new(TestPaths.QwenTokenizerJson, MaxTokens);

    [Fact]
    public void Create_WithoutAPrefix_ReturnsNull()
    {
        using var tok = NewQwen();

        Assert.Null(ChunkPrefix.Create(tok, null, chunkLength: 256));
        Assert.Null(ChunkPrefix.Create(tok, "", chunkLength: 256));
        Assert.Null(ChunkPrefix.Create(tok, "   \n ", chunkLength: 256));
    }

    [Fact]
    public void Create_ShortPrefix_IsKeptWholeAndCosts_ItsTokens()
    {
        using var tok = NewQwen();

        var prefix = ChunkPrefix.Create(tok, "Quarterly report", chunkLength: 256);

        Assert.False(prefix.WasCropped);
        Assert.Equal("Quarterly report\n", prefix.Text);
        Assert.Equal(tok.TokenizeRaw(prefix.Text.AsSpan()).Count, prefix.TokenCount);
        Assert.Equal(256 - prefix.TokenCount, ChunkPrefix.EffectiveChunkLength(prefix, 256));
    }

    [Fact]
    public void Create_LongPrefix_IsCroppedToItsShareOfTheChunk()
    {
        using var tok = NewQwen();

        var longTitle = string.Join(' ', Enumerable.Range(0, 400).Select(i => $"word{i}"));

        //Half of a small chunk...
        var small = ChunkPrefix.Create(tok, longTitle, chunkLength: 64);
        Assert.True(small.WasCropped);
        Assert.True(small.TokenCount <= 64 / 2 + 2, $"Prefix took {small.TokenCount} tokens of a 64-token chunk");
        Assert.True(longTitle.StartsWith(small.Text.TrimEnd()), "The cropped prefix should be a prefix of the original");

        //...but never more than the hard cap, however large the chunk is
        var large = ChunkPrefix.Create(tok, longTitle, chunkLength: 4096);
        Assert.True(large.WasCropped);
        Assert.True(large.TokenCount <= ChunkPrefix.MaxTokens + 2, $"Prefix took {large.TokenCount} tokens");
    }

    [Fact]
    public void MaxTokensFor_IsHalfTheChunkUpToTheCap()
    {
        Assert.Equal(32,                     ChunkPrefix.MaxTokensFor(64));
        Assert.Equal(128,                    ChunkPrefix.MaxTokensFor(256));
        Assert.Equal(ChunkPrefix.MaxTokens,  ChunkPrefix.MaxTokensFor(8192));
        Assert.Equal(1,                      ChunkPrefix.MaxTokensFor(1));
    }

    [Fact]
    public void EffectiveChunkLength_WithoutAPrefix_IsUnchanged()
    {
        Assert.Equal(512, ChunkPrefix.EffectiveChunkLength(null, 512));
        Assert.Equal("chunk", ChunkPrefix.Apply(null, "chunk"));
    }

    [Fact]
    public void ChunkTokens_LeavesRoomForThePrefix()
    {
        using var tok = NewQwen();

        var text   = string.Join(' ', Enumerable.Range(0, 600).Select(i => $"word{i}"));
        var prefix = ChunkPrefix.Create(tok, "Annual report of the working group", chunkLength: 64);

        var withPrefix = BPEChunkAndEncodeHelpers.ChunkTokens(tok, text, chunkLength: 64, chunkOverlap: 4, prefix: prefix);
        var without    = BPEChunkAndEncodeHelpers.ChunkTokens(tok, text, chunkLength: 64, chunkOverlap: 4);

        Assert.True(withPrefix.Count > without.Count, "Shorter chunks means more of them");

        //Prefix + chunk has to fit the budget the caller asked for
        foreach (var chunk in withPrefix)
        {
            var tokens = tok.TokenizeRaw(ChunkPrefix.Apply(prefix, chunk).AsSpan()).Count;
            Assert.True(tokens <= 64, $"Prefix + chunk took {tokens} tokens of a 64-token budget");
        }
    }

    [Fact]
    public async Task ChunkAndEncodeAsync_PrefixesEveryChunkButReportsThePlainText()
    {
        using var tok     = NewQwen();
        using var encoder = new FakeSentenceEncoder(tok, MaxTokens);

        var text   = string.Join(' ', Enumerable.Range(0, 300).Select(i => $"word{i}"));
        var prefix = ChunkPrefix.Create(tok, "Contract with ACME", chunkLength: 64);

        var encoded = await BPEChunkAndEncodeHelpers.ChunkAndEncodeAsync(encoder, text, chunkLength: 64, chunkOverlap: 4, prefix: prefix);

        Assert.True(encoded.Length > 1, "Expected more than one chunk");

        //Every chunk went to the model with the prefix in front of it...
        var sentToModel = encoder.ReceivedBatches.SelectMany(b => b).ToArray();
        Assert.Equal(encoded.Length, sentToModel.Length);
        Assert.All(sentToModel, s => Assert.StartsWith(prefix.Text, s));

        //...while the chunk each vector belongs to stays the plain source text
        Assert.All(encoded, c => Assert.DoesNotContain(prefix.Text, c.Text));
        Assert.All(encoded, c => Assert.Contains(c.Text, text));
    }

    [Fact]
    public async Task ChunkAndEncodeAsync_WordPieceDefault_AlsoPrefixesEveryChunk()
    {
        //The WordPiece path (MiniLM, Arctic) goes through the ISentenceEncoder default implementations
        using var enc = new MiniLM.SentenceEncoder();
        ISentenceEncoder iface = enc;

        var text   = string.Join(' ', Enumerable.Range(0, 300).Select(i => $"word{i}"));
        var prefix = ChunkPrefix.Create(enc.Tokenizer, "Meeting notes", chunkLength: 64);

        var encoded = await iface.ChunkAndEncodeAsync(text, chunkLength: 64, chunkOverlap: 4, prefix: prefix);

        Assert.True(encoded.Length > 1, "Expected more than one chunk");
        Assert.All(encoded, c => Assert.DoesNotContain(prefix.Text, c.Text));

        var chunksWithout = iface.ChunkTokens(text, chunkLength: 64, chunkOverlap: 4);
        Assert.True(encoded.Length > chunksWithout.Count, "Shorter chunks means more of them");
    }
}
