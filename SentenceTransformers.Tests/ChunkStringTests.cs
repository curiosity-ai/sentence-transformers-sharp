namespace SentenceTransformers.Tests;

/// <summary>
/// Tests for the tokenizer-free <see cref="ISentenceEncoder.ChunkString"/> static helper.
/// </summary>
public class ChunkStringTests
{
    [Fact]
    public void ChunkString_ShortText_ReturnsSingleChunk()
    {
        var chunks = ISentenceEncoder.ChunkString("hello world", chunkLength: 100, chunkOverlap: 10);
        Assert.Single(chunks);
        Assert.Equal("hello world", chunks[0]);
    }

    [Fact]
    public void ChunkString_LongText_SplitsOnWhitespace_AndRespectsChunkLength()
    {
        var words = Enumerable.Range(0, 50).Select(i => $"w{i}").ToArray();
        var text = string.Join(' ', words);

        var chunks = ISentenceEncoder.ChunkString(text, chunkLength: 30, chunkOverlap: 8);
        Assert.True(chunks.Count > 1, $"Expected multiple chunks, got {chunks.Count}");
        Assert.All(chunks, c => Assert.True(c.Length <= 30, $"Chunk length {c.Length} exceeded chunkLength=30: {c}"));
    }

    [Fact]
    public void ChunkString_MaxChunksCap_IsHonored()
    {
        var words = Enumerable.Range(0, 200).Select(i => $"w{i}").ToArray();
        var text = string.Join(' ', words);

        var chunks = ISentenceEncoder.ChunkString(text, chunkLength: 30, chunkOverlap: 5, maxChunks: 3);
        Assert.True(chunks.Count <= 4, $"Expected ~3 chunks, got {chunks.Count}");
    }

    [Fact]
    public void ChunkString_EmptyInput_ReturnsEmpty()
    {
        var chunks = ISentenceEncoder.ChunkString(string.Empty);
        Assert.Empty(chunks);
    }
}
