using SentenceTransformers.Harrier;
using SentenceTransformers.Qwen3;
using SentenceTransformers.Tests.Support;

namespace SentenceTransformers.Tests;

/// <summary>
/// Tests for <see cref="BPEChunkAndEncodeHelpers"/>: the offset-based chunker used by Qwen3 and
/// Harrier. The chunk text must always equal the source substring between the first and last
/// token's offsets — preserving whitespace, punctuation, and any injected markers verbatim.
/// </summary>
public class BPEChunkingHelpersTests
{
    private const int MaxTokens = 1024;

    private static QwenTokenizer NewQwen() => new(TestPaths.QwenTokenizerJson, MaxTokens);
    private static HarrierTokenizer NewHarrier() => new(TestPaths.HarrierTokenizerJson, MaxTokens);

    [Fact]
    public void ChunkTokens_EmptyText_ReturnsEmpty()
    {
        using var tok = NewQwen();
        var chunks = BPEChunkAndEncodeHelpers.ChunkTokens(tok, string.Empty);
        Assert.Empty(chunks);
    }

    [Fact]
    public void ChunkTokens_ShortText_FitsInOneChunk()
    {
        using var tok = NewQwen();
        var text = "hello world";
        var chunks = BPEChunkAndEncodeHelpers.ChunkTokens(tok, text, chunkLength: 32, chunkOverlap: 4);
        Assert.Single(chunks);
        Assert.Equal(text, chunks[0]);
    }

    [Fact]
    public void ChunkTokens_LongText_ProducesMultipleChunksWithExpectedOverlap()
    {
        using var tok = NewQwen();
        var text = string.Join(' ', Enumerable.Range(0, 200).Select(i => $"word{i}"));

        var chunks = BPEChunkAndEncodeHelpers.ChunkTokens(tok, text, chunkLength: 16, chunkOverlap: 4);
        Assert.True(chunks.Count > 1, "Expected multiple chunks for long input");

        // Each chunk must be a substring of the source.
        foreach (var c in chunks)
        {
            Assert.Contains(c, text);
        }

        // Consecutive chunks should share content (overlap > 0).
        var aligned = BPEChunkAndEncodeHelpers.ChunkTokensAligned(tok, text, chunkLength: 16, chunkOverlap: 4);
        Assert.Equal(chunks.Count, aligned.Count);
        for (int i = 1; i < aligned.Count; i++)
        {
            Assert.True(aligned[i].Start < aligned[i - 1].ApproximateEnd,
                $"Chunk {i} (start {aligned[i].Start}) should start before chunk {i - 1} ends ({aligned[i - 1].ApproximateEnd}) when overlap > 0");
        }
    }

    [Fact]
    public void ChunkTokensAligned_ChunkValueEqualsSourceSubstring()
    {
        using var tok = NewQwen();
        var text = "The quick brown fox jumps over the lazy dog. Pack my box with five dozen liquor jugs.";

        var aligned = BPEChunkAndEncodeHelpers.ChunkTokensAligned(tok, text, chunkLength: 8, chunkOverlap: 2);
        Assert.NotEmpty(aligned);

        foreach (var c in aligned)
        {
            var expected = text.Substring(c.Start, c.ApproximateEnd - c.Start);
            Assert.Equal(expected, c.Value);
        }
    }

    [Fact]
    public void ChunkTokensAligned_PreservesInjectedMarkers_Verbatim()
    {
        // Inject page markers between sentences; the chunker must keep them verbatim in the chunk
        // text so a downstream stripTags callback can recover the page numbers.
        using var tok = NewQwen();
        var text = "⁎1⁑first sentence one. first sentence two. ⁎2⁑second sentence one. second sentence two.";

        var aligned = BPEChunkAndEncodeHelpers.ChunkTokensAligned(tok, text, chunkLength: 12, chunkOverlap: 2);
        Assert.NotEmpty(aligned);

        var allChunks = string.Concat(aligned.Select(c => c.Value));
        Assert.Contains("⁎1⁑", allChunks);
        Assert.Contains("⁎2⁑", allChunks);

        foreach (var c in aligned)
        {
            Assert.Equal(text.Substring(c.Start, c.ApproximateEnd - c.Start), c.Value);
        }
    }

    [Fact]
    public void ChunkTokens_MaxChunksCap_IsHonored()
    {
        using var tok = NewQwen();
        var text = string.Join(' ', Enumerable.Range(0, 500).Select(i => $"w{i}"));

        var chunks = BPEChunkAndEncodeHelpers.ChunkTokens(tok, text, chunkLength: 16, chunkOverlap: 2, maxChunks: 3);
        Assert.True(chunks.Count <= 3, $"Expected <=3 chunks, got {chunks.Count}");
    }

    [Fact]
    public void ChunkTokens_ClampsChunkLengthToMaxTokens()
    {
        var maxTokens = 32;
        using var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, maxTokens);
        var text = string.Join(' ', Enumerable.Range(0, 200).Select(i => $"word{i}"));

        // Pass an out-of-range chunkLength; the helper should clamp to maxTokens (32) instead.
        var chunks = BPEChunkAndEncodeHelpers.ChunkTokens(tok, text, chunkLength: 1_000_000, chunkOverlap: 100);
        Assert.NotEmpty(chunks);
    }

    [Fact]
    public void ChunkTokens_DefaultsOverlap_WhenOutOfRange()
    {
        using var tok = NewQwen();
        var text = string.Join(' ', Enumerable.Range(0, 100).Select(i => $"w{i}"));

        // Passing chunkOverlap >= chunkLength should default to chunkLength / 5. We don't assert
        // the exact value but we do require the call to succeed and produce at least one chunk.
        var chunks = BPEChunkAndEncodeHelpers.ChunkTokens(tok, text, chunkLength: 16, chunkOverlap: 16);
        Assert.NotEmpty(chunks);
    }

    [Fact]
    public void ChunkTokensAligned_Harrier_PreservesUnicodeContent()
    {
        using var tok = NewHarrier();
        // Mix of scripts to exercise multi-byte handling.
        var text = "English text. Texte français. 中文文本。 العربية. русский.";
        var aligned = BPEChunkAndEncodeHelpers.ChunkTokensAligned(tok, text, chunkLength: 8, chunkOverlap: 2);
        Assert.NotEmpty(aligned);

        foreach (var c in aligned)
        {
            Assert.Equal(text.Substring(c.Start, c.ApproximateEnd - c.Start), c.Value);
        }
    }

    [Fact]
    public void ChunkTokens_NullTokenizer_Throws()
    {
        Assert.Throws<ArgumentNullException>(() => BPEChunkAndEncodeHelpers.ChunkTokens(null, "text"));
    }
}
