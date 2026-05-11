using SentenceTransformers.Harrier;
using SentenceTransformers.Qwen3;
using SentenceTransformers.Tests.Support;

namespace SentenceTransformers.Tests;

/// <summary>
/// Tests for the Hugging Face byte-level BPE tokenizers used by Qwen3 and Harrier. The tests
/// construct the tokenizers directly from the shipped <c>tokenizer.json</c> files (no ONNX model
/// is loaded).
/// </summary>
public class BPETokenizerTests
{
    private const int MaxTokens = 1024;

    [Fact]
    public void QwenTokenizer_TokenizeRaw_OriginalsConcatenateToSource()
    {
        using var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, MaxTokens);
        var text = "Hello world, this is a test.";
        var raw = tok.TokenizeRaw(text);
        Assert.NotEmpty(raw);

        // Byte-level BPE offsets are expected to be contiguous; concatenating the originals
        // should reconstruct the source exactly. (This is what BPEChunkAndEncodeHelpers relies on
        // indirectly via offsets — verifying it here catches tokenizer-config drift.)
        var concat = string.Concat(raw.Select(t => t.Original ?? string.Empty));
        Assert.Equal(text, concat);
    }

    [Fact]
    public void QwenTokenizer_TokenizeRawAligned_OffsetsAreContiguousAndMonotonic()
    {
        using var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, MaxTokens);
        var text = "The quick brown fox jumps over the lazy dog.";
        var aligned = tok.TokenizeRawAligned(text);
        Assert.NotEmpty(aligned);

        Assert.Equal(0, aligned[0].Start);
        Assert.Equal(text.Length, aligned[^1].ApproximateEnd);

        for (int i = 1; i < aligned.Count; i++)
        {
            Assert.True(aligned[i].Start >= aligned[i - 1].Start,
                $"Token {i} starts at {aligned[i].Start} but previous at {aligned[i - 1].Start}");
        }

        // Every aligned token's Original must be the exact substring of source at its offsets.
        foreach (var t in aligned)
        {
            var expected = text.Substring(t.Start, t.ApproximateEnd - t.Start);
            Assert.Equal(expected, t.Original);
        }
    }

    [Fact]
    public void QwenTokenizer_Encode_AddsSpecialTokens_AndProducesAlignedArrays()
    {
        using var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, MaxTokens);
        var encoded = tok.Encode(["hello", "hi there world"]);
        Assert.Equal(2, encoded.Count);
        Assert.All(encoded, row =>
        {
            Assert.NotEmpty(row.InputIds);
            Assert.Equal(row.InputIds.Length, row.AttentionMask.Length);
            Assert.Equal(row.InputIds.Length, row.TokenTypeIds.Length);
        });
    }

    [Fact]
    public void QwenTokenizer_Encode_TruncatesToMaxTokens()
    {
        var maxTokens = 8;
        using var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, maxTokens);
        var encoded = tok.Encode(["one two three four five six seven eight nine ten eleven twelve thirteen"]);
        Assert.Single(encoded);
        Assert.True(encoded[0].InputIds.Length <= maxTokens,
            $"InputIds length {encoded[0].InputIds.Length} exceeds maxTokens {maxTokens}");
    }

    [Fact]
    public void QwenTokenizer_TokenizeRaw_OnEmptyText_ReturnsEmptyList()
    {
        using var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, MaxTokens);
        var raw = tok.TokenizeRaw(string.Empty);
        Assert.Empty(raw);
    }

    [Fact]
    public void QwenTokenizer_TokenizeRawAligned_OffsetsCoverEveryCharacter_EvenForUnicodeMarkers()
    {
        // For some Unicode inputs the HF BPE tokenizer emits tokens with overlapping offsets, so
        // concatenating Original substrings naively may not reproduce the source. The offset-based
        // BPE chunker handles this correctly because it slices the source between the first and
        // last token's offsets rather than relying on per-token text round-trips. This test asserts
        // the property the chunker actually depends on: the union of every token's
        // [Start, ApproximateEnd) range covers every character of the input, and the first/last
        // token bracket the source exactly.
        using var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, MaxTokens);
        var text = "⁎1⁑ first page text ⁎2⁑ second page text";
        var aligned = tok.TokenizeRawAligned(text);

        Assert.NotEmpty(aligned);
        Assert.Equal(0, aligned[0].Start);
        Assert.Equal(text.Length, aligned[^1].ApproximateEnd);

        var covered = new bool[text.Length];
        foreach (var t in aligned)
        {
            for (int i = t.Start; i < t.ApproximateEnd && i < text.Length; i++) covered[i] = true;
        }
        for (int i = 0; i < covered.Length; i++)
        {
            Assert.True(covered[i], $"Character at offset {i} ('{text[i]}') is not covered by any token");
        }
    }

    [Fact]
    public void HarrierTokenizer_TokenizeRaw_OriginalsConcatenateToSource()
    {
        using var tok = new HarrierTokenizer(TestPaths.HarrierTokenizerJson, MaxTokens);
        var text = "Multilingual embeddings test 123.";
        var raw = tok.TokenizeRaw(text);
        var concat = string.Concat(raw.Select(t => t.Original ?? string.Empty));
        Assert.Equal(text, concat);
    }

    [Fact]
    public void HarrierTokenizer_TokenizeRawAligned_OffsetsCoverSource()
    {
        using var tok = new HarrierTokenizer(TestPaths.HarrierTokenizerJson, MaxTokens);
        var text = "Bonjour le monde, ceci est un test.";
        var aligned = tok.TokenizeRawAligned(text);
        Assert.NotEmpty(aligned);
        Assert.Equal(0, aligned[0].Start);
        Assert.Equal(text.Length, aligned[^1].ApproximateEnd);
    }

    [Fact]
    public void HarrierTokenizer_Encode_ProducesAlignedArrays()
    {
        using var tok = new HarrierTokenizer(TestPaths.HarrierTokenizerJson, MaxTokens);
        var encoded = tok.Encode(["hello world"]);
        Assert.Single(encoded);
        Assert.Equal(encoded[0].InputIds.Length, encoded[0].AttentionMask.Length);
        Assert.Equal(encoded[0].InputIds.Length, encoded[0].TokenTypeIds.Length);
    }

    [Fact]
    public void HarrierTokenizer_TokenizeSimple_NotEmpty()
    {
        using var tok = new HarrierTokenizer(TestPaths.HarrierTokenizerJson, MaxTokens);
        var tokens = tok.TokenizeSimple("Embeddings for multilingual text");
        Assert.NotEmpty(tokens);
    }
}
