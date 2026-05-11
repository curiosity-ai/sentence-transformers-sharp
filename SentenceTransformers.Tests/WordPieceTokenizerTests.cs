using BERTTokenizers.Base;
using SentenceTransformers.ArcticXs;
using SentenceTransformers.MiniLM;

namespace SentenceTransformers.Tests;

/// <summary>
/// Tests for the WordPiece-based tokenizers (MiniLM and Arctic). They share an implementation
/// via <see cref="TokenizerBase"/>, so most behaviors are exercised through the MiniLM tokenizer
/// with a couple of Arctic-specific smoke tests.
/// </summary>
public class WordPieceTokenizerTests
{
    [Fact]
    public void MiniLMTokenizer_TokenizeSimple_ReturnsSubwordTokens()
    {
        var tok = new MiniLMTokenizer();
        var tokens = tok.TokenizeSimple("Hello world");
        Assert.NotEmpty(tokens);
        Assert.Contains("hello", tokens);
        Assert.Contains("world", tokens);
    }

    [Fact]
    public void MiniLMTokenizer_Encode_AddsCLSAndSEP_AndPadsBatch()
    {
        var tok = new MiniLMTokenizer();
        tok.SetMaxTokens(32);
        var encoded = tok.Encode("hello", "hi there world");
        Assert.Equal(2, encoded.Count);

        // All rows are padded to the longest row in the batch.
        var len = encoded[0].InputIds.Length;
        Assert.True(len > 2, "Encoded length should include CLS + content + SEP");
        Assert.All(encoded, row =>
        {
            Assert.Equal(len, row.InputIds.Length);
            Assert.Equal(len, row.AttentionMask.Length);
            Assert.Equal(len, row.TokenTypeIds.Length);
        });

        // The CLS id appears as the first token in every row (it's the special [CLS]).
        var clsId = encoded[0].InputIds[0];
        Assert.All(encoded, row => Assert.Equal(clsId, row.InputIds[0]));
    }

    [Fact]
    public void MiniLMTokenizer_TokenizeRaw_RoundTrips_ViaUntokenize()
    {
        var tok = new MiniLMTokenizer();
        var input = "hello world this is a test";
        var raw = tok.TokenizeRaw(input);
        Assert.NotEmpty(raw);

        var words = tok.Untokenize(raw);
        var joined = string.Join(' ', words);
        Assert.Equal("hello world this is a test", joined);
    }

    [Fact]
    public void MiniLMTokenizer_TokenizeRawAligned_OffsetsCoverInput()
    {
        var tok = new MiniLMTokenizer();
        var input = "hello world this is a test";
        var aligned = tok.TokenizeRawAligned(input);
        Assert.NotEmpty(aligned);

        // The first token should start at 0; the last token should end at or before the input length.
        Assert.Equal(0, aligned[0].Start);
        Assert.True(aligned[^1].ApproximateEnd <= input.Length);

        // Offsets should be non-decreasing.
        for (int i = 1; i < aligned.Count; i++)
        {
            Assert.True(aligned[i].Start >= aligned[i - 1].Start,
                $"Token {i} starts at {aligned[i].Start} but previous starts at {aligned[i - 1].Start}");
        }
    }

    [Fact]
    public void MiniLMTokenizer_TokenizeSentence_StripsRepeatedSpecialChars()
    {
        // The WordPiece tokenizer collapses runs of repeated special chars (per
        // TokenizerBase.RemoveRepeatedSpecialChars). Verify that "????????" becomes one "?" so the
        // chunker doesn't waste tokens on noise.
        var tok = new MiniLMTokenizer();
        var raw = tok.TokenizeRaw("hello????????world");
        var untoken = string.Join(' ', tok.Untokenize(raw));
        Assert.DoesNotContain("????", untoken);
    }

    [Fact]
    public void ArcticTokenizer_TokenizeSimple_ReturnsSubwordTokens()
    {
        var tok = new ArcticTokenizer();
        var tokens = tok.TokenizeSimple("Embedding tests");
        Assert.NotEmpty(tokens);
    }

    [Fact]
    public void ArcticTokenizer_Encode_ProducesAlignedBatch()
    {
        var tok = new ArcticTokenizer();
        tok.SetMaxTokens(64);
        var encoded = tok.Encode("the quick brown fox");
        Assert.Single(encoded);
        Assert.NotEmpty(encoded[0].InputIds);
        Assert.Equal(encoded[0].InputIds.Length, encoded[0].AttentionMask.Length);
    }
}
