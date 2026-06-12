using SentenceTransformers.Harrier.Small.Pure;
using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Harrier.Small.Pure.Tokenizer;
using SentenceTransformers.Tests.Support;

namespace SentenceTransformers.Tests;

/// <summary>
/// Tests for how the pure Harrier Small encoder limits the context length. The model is decoder-only
/// with last-token (&lt;eos&gt;) pooling, so the two things that must hold for any input - no matter how
/// long - are: (1) the sequence is truncated to the model's context window, and (2) the final kept token
/// is still &lt;eos&gt;, otherwise the pooled embedding is taken over an arbitrary interior token.
///
/// These run fully offline against the embedded <c>tokenizer.json</c>; no weights download or native
/// tokenizer is required.
/// </summary>
public class PureContextLimitTests
{
    private const int Bos = 2;
    private const int Eos = 1;

    // A string that tokenizes to comfortably more than the small MaxTokens used below.
    private const string LongText =
        "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen " +
        "sixteen seventeen eighteen nineteen twenty twenty-one twenty-two twenty-three twenty-four";

    private static HarrierSmallPureTokenizer Tokenizer(int maxTokens)
        => HarrierSmallPureTokenizer.FromFile(TestPaths.HarrierSmallTokenizerJson, maxTokens);

    [Fact]
    public void MaxChunkLength_WithinModelContextWindow()
    {
        // The exposed chunk length must never exceed the model's positional context window, otherwise
        // chunks would be produced that the model cannot position-encode. It is deliberately capped far
        // below the 32768-position window: the pure forward pass uses full O(n^2) causal attention, so
        // encoding a 32k-token chunk takes hours of single-threaded CPU (see SentenceEncoder).
        Assert.True(SentenceEncoder.GetMaxChunkLength() <= new Gemma3Config().MaxPositionEmbeddings);
        Assert.Equal(2048, SentenceEncoder.GetMaxChunkLength());
    }

    [Fact]
    public void Tokenizer_MaxTokens_WiredToMaxChunkLength()
    {
        // Mirrors how SentenceEncoder.LoadTokenizer constructs the tokenizer.
        var tok = Tokenizer(SentenceEncoder.GetMaxChunkLength());
        Assert.Equal(SentenceEncoder.GetMaxChunkLength(), tok.MaxTokens);
    }

    [Fact]
    public void EncodeIds_TruncatesToMaxTokens_AndKeepsBosEos()
    {
        const int max = 8;
        var ids = Tokenizer(max).EncodeIds(LongText);

        Assert.Equal(max, ids.Length);          // truncated exactly to the limit
        Assert.Equal(Bos, ids[0]);              // <bos> retained at the front
        Assert.Equal(Eos, ids[^1]);             // <eos> retained at the end (last-token pooling)
    }

    [Fact]
    public void EncodeIds_ShortInput_NotTruncated_StillEndsWithEos()
    {
        var ids = Tokenizer(32768).EncodeIds("hello world");
        Assert.True(ids.Length < 32768);
        Assert.Equal(Bos, ids[0]);
        Assert.Equal(Eos, ids[^1]);
    }

    [Fact]
    public void Encode_Batch_TruncatesToMaxTokens_AndKeepsEos()
    {
        const int max = 10;
        var encoded = Tokenizer(max).Encode([LongText], addSpecialTokens: true);

        Assert.Single(encoded);
        var ids = encoded[0].InputIds;
        Assert.Equal(max, ids.Length);
        Assert.Equal(Bos, ids[0]);
        Assert.Equal(Eos, ids[^1]);
        // attention mask covers every (non-padded) kept token
        Assert.Equal(ids.Length, encoded[0].AttentionMask.Length);
        Assert.All(encoded[0].AttentionMask, m => Assert.Equal(1L, m));
    }

    [Fact]
    public void Encode_WithoutSpecialTokens_DoesNotForceEos()
    {
        // When special tokens are not requested there is no <eos> to preserve; truncation is a plain cut.
        const int max = 6;
        var encoded = Tokenizer(max).Encode([LongText], addSpecialTokens: false);
        var ids = encoded[0].InputIds;
        Assert.Equal(max, ids.Length);
        Assert.NotEqual(Eos, ids[^1]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(16)]
    public void EncodeIds_NeverExceedsMaxTokens_ForVariousLimits(int max)
    {
        var ids = Tokenizer(max).EncodeIds(LongText);
        Assert.True(ids.Length <= max, $"length {ids.Length} exceeded max {max}");
        Assert.Equal(Eos, ids[^1]);
    }
}
