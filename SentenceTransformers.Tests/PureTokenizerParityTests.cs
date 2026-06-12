using SentenceTransformers.Harrier.Small.Pure.Tokenizer;
using SentenceTransformers.Tests.Support;
using HFTokenizer = Tokenizers.HuggingFace.Tokenizer.Tokenizer;

namespace SentenceTransformers.Tests;

/// <summary>
/// Verifies that the dependency-free <see cref="HarrierSmallPureTokenizer"/> (pure-C# Gemma BPE)
/// produces exactly the same token ids as the native <c>Tokenizers.HuggingFace</c> implementation
/// for the same <c>tokenizer.json</c>. This is the contract the pure encoder relies on, so any drift
/// in the BPE/normalizer/added-token handling shows up here without needing to load the model.
/// </summary>
public class PureTokenizerParityTests
{
    private const int MaxTokens = 32768;

    public static IEnumerable<object[]> Samples =>
    [
        ["how much protein should a female eat"],
        ["summit define"],
        ["Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: how much protein should a female eat"],
        ["As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day."],
        ["Good morning"],
        ["Buenos días"],
        ["おはよう"],
        ["Привет, как дела?"],
        ["Naïve café — résumé façade."],
        ["Multiple   spaces\tand\ttabs\n\nand newlines."],
        ["中文测试：你好，世界！"],
        ["Emoji test 👍🏽🚀 done."],
        ["a"],
    ];

    [Theory]
    [MemberData(nameof(Samples))]
    public void PureTokenizer_MatchesNativeHuggingFace_TokenIds(string text)
    {
        var pure = HarrierSmallPureTokenizer.FromFile(TestPaths.HarrierSmallTokenizerJson, MaxTokens);
        var hf = HFTokenizer.FromFile(TestPaths.HarrierSmallTokenizerJson);

        var pureIds = pure.Encode([text], addSpecialTokens: true)[0].InputIds.Select(x => (int)x).ToArray();
        var hfIds = hf.Encode(text, addSpecialTokens: true,
                includeTypeIds: false, includeTokens: false, includeWords: false,
                includeOffsets: false, includeSpecialTokensMask: false,
                includeAttentionMask: false, includeOverflowing: false)
            .First().Ids.Select(x => (int)x).ToArray();

        Assert.Equal(hfIds, pureIds);
    }

    [Fact]
    public void PureTokenizer_AddsBosAndEos()
    {
        var pure = HarrierSmallPureTokenizer.FromFile(TestPaths.HarrierSmallTokenizerJson, MaxTokens);
        var ids = pure.EncodeIds("hello world");
        Assert.True(ids.Length >= 3);
        Assert.Equal(2, ids[0]);          // <bos>
        Assert.Equal(1, ids[^1]);         // <eos>
    }

    [Fact]
    public void PureTokenizer_TokenizeRawAligned_OffsetsCoverSource()
    {
        var pure = HarrierSmallPureTokenizer.FromFile(TestPaths.HarrierSmallTokenizerJson, MaxTokens);
        var text = "Bonjour le monde, ceci est un test.";
        var aligned = pure.TokenizeRawAligned(text);

        Assert.NotEmpty(aligned);
        Assert.Equal(0, aligned[0].Start);
        Assert.Equal(text.Length, aligned[^1].ApproximateEnd);

        foreach (var t in aligned)
        {
            var expected = text.Substring(t.Start, t.ApproximateEnd - t.Start);
            Assert.Equal(expected, t.Original);
        }
    }

    [Fact]
    public void PureTokenizer_Encode_TruncatesToMaxTokens()
    {
        var pure = HarrierSmallPureTokenizer.FromFile(TestPaths.HarrierSmallTokenizerJson, 8);
        var encoded = pure.Encode(["one two three four five six seven eight nine ten eleven twelve"]);
        Assert.Single(encoded);
        Assert.True(encoded[0].InputIds.Length <= 8);
        // Truncation must keep <eos> last so the model's last-token pooling stays well-defined.
        Assert.Equal(1, encoded[0].InputIds[^1]);
    }
}
