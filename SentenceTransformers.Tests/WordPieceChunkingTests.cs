using SentenceTransformers.MiniLM;
using SentenceTransformers.Tests.Support;

namespace SentenceTransformers.Tests;

/// <summary>
/// Tests for the WordPiece-oriented chunking defaults that <see cref="ISentenceEncoder"/> provides
/// (used by MiniLM and Arctic). The MiniLM encoder is used as a vehicle for the interface default
/// methods. The ONNX model is loaded from the embedded resources of the MiniLM assembly; no
/// network access is needed.
/// </summary>
public class WordPieceChunkingTests
{
    [Fact]
    public void ChunkTokens_LongText_ProducesMultipleChunks()
    {
        using var enc = new SentenceEncoder();
        ISentenceEncoder iface = enc;

        var text = string.Join(' ', Enumerable.Range(0, 400).Select(i => $"word{i}"));
        var chunks = iface.ChunkTokens(text, chunkLength: 32, chunkOverlap: 4);
        Assert.True(chunks.Count > 1, "Expected multiple chunks");
        Assert.All(chunks, c => Assert.False(string.IsNullOrWhiteSpace(c)));
    }

    [Fact]
    public void ChunkTokensAligned_ChunksHaveValidOffsets()
    {
        using var enc = new SentenceEncoder();
        ISentenceEncoder iface = enc;

        var text = "The quick brown fox jumps over the lazy dog. " +
                   "Pack my box with five dozen liquor jugs. " +
                   "Sphinx of black quartz, judge my vow.";

        var chunks = iface.ChunkTokensAligned(text, chunkLength: 16, chunkOverlap: 2);
        Assert.NotEmpty(chunks);
        Assert.All(chunks, c =>
        {
            Assert.True(c.Start >= 0);
            Assert.True(c.ApproximateEnd <= text.Length);
            Assert.True(c.ApproximateEnd >= c.Start);
            Assert.Equal(text, c.OriginalText);
        });
    }

    [Fact]
    public void ChunkTokens_MaxChunksCap_IsRespected()
    {
        using var enc = new SentenceEncoder();
        ISentenceEncoder iface = enc;

        var text = string.Join(' ', Enumerable.Range(0, 500).Select(i => $"w{i}"));
        var chunks = iface.ChunkTokens(text, chunkLength: 16, chunkOverlap: 2, maxChunks: 3);
        Assert.True(chunks.Count <= 4, $"Expected ~3 chunks, got {chunks.Count}");
    }

    [Fact]
    public async Task ChunkAndEncodeAsync_ReturnsOneVectorPerChunk_AndVectorsAreNormalized()
    {
        using var enc = new SentenceEncoder();
        ISentenceEncoder iface = enc;

        var text = string.Join(' ', Enumerable.Range(0, 200).Select(i => $"word{i}"));
        var result = await iface.ChunkAndEncodeAsync(text, chunkLength: 32, chunkOverlap: 4);

        Assert.NotEmpty(result);
        Assert.All(result, c =>
        {
            Assert.False(string.IsNullOrEmpty(c.Text));
            Assert.Equal(384, c.Vector.Length);
            var norm = MathF.Sqrt(c.Vector.Sum(v => v * v));
            Assert.InRange(norm, 0.99f, 1.01f);
        });
    }

    [Fact]
    public async Task ChunkAndEncodeTaggedAsync_RunsStripTagsCallback_AndEncodesCleanedText()
    {
        // The WordPiece pipeline runs the input through Unidecode + lowercase + delimiter
        // splitting before chunk text is reassembled. Unicode markers (like the page tags used by
        // the BPE tests) do not survive that pipeline, so this test verifies the contract — chunk
        // text is fed through the stripTags callback and the callback's outputs reach EncodeAsync
        // — without relying on a specific marker style.
        using var enc = new SentenceEncoder();
        ISentenceEncoder iface = enc;

        var text = string.Join(' ', Enumerable.Range(0, 60).Select(i => $"sentence{i} alpha beta gamma delta."));

        const string TagValue = "TAG-X";
        const string CleanPrefix = "CLEAN: ";
        TaggedChunk Strip(string chunk) => new TaggedChunk(CleanPrefix + chunk, TagValue);

        var result = await iface.ChunkAndEncodeTaggedAsync(text, Strip, chunkLength: 24, chunkOverlap: 4);
        Assert.NotEmpty(result);

        Assert.All(result, c =>
        {
            Assert.StartsWith(CleanPrefix, c.Text);
            Assert.Equal(TagValue, c.Tag);
            Assert.Equal(384, c.Vector.Length);
        });
    }

    [Fact]
    public async Task EncodeAsync_BatchOfTwo_ProducesAlignedVectorPair()
    {
        using var enc = new SentenceEncoder();
        var vectors = await enc.EncodeAsync(["hello world", "different sentence"]);
        Assert.Equal(2, vectors.Length);
        Assert.Equal(384, vectors[0].Length);
        Assert.Equal(384, vectors[1].Length);

        // The two vectors should differ — same vectors for distinct inputs would mean broken encoding.
        var different = false;
        for (int i = 0; i < vectors[0].Length; i++)
        {
            if (Math.Abs(vectors[0][i] - vectors[1][i]) > 1e-3f) { different = true; break; }
        }
        Assert.True(different, "Different sentences should produce different vectors");
    }
}
