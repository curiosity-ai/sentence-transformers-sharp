using SentenceTransformers.Qwen3;
using SentenceTransformers.Tests.Support;

namespace SentenceTransformers.Tests;

/// <summary>
/// End-to-end tests for the BPE chunk+encode pipeline, using a <see cref="FakeSentenceEncoder"/>
/// so no model is loaded. These tests focus on:
/// <list type="bullet">
/// <item>What text actually reaches <c>EncodeAsync</c> after chunking.</item>
/// <item>Marker injection / stripping: tags are recovered, markers are removed before encoding.</item>
/// <item>Cancellation and <c>keepResultsOnCancellation</c>.</item>
/// </list>
/// </summary>
public class BPEChunkAndEncodeTests
{
    private const int MaxTokens = 1024;

    private static (FakeSentenceEncoder encoder, QwenTokenizer tokenizer) NewFakeWithQwen()
    {
        var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, MaxTokens);
        return (new FakeSentenceEncoder(tok, MaxTokens), tok);
    }

    [Fact]
    public async Task ChunkAndEncodeAsync_ReturnsOneVectorPerChunk()
    {
        var (encoder, tok) = NewFakeWithQwen();
        try
        {
            var text = string.Join(' ', Enumerable.Range(0, 200).Select(i => $"word{i}"));
            var result = await BPEChunkAndEncodeHelpers.ChunkAndEncodeAsync(encoder, text, chunkLength: 16, chunkOverlap: 2);
            Assert.NotEmpty(result);
            Assert.All(result, c =>
            {
                Assert.False(string.IsNullOrEmpty(c.Text));
                Assert.NotEmpty(c.Vector);
            });
        }
        finally
        {
            tok.Dispose();
        }
    }

    [Fact]
    public async Task ChunkAndEncodeAsync_Sequentially_SendsOneChunkPerCall()
    {
        var (encoder, tok) = NewFakeWithQwen();
        try
        {
            var text = string.Join(' ', Enumerable.Range(0, 100).Select(i => $"w{i}"));
            var result = await BPEChunkAndEncodeHelpers.ChunkAndEncodeAsync(encoder, text, chunkLength: 16, chunkOverlap: 2, sequentially: true);

            Assert.Equal(result.Length, encoder.CallCount);
            Assert.All(encoder.ReceivedBatches, b => Assert.Single(b));
        }
        finally
        {
            tok.Dispose();
        }
    }

    [Fact]
    public async Task ChunkAndEncodeAsync_Batched_SendsAllChunksInOneCall()
    {
        var (encoder, tok) = NewFakeWithQwen();
        try
        {
            var text = string.Join(' ', Enumerable.Range(0, 100).Select(i => $"w{i}"));
            var result = await BPEChunkAndEncodeHelpers.ChunkAndEncodeAsync(encoder, text, chunkLength: 16, chunkOverlap: 2, sequentially: false);

            Assert.Equal(1, encoder.CallCount);
            Assert.Equal(result.Length, encoder.ReceivedBatches[0].Length);
        }
        finally
        {
            tok.Dispose();
        }
    }

    [Fact]
    public async Task ChunkAndEncodeAlignedAsync_OffsetsMatchSource()
    {
        var (encoder, tok) = NewFakeWithQwen();
        try
        {
            var text = "The quick brown fox jumps over the lazy dog. " +
                       "Pack my box with five dozen liquor jugs. " +
                       "Sphinx of black quartz, judge my vow.";

            var result = await BPEChunkAndEncodeHelpers.ChunkAndEncodeAlignedAsync(encoder, text, chunkLength: 12, chunkOverlap: 2);
            Assert.NotEmpty(result);
            Assert.All(result, c =>
            {
                Assert.Equal(text, c.OriginalText);
                Assert.True(c.Start >= 0);
                Assert.True(c.ApproximateEnd <= text.Length);
                Assert.Equal(text.Substring(c.Start, c.ApproximateEnd - c.Start), c.Text);
            });
        }
        finally
        {
            tok.Dispose();
        }
    }

    [Fact]
    public async Task ChunkAndEncodeTaggedAsync_StripsInjectedMarkers_AndRecoversTags()
    {
        var (encoder, tok) = NewFakeWithQwen();
        try
        {
            var text = "⁎1⁑page one body. more page one. ⁎2⁑page two body. yet more page two text content here.";

            var result = await BPEChunkAndEncodeHelpers.ChunkAndEncodeTaggedAsync(
                encoder, text, PageMarker.StripPageTags, chunkLength: 16, chunkOverlap: 2);
            Assert.NotEmpty(result);

            // The encoder must not see any of the injected markers — they were stripped first.
            foreach (var batch in encoder.ReceivedBatches)
            {
                foreach (var s in batch)
                {
                    Assert.DoesNotContain(PageMarker.Start, s);
                    Assert.DoesNotContain(PageMarker.End, s);
                }
            }

            // Every chunk's Text equals what was sent to EncodeAsync.
            foreach (var c in result)
            {
                Assert.DoesNotContain(PageMarker.Start, c.Text);
                Assert.DoesNotContain(PageMarker.End, c.Text);
            }

            // At least some chunks should carry a non-empty tag derived from the markers.
            Assert.Contains(result, c => !string.IsNullOrEmpty(c.Tag));

            // Tags should only ever be "1", "2", or "1-2" given our input.
            Assert.All(result, c =>
            {
                Assert.Contains(c.Tag, new[] { string.Empty, "1", "2", "1-2" });
            });
        }
        finally
        {
            tok.Dispose();
        }
    }

    [Fact]
    public async Task ChunkAndEncodeTaggedAlignedAsync_OffsetsMatchSource_AndMarkersStripped()
    {
        var (encoder, tok) = NewFakeWithQwen();
        try
        {
            var text = "⁎1⁑First page sentence. Another first page sentence here. ⁎2⁑Second page sentence. Another second page sentence.";

            var result = await BPEChunkAndEncodeHelpers.ChunkAndEncodeTaggedAlignedAsync(
                encoder, text, PageMarker.StripPageTags, chunkLength: 16, chunkOverlap: 2);

            Assert.NotEmpty(result);
            Assert.All(result, c =>
            {
                Assert.Equal(text, c.OriginalText);
                Assert.True(c.Start >= 0 && c.ApproximateEnd <= text.Length);
                Assert.DoesNotContain(PageMarker.Start, c.Text);
                Assert.DoesNotContain(PageMarker.End, c.Text);
            });

            // The substring extracted from OriginalText should still contain markers (those are
            // injected in the source); only the cleaned chunk Text drops them.
            Assert.Contains(result, c => text.Substring(c.Start, c.ApproximateEnd - c.Start).Contains(PageMarker.Start));
        }
        finally
        {
            tok.Dispose();
        }
    }

    [Fact]
    public async Task ChunkAndEncodeTaggedAsync_NullStripTags_Throws()
    {
        var (encoder, tok) = NewFakeWithQwen();
        try
        {
            await Assert.ThrowsAsync<ArgumentNullException>(async () =>
                await BPEChunkAndEncodeHelpers.ChunkAndEncodeTaggedAsync(encoder, "hi", null));
        }
        finally
        {
            tok.Dispose();
        }
    }

    [Fact]
    public async Task ChunkAndEncodeAsync_Cancellation_WithKeepResults_ReturnsPartial()
    {
        // Encoder that throws OperationCanceledException after the first batch.
        using var tok = new QwenTokenizer(TestPaths.QwenTokenizerJson, MaxTokens);
        using var cts = new CancellationTokenSource();

        var encoder = new CountingCancellingEncoder(tok, MaxTokens, cancelAfter: 2, cts);

        var text = string.Join(' ', Enumerable.Range(0, 200).Select(i => $"w{i}"));

        var result = await BPEChunkAndEncodeHelpers.ChunkAndEncodeAsync(
            encoder, text, chunkLength: 16, chunkOverlap: 2,
            sequentially: true, keepResultsOnCancellation: true,
            cancellationToken: cts.Token);

        // We expect at least one chunk to have been encoded before cancellation.
        Assert.NotEmpty(result);
        // ...but fewer than the full chunk count.
        var allChunks = BPEChunkAndEncodeHelpers.ChunkTokens(tok, text, 16, 2);
        Assert.True(result.Length < allChunks.Count,
            $"Expected partial results (<{allChunks.Count}), got {result.Length}");
    }

    /// <summary>Encoder helper that cancels its <see cref="CancellationTokenSource"/> after a fixed number of calls.</summary>
    private sealed class CountingCancellingEncoder : ISentenceEncoder
    {
        private readonly CancellationTokenSource _cts;
        private readonly int _cancelAfter;
        private int _calls;

        public CountingCancellingEncoder(BERTTokenizers.Base.TokenizerBase tokenizer, int maxChunkLength, int cancelAfter, CancellationTokenSource cts)
        {
            Tokenizer = tokenizer;
            MaxChunkLength = maxChunkLength;
            _cancelAfter = cancelAfter;
            _cts = cts;
        }

        public int MaxChunkLength { get; }
        public BERTTokenizers.Base.TokenizerBase Tokenizer { get; }

        public Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            _calls++;
            if (_calls > _cancelAfter)
            {
                _cts.Cancel();
                cancellationToken.ThrowIfCancellationRequested();
            }
            var v = new float[sentences.Length][];
            for (int i = 0; i < sentences.Length; i++) v[i] = new float[1] { sentences[i].Length };
            return Task.FromResult(v);
        }

        public void Dispose() { }
    }
}
