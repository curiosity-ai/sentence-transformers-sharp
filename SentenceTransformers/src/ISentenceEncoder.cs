using BERTTokenizers.Base;
using Mosaik.Core;

namespace SentenceTransformers;

/// <summary>A chunk of text paired with its embedding vector.</summary>
/// <param name="Text">The chunk text that was encoded.</param>
/// <param name="Vector">Embedding vector produced for <paramref name="Text"/>.</param>
public record struct EncodedChunk(string              Text, float[] Vector);

/// <summary>A chunk of text paired with its embedding vector and offsets back into the original input.</summary>
/// <param name="Text">The chunk text that was encoded (may have markers stripped).</param>
/// <param name="Vector">Embedding vector produced for <paramref name="Text"/>.</param>
/// <param name="Start">Character offset of the first token of the chunk in <paramref name="OriginalText"/>.</param>
/// <param name="LastStart">Character offset of the last token of the chunk in <paramref name="OriginalText"/>.</param>
/// <param name="ApproximateEnd">Approximate end-of-chunk character offset in <paramref name="OriginalText"/>.</param>
/// <param name="OriginalText">The full source text the chunk was extracted from.</param>
public record struct EncodedChunkAligned(string       Text, float[] Vector, int    Start, int LastStart, int ApproximateEnd, string OriginalText);

/// <summary>
/// A chunk + embedding + tag. The tag is the metadata produced by a <see cref="ISentenceEncoder"/>
/// tagged-chunking call (for example, the page numbers extracted from injected page markers).
/// </summary>
/// <param name="Text">The chunk text that was encoded (markers removed).</param>
/// <param name="Vector">Embedding vector produced for <paramref name="Text"/>.</param>
/// <param name="Tag">Tag value derived from the markers found inside the chunk.</param>
public record struct TaggedEncodedChunk(string        Text, float[] Vector, string Tag);

/// <summary>Aligned tagged chunk: a <see cref="TaggedEncodedChunk"/> plus offsets into the original input.</summary>
public record struct TaggedEncodedChunkAligned(string Text, float[] Vector, string Tag, int Start, int LastStart, int ApproximateEnd, string OriginalText);

/// <summary>
/// Result of stripping injected markers out of a chunk: the cleaned text plus a tag derived from the
/// markers. Returned by the <c>stripTags</c> callback passed to the tagged chunking helpers.
/// </summary>
/// <param name="Text">The chunk text with markers removed.</param>
/// <param name="Tag">Tag derived from the markers (for example, a page range).</param>
public record struct TaggedChunk(string               Text, string  Tag);

/// <summary>Aligned variant of <see cref="TaggedChunk"/> carrying offsets into the original input.</summary>
public record struct TaggedChunkAligned(string        Text, string  Tag, int Start, int LastStart, int ApproximateEnd, string OriginalText);

/// <summary>
/// Common contract for sentence encoders. Provides <see cref="EncodeAsync"/> for direct batch
/// embedding plus a family of default-implementation helpers for tokenizer-aware text chunking,
/// optionally combined with marker-injection / marker-stripping flows. The chunking helpers split
/// long text on token boundaries (so chunks never exceed the model's context window) and can encode
/// each chunk to a vector in a single call.
/// </summary>
public interface ISentenceEncoder: IDisposable
{
    /// <summary>
    /// Maximum number of tokens (including special tokens) the underlying model accepts per call.
    /// Chunking helpers clamp <c>chunkLength</c> to this value.
    /// </summary>
    public int MaxChunkLength { get; }

    /// <summary>The tokenizer used to split text into tokens for chunking and encoding.</summary>
    public TokenizerBase Tokenizer { get; }

    /// <summary>
    /// Encodes a batch of input strings into embedding vectors. Each input must fit in
    /// <see cref="MaxChunkLength"/> tokens (use one of the <c>ChunkAndEncode*</c> helpers when it does not).
    /// </summary>
    /// <param name="sentences">Texts to embed.</param>
    /// <param name="cancellationToken">Token used to terminate the inference run early.</param>
    /// <returns>One embedding vector per input sentence, in the same order.</returns>
    public Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default);

    /// <summary>
    /// Splits <paramref name="text"/> into token-bounded chunks and encodes each chunk to an embedding.
    /// </summary>
    /// <param name="text">Source text. May be longer than <see cref="MaxChunkLength"/>.</param>
    /// <param name="chunkLength">Maximum tokens per chunk. Values <c>&lt;= 0</c> or above <see cref="MaxChunkLength"/> are clamped to <see cref="MaxChunkLength"/>.</param>
    /// <param name="chunkOverlap">Number of tokens to overlap between consecutive chunks. Out-of-range values default to <c>chunkLength / 5</c>.</param>
    /// <param name="sequentially">When true, encode chunks one-by-one (lower peak memory, supports incremental cancellation). When false, encode the whole batch in a single call.</param>
    /// <param name="maxChunks">Hard cap on the number of chunks to produce.</param>
    /// <param name="keepResultsOnCancellation">When true and the operation is cancelled, return the chunks that finished encoding instead of throwing.</param>
    /// <param name="reportProgress">Optional progress callback receiving values in <c>[0,1]</c>.</param>
    /// <param name="cancellationToken">Cancellation token for the encoding loop.</param>
    /// <returns>Array of <see cref="EncodedChunk"/>, one per produced chunk, in source order.</returns>
    public async Task<EncodedChunk[]> ChunkAndEncodeAsync(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
    {
        if (chunkLength <= 0 || chunkLength > MaxChunkLength)
        {
            chunkLength = MaxChunkLength;
        }

        if (chunkOverlap < 0 || chunkOverlap > chunkLength)
        {
            chunkOverlap = chunkLength / 5;
        }

        var sw = ValueStopwatch.StartNew();

        var chunks = ChunkTokens(text, chunkLength, chunkOverlap, maxChunks, reportProgress: reportProgress is object ? p => reportProgress(p * 0.5f) : null);

        var encodedChunks = new EncodedChunk[chunks.Count];

        try
        {
            if (sequentially)
            {
                var oneChunk = new string[1];

                for (int i = 0; i < chunks.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    oneChunk[0] = chunks[i];
                    var oneVector = await EncodeAsync(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new EncodedChunk(oneChunk[0], oneVector[0]);

                    if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                    {
                        reportProgress(((float)i / chunks.Count) * 0.5f + 0.5f);
                        sw = ValueStopwatch.StartNew();
                    }
                }
            }
            else
            {
                var vectors = await EncodeAsync(chunks.ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new EncodedChunk(chunks[i], vectors[i]);

                    if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                    {
                        reportProgress(((float)i / chunks.Count) * 0.5f + 0.5f);
                        sw = ValueStopwatch.StartNew();
                    }
                }
            }
        }
        catch (Exception E)
        {
            if ((E is OperationCanceledException || cancellationToken.IsCancellationRequested) && keepResultsOnCancellation)
            {
                return encodedChunks.Where(c => c.Text is not null).ToArray();
            }
            else
            {
                throw;
            }
        }

        return encodedChunks;
    }

    /// <summary>
    /// Same as <see cref="ChunkAndEncodeAsync"/> but each result also exposes character offsets back
    /// into <paramref name="text"/>, allowing the original (un-normalized) substring to be recovered
    /// via <see cref="AlignedChunkHelpers.FromOriginal(EncodedChunkAligned)"/>.
    /// </summary>
    /// <inheritdoc cref="ChunkAndEncodeAsync"/>
    public async Task<EncodedChunkAligned[]> ChunkAndEncodeAlignedAsync(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
    {
        if (chunkLength <= 0 || chunkLength > MaxChunkLength)
        {
            chunkLength = MaxChunkLength;
        }

        if (chunkOverlap < 0 || chunkOverlap > chunkLength)
        {
            chunkOverlap = chunkLength / 5;
        }

        var chunks = ChunkTokensAligned(text, chunkLength, chunkOverlap, maxChunks, reportProgress: reportProgress is object ? p => reportProgress(p * 0.5f) : null);

        var encodedChunks = new EncodedChunkAligned[chunks.Count];
        var sw            = ValueStopwatch.StartNew();

        try
        {
            if (sequentially)
            {
                var oneChunk = new string[1];

                for (int i = 0; i < chunks.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    oneChunk[0] = chunks[i].Value;
                    var oneVector = await EncodeAsync(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new EncodedChunkAligned(oneChunk[0], oneVector[0], chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);

                    if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                    {
                        reportProgress(((float)i / chunks.Count) * 0.5f + 0.5f);
                        sw = ValueStopwatch.StartNew();
                    }
                }
            }
            else
            {
                var vectors = await EncodeAsync(chunks.Select(v => v.Value).ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new EncodedChunkAligned(chunks[i].Value, vectors[i], chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);

                    if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                    {
                        reportProgress(((float)i / chunks.Count) * 0.5f + 0.5f);
                        sw = ValueStopwatch.StartNew();
                    }
                }
            }
        }
        catch (Exception E)
        {
            if ((E is OperationCanceledException || cancellationToken.IsCancellationRequested) && keepResultsOnCancellation)
            {
                return encodedChunks.Where(c => c.Text is not null).ToArray();
            }
            else
            {
                throw;
            }
        }

        return encodedChunks;
    }

    /// <summary>
    /// Chunks <paramref name="text"/> on token boundaries, then for each chunk invokes
    /// <paramref name="stripTags"/> to extract a tag and remove injected markers before encoding.
    /// </summary>
    /// <remarks>
    /// This supports the "marker injection / marker removal" workflow: callers pre-process the
    /// source text to embed lightweight markers (for example, page boundaries written as
    /// <c>⁎12⁑</c>), pass the marked text in here, and <paramref name="stripTags"/> recovers the
    /// page numbers as the <see cref="TaggedChunk.Tag"/> while returning the cleaned text that is
    /// actually fed to the model. The chunker treats markers as ordinary tokens, so chunk boundaries
    /// are independent of which page a chunk happens to cover.
    /// </remarks>
    /// <param name="text">Source text, optionally with injected markers.</param>
    /// <param name="stripTags">Callback that removes injected markers from a chunk and returns the cleaned text plus a tag derived from those markers.</param>
    /// <inheritdoc cref="ChunkAndEncodeAsync"/>
    public async Task<TaggedEncodedChunk[]> ChunkAndEncodeTaggedAsync(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
    {
        if (chunkLength <= 0 || chunkLength > MaxChunkLength)
        {
            chunkLength = MaxChunkLength;
        }

        if (chunkOverlap < 0 || chunkOverlap > chunkLength)
        {
            chunkOverlap = chunkLength / 5;
        }

        var chunks = ChunkTokens(text, chunkLength, chunkOverlap, maxChunks: maxChunks, reportProgress: reportProgress is object ? p => reportProgress(p * 0.5f) : null)
           .Select(chunk => stripTags(chunk))
           .ToArray();

        var encodedChunks = new TaggedEncodedChunk[chunks.Length];
        var sw            = ValueStopwatch.StartNew();

        try
        {
            if (sequentially)
            {
                var oneChunk = new string[1];

                for (int i = 0; i < chunks.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    oneChunk[0] = chunks[i].Text;
                    var oneVector = await EncodeAsync(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new TaggedEncodedChunk(chunks[i].Text, oneVector[0], chunks[i].Tag);

                    if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                    {
                        reportProgress(((float)i / chunks.Length) * 0.5f + 0.5f);
                        sw = ValueStopwatch.StartNew();
                    }
                }
            }
            else
            {
                var vectors = await EncodeAsync(chunks.Select(c => c.Text).ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new TaggedEncodedChunk(chunks[i].Text, vectors[i], chunks[i].Tag);

                    if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                    {
                        reportProgress(((float)i / chunks.Length) * 0.5f + 0.5f);
                        sw = ValueStopwatch.StartNew();
                    }
                }
            }
        }
        catch (Exception E)
        {
            if ((E is OperationCanceledException || cancellationToken.IsCancellationRequested) && keepResultsOnCancellation)
            {
                return encodedChunks.Where(c => c.Text is not null).ToArray();
            }
            else
            {
                throw;
            }
        }

        return encodedChunks;
    }

    /// <summary>
    /// Aligned variant of <see cref="ChunkAndEncodeTaggedAsync"/>: chunks the (marked) text on token
    /// boundaries, then for each chunk extracts the original (unmarked) substring via the aligned
    /// tokenizer, runs <paramref name="stripTags"/> on it, and encodes the cleaned text. Results
    /// carry offsets back into <paramref name="text"/>.
    /// </summary>
    /// <param name="text">Source text, optionally with injected markers.</param>
    /// <param name="stripTags">Callback that removes injected markers and produces a tag from the chunk.</param>
    /// <inheritdoc cref="ChunkAndEncodeAsync"/>
    public async Task<TaggedEncodedChunkAligned[]> ChunkAndEncodeTaggedAlignedAsync(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
    {
        if (chunkLength <= 0 || chunkLength > MaxChunkLength)
        {
            chunkLength = MaxChunkLength;
        }

        if (chunkOverlap < 0 || chunkOverlap > chunkLength)
        {
            chunkOverlap = chunkLength / 5;
        }

        var chunks = ChunkTokensAligned(text, chunkLength, chunkOverlap, maxChunks: maxChunks, reportProgress: reportProgress is object ? p => reportProgress(p * 0.5f) : null)
           .Select(chunk =>
            {
                var t = stripTags(chunk.FromOriginal());
                return new TaggedChunkAligned(t.Text, t.Tag, chunk.Start, chunk.LastStart, chunk.ApproximateEnd, text);
            })
           .ToArray();

        var encodedChunks = new TaggedEncodedChunkAligned[chunks.Length];

        var sw = ValueStopwatch.StartNew();

        try
        {
            if (sequentially)
            {
                var oneChunk = new string[1];

                for (int i = 0; i < chunks.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    oneChunk[0] = chunks[i].Text;
                    var oneVector = await EncodeAsync(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new TaggedEncodedChunkAligned(chunks[i].Text, oneVector[0], chunks[i].Tag, chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);

                    if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                    {
                        reportProgress(((float)i / chunks.Length) * 0.5f + 0.5f);
                        sw = ValueStopwatch.StartNew();
                    }
                }
            }
            else
            {
                var vectors = await EncodeAsync(chunks.Select(c => c.Text).ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new TaggedEncodedChunkAligned(chunks[i].Text, vectors[i], chunks[i].Tag, chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);

                    if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                    {
                        reportProgress(((float)i / chunks.Length) * 0.5f + 0.5f);
                        sw = ValueStopwatch.StartNew();
                    }
                }
            }
        }
        catch (Exception E)
        {
            if ((E is OperationCanceledException || cancellationToken.IsCancellationRequested) && keepResultsOnCancellation)
            {
                return encodedChunks.Where(c => c.Text is not null).ToArray();
            }
            else
            {
                throw;
            }
        }

        return encodedChunks;
    }

    /// <summary>
    /// Tokenizes <paramref name="text"/> and merges tokens into chunks of at most
    /// <paramref name="chunkLength"/> tokens with <paramref name="chunkOverlap"/> tokens of overlap
    /// between consecutive chunks. The text is first truncated to roughly
    /// <c>chunkLength * maxChunks * Tokenizer.ApproxCharToTokenRatio</c> characters when
    /// <paramref name="maxChunks"/> is bounded, to avoid tokenizing material that would be dropped.
    /// </summary>
    /// <param name="text">Source text to chunk.</param>
    /// <param name="chunkLength">Maximum tokens per chunk.</param>
    /// <param name="chunkOverlap">Tokens of overlap kept between consecutive chunks.</param>
    /// <param name="maxChunks">Hard cap on the number of chunks returned.</param>
    /// <param name="reportProgress">Optional progress callback receiving values in <c>[0,1]</c>.</param>
    /// <returns>The chunks as untokenized strings, in source order.</returns>
    public List<string> ChunkTokens(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null)
    {
        reportProgress?.Invoke(0.001f);

        checked //Ensure the max text substring length computed is not overflowing
        {
            var tokenized = Tokenizer.TokenizeRaw(maxChunks != int.MaxValue ? text.AsSpan(0, Math.Min(chunkLength * maxChunks * Tokenizer.ApproxCharToTokenRatio, text.Length)) : text);
            reportProgress?.Invoke(0.002f);
            return MergeTokenSplits(tokenized, chunkLength, chunkOverlap, maxChunks, reportProgress);
        }
    }

    /// <summary>
    /// Aligned variant of <see cref="ChunkTokens"/>: each chunk carries character offsets back into
    /// the (possibly truncated) source text, so callers can recover the exact original substring via
    /// <see cref="AlignedChunkHelpers.FromOriginal(AlignedString)"/>.
    /// </summary>
    /// <inheritdoc cref="ChunkTokens"/>
    public List<AlignedString> ChunkTokensAligned(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null)
    {
        checked //Ensure the max text substring length computed is not overflowing
        {
            reportProgress?.Invoke(0.001f);
            var tokenized = Tokenizer.TokenizeRawAligned(maxChunks != int.MaxValue ? text.AsSpan(0, Math.Min(chunkLength * maxChunks * Tokenizer.ApproxCharToTokenRatio, text.Length)) : text);
            reportProgress?.Invoke(0.002f);
            return MergeTokenSplitsAligned(tokenized, chunkLength, chunkOverlap, maxChunks, text, reportProgress);
        }
    }

    private List<AlignedString> MergeTokenSplitsAligned(List<TokenizedTokenAligned> splits, int chunkLength, int chunkOverlap, int maxChunks, string originalText, Action<float> reportProgress)
    {
        var docs       = new List<AlignedString>();
        var currentDoc = new List<TokenizedTokenAligned>();
        var sw         = ValueStopwatch.StartNew();

        for (var index = 0; index < splits.Count; index++)
        {
            var d = splits[index];

            if (currentDoc.Count + 1 > chunkLength)
            {
                if (currentDoc.Count > 0)
                {
                    if (currentDoc.Any(c => !string.IsNullOrWhiteSpace(c.Original)))
                    {
                        var untokenized = string.Join(' ', Tokenizer.Untokenize(currentDoc, originalText).Select(v => v.Value));
                        docs.Add(new AlignedString(untokenized, currentDoc[0].Start, currentDoc.Last().Start, currentDoc.Last().ApproximateEnd, originalText));

                        if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                        {
                            reportProgress((float)docs.Count / maxChunks);
                            sw = ValueStopwatch.StartNew();
                        }
                    }

                    while (currentDoc.Count > chunkOverlap || (currentDoc.Count + 1 > chunkLength && currentDoc.Count > 0))
                    {
                        currentDoc.RemoveAt(0);
                    }
                }
            }
            currentDoc.Add(d);

            if (docs.Count > maxChunks)
            {
                return docs;
            }
        }

        string final_doc = string.Join(' ', Tokenizer.Untokenize(currentDoc, originalText).Select(v => v.Value));

        if (!string.IsNullOrWhiteSpace(final_doc))
        {
            docs.Add(new AlignedString(final_doc, currentDoc[0].Start, currentDoc.Last().Start, currentDoc.Last().ApproximateEnd, originalText));
        }

        return docs;
    }

    private List<string> MergeTokenSplits(List<TokenizedToken> splits, int chunkLength, int chunkOverlap, int maxChunks, Action<float> reportProgress)
    {
        var docs       = new List<string>();
        var currentDoc = new List<TokenizedToken>();
        var sw         = ValueStopwatch.StartNew();

        for (var index = 0; index < splits.Count; index++)
        {
            var d = splits[index];

            if (currentDoc.Count + 1 > chunkLength)
            {
                if (currentDoc.Count > 0)
                {
                    if (currentDoc.Any(c => !string.IsNullOrWhiteSpace(c.Original)))
                    {
                        var untokenized = string.Join(' ', Tokenizer.Untokenize(currentDoc));
                        docs.Add(untokenized);
                    }

                    while (currentDoc.Count > chunkOverlap || (currentDoc.Count + 1 > chunkLength && currentDoc.Count > 0))
                    {
                        currentDoc.RemoveAt(0);
                    }
                }
            }
            currentDoc.Add(d);

            if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
            {
                reportProgress((float)index / splits.Count);
                sw = ValueStopwatch.StartNew();
            }

            if (docs.Count > maxChunks)
            {
                return docs;
            }
        }

        string final_doc = string.Join(' ', Tokenizer.Untokenize(currentDoc));

        if (!string.IsNullOrWhiteSpace(final_doc))
        {
            docs.Add(final_doc);
        }

        return docs;
    }

    /// <summary>
    /// Character-based fallback chunker that does not require a tokenizer. Splits on whitespace and
    /// rejoins runs of tokens into chunks of at most <paramref name="chunkLength"/> characters with
    /// <paramref name="chunkOverlap"/> characters of overlap.
    /// </summary>
    /// <param name="text">Source text to chunk.</param>
    /// <param name="separator">Separator used when re-joining whitespace-split tokens.</param>
    /// <param name="chunkLength">Maximum characters per chunk.</param>
    /// <param name="chunkOverlap">Characters of overlap between consecutive chunks.</param>
    /// <param name="maxChunks">Hard cap on the number of chunks returned.</param>
    public static List<string> ChunkString(string text, char separator = ' ', int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue)
    {
        return MergeStringSplits(text.Split(new char[] { '\n', '\r', ' ' }, StringSplitOptions.RemoveEmptyEntries), separator, chunkLength, chunkOverlap, maxChunks);
    }

    private static List<string> MergeStringSplits(IEnumerable<string> splits, char separator, int chunkLength, int chunkOverlap, int maxChunks)
    {
        const int separatorLength = 1;
        var       docs            = new List<string>();
        var       currentDoc      = new List<string>();
        int       total           = 0;

        foreach (string d in splits)
        {
            int len = d.Length;

            if (total + len + (currentDoc.Count > 0 ? separatorLength : 0) > chunkLength)
            {
                if (currentDoc.Count > 0)
                {
                    string doc = string.Join(separator, currentDoc);

                    if (!string.IsNullOrWhiteSpace(doc))
                    {
                        docs.Add(doc);
                    }

                    while (total > chunkOverlap || (total + len + (currentDoc.Count > 0 ? separatorLength : 0) > chunkLength && total > 0))
                    {
                        total -= currentDoc[0].Length + (currentDoc.Count > 1 ? separatorLength : 0);
                        currentDoc.RemoveAt(0);
                    }
                }
            }
            currentDoc.Add(d);
            total += len + (currentDoc.Count > 1 ? separatorLength : 0);

            if (docs.Count > maxChunks) return docs;
        }

        string final_doc = string.Join(separator, currentDoc);

        if (!string.IsNullOrWhiteSpace(final_doc))
        {
            docs.Add(final_doc);
        }

        return docs;
    }
}