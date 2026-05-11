using BERTTokenizers.Base;
using Mosaik.Core;

namespace SentenceTransformers;

/// <summary>
/// Tokenizer-aware chunking and encoding helpers specialized for byte-level BPE tokenizers
/// (Qwen, Harrier, and similar Hugging Face <c>tokenizer.json</c>-driven encoders).
///
/// Each chunk is reconstructed by slicing the source text between the first and last token's
/// character offsets returned by <see cref="TokenizerBase.TokenizeRawAligned"/>. This is more
/// robust than the WordPiece-oriented <c>Untokenize</c> + <c>string.Join(' ', …)</c> path used by
/// the default <see cref="ISentenceEncoder"/> helpers, because:
/// <list type="bullet">
/// <item>It does not depend on subword markers (<c>##</c>) that BPE tokenizers do not emit.</item>
/// <item>It preserves the exact original whitespace and punctuation, including any injected
/// markers (for example <c>⁎12⁑</c>) added by the caller.</item>
/// <item>It does not rely on token-by-token concatenation being contiguous, which can drop content
/// if a tokenizer's pre-tokenizer ever yields non-adjacent offsets.</item>
/// </list>
/// The marker-injection / marker-stripping flow works the same way as the WordPiece helpers: the
/// caller injects markers into the text, the chunker emits chunk substrings that still contain
/// those markers, and a user-supplied <c>stripTags</c> callback removes them and produces a tag
/// before each chunk is encoded.
/// </summary>
public static class BPEChunkAndEncodeHelpers
{
    /// <summary>Tokenizes <paramref name="text"/> using <paramref name="tokenizer"/> and emits chunks of at most
    /// <paramref name="chunkLength"/> tokens with <paramref name="chunkOverlap"/> tokens of overlap. Each
    /// chunk's text is sliced directly from <paramref name="text"/> between the first and last token's offsets.</summary>
    /// <param name="tokenizer">Tokenizer providing offset-aligned tokenization.</param>
    /// <param name="text">Source text to chunk. May be longer than the tokenizer's context window.</param>
    /// <param name="chunkLength">Maximum tokens per chunk. Values <c>&lt;= 0</c> or above <see cref="TokenizerBase.MaxTokens"/> are clamped to <see cref="TokenizerBase.MaxTokens"/>.</param>
    /// <param name="chunkOverlap">Tokens of overlap kept between consecutive chunks. Out-of-range values default to <c>chunkLength / 5</c>.</param>
    /// <param name="maxChunks">Hard cap on chunks produced.</param>
    /// <param name="reportProgress">Optional progress callback receiving values in <c>[0,1]</c>.</param>
    public static List<string> ChunkTokens(TokenizerBase tokenizer, string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null)
    {
        var aligned = ChunkTokensAligned(tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress);
        var result = new List<string>(aligned.Count);
        for (int i = 0; i < aligned.Count; i++)
        {
            result.Add(aligned[i].Value);
        }
        return result;
    }

    /// <summary>Aligned variant of <see cref="ChunkTokens"/>: each chunk carries the start/last-start/approximate-end
    /// offsets back into <paramref name="text"/>, so callers can recover the original substring via
    /// <see cref="AlignedChunkHelpers.FromOriginal(AlignedString)"/>.</summary>
    /// <inheritdoc cref="ChunkTokens"/>
    public static List<AlignedString> ChunkTokensAligned(TokenizerBase tokenizer, string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null)
    {
        if (tokenizer is null) throw new ArgumentNullException(nameof(tokenizer));
        if (string.IsNullOrEmpty(text)) return new List<AlignedString>();

        var maxTokens = tokenizer.MaxTokens;
        if (chunkLength <= 0 || chunkLength > maxTokens) chunkLength = maxTokens;
        if (chunkOverlap < 0 || chunkOverlap >= chunkLength) chunkOverlap = chunkLength / 5;

        reportProgress?.Invoke(0.001f);

        List<TokenizedTokenAligned> tokens;
        checked
        {
            // Avoid tokenizing material that would be discarded when maxChunks is bounded.
            if (maxChunks != int.MaxValue)
            {
                var charBudget = Math.Min(chunkLength * maxChunks * tokenizer.ApproxCharToTokenRatio, text.Length);
                tokens = tokenizer.TokenizeRawAligned(text.AsSpan(0, charBudget));
            }
            else
            {
                tokens = tokenizer.TokenizeRawAligned(text);
            }
        }

        reportProgress?.Invoke(0.002f);

        var docs = new List<AlignedString>();
        if (tokens.Count == 0) return docs;

        int step = Math.Max(1, chunkLength - chunkOverlap);
        int i = 0;
        var sw = ValueStopwatch.StartNew();

        while (true)
        {
            int end = Math.Min(tokens.Count, i + chunkLength);
            if (end <= i) break;

            var first = tokens[i];
            var last = tokens[end - 1];

            int s = Math.Max(0, first.Start);
            int e = Math.Min(text.Length, last.ApproximateEnd);

            if (e > s)
            {
                var value = text.Substring(s, e - s);
                if (!string.IsNullOrWhiteSpace(value))
                {
                    docs.Add(new AlignedString(value, s, last.Start, e, text));

                    if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
                    {
                        reportProgress(maxChunks == int.MaxValue ? (float)i / tokens.Count : (float)docs.Count / maxChunks);
                        sw = ValueStopwatch.StartNew();
                    }

                    if (docs.Count >= maxChunks) return docs;
                }
            }

            if (end >= tokens.Count) break;
            i += step;
        }

        return docs;
    }

    /// <summary>Splits <paramref name="text"/> into BPE-token-bounded chunks and encodes each chunk to an embedding using
    /// <paramref name="encoder"/>.</summary>
    /// <inheritdoc cref="ISentenceEncoder.ChunkAndEncodeAsync"/>
    public static async Task<EncodedChunk[]> ChunkAndEncodeAsync(ISentenceEncoder encoder, string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
    {
        ClampChunkArgs(encoder, ref chunkLength, ref chunkOverlap);

        var sw = ValueStopwatch.StartNew();
        var chunks = ChunkTokens(encoder.Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress is object ? p => reportProgress(p * 0.5f) : null);
        var encoded = new EncodedChunk[chunks.Count];

        try
        {
            if (sequentially)
            {
                var one = new string[1];
                for (int i = 0; i < chunks.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    one[0] = chunks[i];
                    var v = await encoder.EncodeAsync(one, cancellationToken);
                    encoded[i] = new EncodedChunk(one[0], v[0]);
                    MaybeReportProgress(reportProgress, ref sw, i, chunks.Count);
                }
            }
            else
            {
                var vectors = await encoder.EncodeAsync(chunks.ToArray(), cancellationToken);
                for (int i = 0; i < encoded.Length; i++)
                {
                    encoded[i] = new EncodedChunk(chunks[i], vectors[i]);
                    MaybeReportProgress(reportProgress, ref sw, i, chunks.Count);
                }
            }
        }
        catch (Exception e) when (HandleCancellation(e, cancellationToken, keepResultsOnCancellation))
        {
            return encoded.Where(c => c.Text is not null).ToArray();
        }

        return encoded;
    }

    /// <summary>Aligned variant of <see cref="ChunkAndEncodeAsync"/>: each result carries offsets into <paramref name="text"/>.</summary>
    /// <inheritdoc cref="ISentenceEncoder.ChunkAndEncodeAlignedAsync"/>
    public static async Task<EncodedChunkAligned[]> ChunkAndEncodeAlignedAsync(ISentenceEncoder encoder, string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
    {
        ClampChunkArgs(encoder, ref chunkLength, ref chunkOverlap);

        var sw = ValueStopwatch.StartNew();
        var chunks = ChunkTokensAligned(encoder.Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress is object ? p => reportProgress(p * 0.5f) : null);
        var encoded = new EncodedChunkAligned[chunks.Count];

        try
        {
            if (sequentially)
            {
                var one = new string[1];
                for (int i = 0; i < chunks.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    one[0] = chunks[i].Value;
                    var v = await encoder.EncodeAsync(one, cancellationToken);
                    encoded[i] = new EncodedChunkAligned(one[0], v[0], chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);
                    MaybeReportProgress(reportProgress, ref sw, i, chunks.Count);
                }
            }
            else
            {
                var vectors = await encoder.EncodeAsync(chunks.Select(c => c.Value).ToArray(), cancellationToken);
                for (int i = 0; i < encoded.Length; i++)
                {
                    encoded[i] = new EncodedChunkAligned(chunks[i].Value, vectors[i], chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);
                    MaybeReportProgress(reportProgress, ref sw, i, chunks.Count);
                }
            }
        }
        catch (Exception e) when (HandleCancellation(e, cancellationToken, keepResultsOnCancellation))
        {
            return encoded.Where(c => c.Text is not null).ToArray();
        }

        return encoded;
    }

    /// <summary>Chunks (using BPE offsets), strips injected markers from each chunk via
    /// <paramref name="stripTags"/>, and encodes the cleaned text. The chunker treats markers as
    /// ordinary tokens, so chunk boundaries are independent of which markers each chunk contains.</summary>
    /// <param name="stripTags">Callback that strips markers from a chunk and produces a <see cref="TaggedChunk"/>.</param>
    /// <inheritdoc cref="ISentenceEncoder.ChunkAndEncodeTaggedAsync"/>
    public static async Task<TaggedEncodedChunk[]> ChunkAndEncodeTaggedAsync(ISentenceEncoder encoder, string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
    {
        if (stripTags is null) throw new ArgumentNullException(nameof(stripTags));
        ClampChunkArgs(encoder, ref chunkLength, ref chunkOverlap);

        var sw = ValueStopwatch.StartNew();
        var rawChunks = ChunkTokens(encoder.Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress is object ? p => reportProgress(p * 0.5f) : null);
        var chunks = rawChunks.Select(c => stripTags(c)).ToArray();
        var encoded = new TaggedEncodedChunk[chunks.Length];

        try
        {
            if (sequentially)
            {
                var one = new string[1];
                for (int i = 0; i < chunks.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    one[0] = chunks[i].Text;
                    var v = await encoder.EncodeAsync(one, cancellationToken);
                    encoded[i] = new TaggedEncodedChunk(chunks[i].Text, v[0], chunks[i].Tag);
                    MaybeReportProgress(reportProgress, ref sw, i, chunks.Length);
                }
            }
            else
            {
                var vectors = await encoder.EncodeAsync(chunks.Select(c => c.Text).ToArray(), cancellationToken);
                for (int i = 0; i < encoded.Length; i++)
                {
                    encoded[i] = new TaggedEncodedChunk(chunks[i].Text, vectors[i], chunks[i].Tag);
                    MaybeReportProgress(reportProgress, ref sw, i, chunks.Length);
                }
            }
        }
        catch (Exception e) when (HandleCancellation(e, cancellationToken, keepResultsOnCancellation))
        {
            return encoded.Where(c => c.Text is not null).ToArray();
        }

        return encoded;
    }

    /// <summary>Aligned variant of <see cref="ChunkAndEncodeTaggedAsync"/>.</summary>
    /// <param name="stripTags">Callback that strips markers from the original chunk substring and produces a <see cref="TaggedChunk"/>.</param>
    /// <inheritdoc cref="ISentenceEncoder.ChunkAndEncodeTaggedAlignedAsync"/>
    public static async Task<TaggedEncodedChunkAligned[]> ChunkAndEncodeTaggedAlignedAsync(ISentenceEncoder encoder, string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
    {
        if (stripTags is null) throw new ArgumentNullException(nameof(stripTags));
        ClampChunkArgs(encoder, ref chunkLength, ref chunkOverlap);

        var sw = ValueStopwatch.StartNew();
        var raw = ChunkTokensAligned(encoder.Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress is object ? p => reportProgress(p * 0.5f) : null);

        var chunks = raw.Select(c =>
        {
            var t = stripTags(c.FromOriginal());
            return new TaggedChunkAligned(t.Text, t.Tag, c.Start, c.LastStart, c.ApproximateEnd, text);
        }).ToArray();

        var encoded = new TaggedEncodedChunkAligned[chunks.Length];

        try
        {
            if (sequentially)
            {
                var one = new string[1];
                for (int i = 0; i < chunks.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    one[0] = chunks[i].Text;
                    var v = await encoder.EncodeAsync(one, cancellationToken);
                    encoded[i] = new TaggedEncodedChunkAligned(chunks[i].Text, v[0], chunks[i].Tag, chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);
                    MaybeReportProgress(reportProgress, ref sw, i, chunks.Length);
                }
            }
            else
            {
                var vectors = await encoder.EncodeAsync(chunks.Select(c => c.Text).ToArray(), cancellationToken);
                for (int i = 0; i < encoded.Length; i++)
                {
                    encoded[i] = new TaggedEncodedChunkAligned(chunks[i].Text, vectors[i], chunks[i].Tag, chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);
                    MaybeReportProgress(reportProgress, ref sw, i, chunks.Length);
                }
            }
        }
        catch (Exception e) when (HandleCancellation(e, cancellationToken, keepResultsOnCancellation))
        {
            return encoded.Where(c => c.Text is not null).ToArray();
        }

        return encoded;
    }

    private static void ClampChunkArgs(ISentenceEncoder encoder, ref int chunkLength, ref int chunkOverlap)
    {
        if (chunkLength <= 0 || chunkLength > encoder.MaxChunkLength) chunkLength = encoder.MaxChunkLength;
        if (chunkOverlap < 0 || chunkOverlap > chunkLength) chunkOverlap = chunkLength / 5;
    }

    private static void MaybeReportProgress(Action<float> reportProgress, ref ValueStopwatch sw, int i, int total)
    {
        if (reportProgress is object && (sw.GetElapsedTime() > TimeSpan.FromMilliseconds(300)))
        {
            reportProgress(((float)i / Math.Max(1, total)) * 0.5f + 0.5f);
            sw = ValueStopwatch.StartNew();
        }
    }

    private static bool HandleCancellation(Exception e, CancellationToken cancellationToken, bool keepResultsOnCancellation)
    {
        return keepResultsOnCancellation && (e is OperationCanceledException || cancellationToken.IsCancellationRequested);
    }
}
