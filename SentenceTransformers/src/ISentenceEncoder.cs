using BERTTokenizers.Base;

namespace SentenceTransformers;

public record struct EncodedChunk(string              Text, float[] Vector);
public record struct EncodedChunkAligned(string       Text, float[] Vector, int    Start, int LastStart, int ApproximateEnd, string OriginalText);
public record struct TaggedEncodedChunk(string        Text, float[] Vector, string Tag);
public record struct TaggedEncodedChunkAligned(string Text, float[] Vector, string Tag, int Start, int LastStart, int ApproximateEnd, string OriginalText);
public record struct TaggedChunk(string               Text, string  Tag);
public record struct TaggedChunkAligned(string        Text, string  Tag, int Start, int LastStart, int ApproximateEnd, string OriginalText);
public interface ISentenceEncoder
{
    public int MaxChunkLength { get; }

    public TokenizerBase Tokenizer { get; }

    public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default);

    public EncodedChunk[] ChunkAndEncode(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
    {
        if (chunkLength <= 0 || chunkLength > MaxChunkLength)
        {
            chunkLength = MaxChunkLength;
        }

        if (chunkOverlap < 0 || chunkOverlap > chunkLength)
        {
            chunkOverlap = chunkLength / 5;
        }

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
                    var oneVector = Encode(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new EncodedChunk(oneChunk[0], oneVector[0]);

                    if (reportProgress is object && (i % 128 == 0))
                    {
                        reportProgress(((float)i / chunks.Count) * 0.5f + 0.5f);
                    }
                }
            }
            else
            {
                var vectors = Encode(chunks.ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new EncodedChunk(chunks[i], vectors[i]);

                    if (reportProgress is object && (i % 128 == 0))
                    {
                        reportProgress(((float)i / chunks.Count) * 0.5f + 0.5f);
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

    public EncodedChunkAligned[] ChunkAndEncodeAligned(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
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

        try
        {
            if (sequentially)
            {
                var oneChunk = new string[1];

                for (int i = 0; i < chunks.Count; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    oneChunk[0] = chunks[i].Value;
                    var oneVector = Encode(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new EncodedChunkAligned(oneChunk[0], oneVector[0], chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);

                    if (reportProgress is object && (i % 128 == 0))
                    {
                        reportProgress(((float)i / chunks.Count) * 0.5f + 0.5f);
                    }
                }
            }
            else
            {
                var vectors = Encode(chunks.Select(v => v.Value).ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new EncodedChunkAligned(chunks[i].Value, vectors[i], chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);

                    if (reportProgress is object && (i % 128 == 0))
                    {
                        reportProgress(((float)i / chunks.Count) * 0.5f + 0.5f);
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

    public TaggedEncodedChunk[] ChunkAndEncodeTagged(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
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

        try
        {
            if (sequentially)
            {
                var oneChunk = new string[1];

                for (int i = 0; i < chunks.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    oneChunk[0] = chunks[i].Text;
                    var oneVector = Encode(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new TaggedEncodedChunk(chunks[i].Text, oneVector[0], chunks[i].Tag);

                    if (reportProgress is object && (i % 128 == 0))
                    {
                        reportProgress(((float)i / chunks.Length) * 0.5f + 0.5f);
                    }
                }
            }
            else
            {
                var vectors = Encode(chunks.Select(c => c.Text).ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new TaggedEncodedChunk(chunks[i].Text, vectors[i], chunks[i].Tag);

                    if (reportProgress is object && (i % 128 == 0))
                    {
                        reportProgress(((float)i / chunks.Length) * 0.5f + 0.5f);
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

    public TaggedEncodedChunkAligned[] ChunkAndEncodeTaggedAligned(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
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

        try
        {
            if (sequentially)
            {
                var oneChunk = new string[1];

                for (int i = 0; i < chunks.Length; i++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    oneChunk[0] = chunks[i].Text;
                    var oneVector = Encode(oneChunk, cancellationToken: cancellationToken);
                    encodedChunks[i] = new TaggedEncodedChunkAligned(chunks[i].Text, oneVector[0], chunks[i].Tag, chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);

                    if (reportProgress is object && (i % 128 == 0))
                    {
                        reportProgress(((float)i / chunks.Length) * 0.5f + 0.5f);
                    }
                }
            }
            else
            {
                var vectors = Encode(chunks.Select(c => c.Text).ToArray(), cancellationToken: cancellationToken);

                for (int i = 0; i < encodedChunks.Length; i++)
                {
                    encodedChunks[i] = new TaggedEncodedChunkAligned(chunks[i].Text, vectors[i], chunks[i].Tag, chunks[i].Start, chunks[i].LastStart, chunks[i].ApproximateEnd, text);

                    if (reportProgress is object && (i % 128 == 0))
                    {
                        reportProgress(((float)i / chunks.Length) * 0.5f + 0.5f);
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

    public List<string> ChunkTokens(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null)
    {
        reportProgress?.Invoke(0.001f);
        checked //Ensure the max text substring length computed is not overflowing
        {
            var tokenized = Tokenizer.TokenizeRaw(maxChunks != int.MaxValue ? text.AsSpan(0, Math.Min(chunkLength* maxChunks *Tokenizer.ApproxCharToTokenRatio, text.Length)) : text);
            reportProgress?.Invoke(0.002f);
            return MergeTokenSplits(tokenized, chunkLength, chunkOverlap, maxChunks, reportProgress);
        }
    }

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

                        if (reportProgress is object && (index % 128 == 0))
                        {
                            reportProgress((float)docs.Count / maxChunks);
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

            if (reportProgress is object && (index % 128 == 0))
            {
                reportProgress((float)index / splits.Count);
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