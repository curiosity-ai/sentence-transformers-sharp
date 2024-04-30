﻿namespace MiniLM.Shared;

public record struct EncodedChunk(string       Text, float[] Vector);
public record struct TaggedEncodedChunk(string Text, float[] Vector, string Tag);
public record struct TaggedChunk(string        Text, string  Tag);
public interface ISentenceEncoder
{
    public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default);

    public EncodedChunk[] ChunkAndEncode(string text, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, CancellationToken cancellationToken = default)
    {
        var chunks = ChunkText(text, ' ', chunkLength, chunkOverlap, maxChunks);

        var encodedChunks = new EncodedChunk[chunks.Count];

        if (sequentially)
        {
            var oneChunk = new string[1];

            for (int i = 0; i < chunks.Count; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                oneChunk[0] = chunks[i];
                var oneVector = Encode(oneChunk, cancellationToken: cancellationToken);
                encodedChunks[i] = new EncodedChunk(oneChunk[0], oneVector[0]);
            }
        }
        else
        {
            var vectors = Encode(chunks.ToArray(), cancellationToken: cancellationToken);

            for (int i = 0; i < encodedChunks.Length; i++)
            {
                encodedChunks[i] = new EncodedChunk(chunks[i], vectors[i]);
            }
        }

        return encodedChunks;
    }

    public TaggedEncodedChunk[] ChunkAndEncodeTagged(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, CancellationToken cancellationToken = default)
    {
        var chunks = ISentenceEncoder.ChunkText(text, ' ', chunkLength, chunkOverlap, maxChunks: maxChunks)
           .Select(chunk => stripTags(chunk))
           .ToArray();

        var encodedChunks = new TaggedEncodedChunk[chunks.Length];

        if (sequentially)
        {
            var oneChunk = new string[1];

            for (int i = 0; i < chunks.Length; i++)
            {
                cancellationToken.ThrowIfCancellationRequested();
                oneChunk[0] = chunks[i].Text;
                var oneVector = Encode(oneChunk, cancellationToken: cancellationToken);
                encodedChunks[i] = new TaggedEncodedChunk(chunks[i].Text, oneVector[0], chunks[i].Tag);
            }
        }
        else
        {
            var vectors = Encode(chunks.Select(c => c.Text).ToArray(), cancellationToken: cancellationToken);

            for (int i = 0; i < encodedChunks.Length; i++)
            {
                encodedChunks[i] = new TaggedEncodedChunk(chunks[i].Text, vectors[i], chunks[i].Tag);
            }
        }

        return encodedChunks;
    }

    public static List<string> ChunkText(string text, char separator = ' ', int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue)
    {
        return MergeSplits(text.Split(new char[] { '\n', '\r', ' ' }, StringSplitOptions.RemoveEmptyEntries), separator, chunkLength, chunkOverlap, maxChunks);
    }

    private static List<string> MergeSplits(IEnumerable<string> splits, char separator, int chunkLength, int chunkOverlap, int maxChunks)
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