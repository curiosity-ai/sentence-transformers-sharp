using BERTTokenizers;
using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static MiniLM.DenseTensorHelpers;
using System.Linq;
using System.Collections.Generic;

namespace MiniLM;

public record struct EncodedChunk(string Text, float[] Vector);
public record struct TaggedEncodedChunk(string Text, float[] Vector, string Tag);
public record struct TaggedChunk(string Text, string Tag);

public sealed class SentenceEncoder : IDisposable
{
    private readonly SessionOptions _sessionOptions;
    private readonly InferenceSession _session;
    private readonly TokenizerBase _tokenizer;
    private readonly string[] _outputNames;

    public SentenceEncoder(SessionOptions sessionOptions = null)
    {
        _sessionOptions = sessionOptions ?? new SessionOptions();
        _session = new InferenceSession(ResourceLoader.GetResource(typeof(SentenceEncoder).Assembly, "model.onnx"), _sessionOptions);
        _tokenizer = new MiniLMTokenizer();
        _outputNames = _session.OutputMetadata.Keys.ToArray();
    }

    public void Dispose()
    {
        _sessionOptions.Dispose();
        _session.Dispose();
    }

    public EncodedChunk[] ChunkAndEncode(string text, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = false, CancellationToken cancellationToken = default)
    {
        var chunks = ChunkText(text, ' ', chunkLength, chunkOverlap);

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

    public TaggedEncodedChunk[] ChunkAndEncodeTagged(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100)
    {
        var chunks = ChunkText(text, ' ', chunkLength, chunkOverlap)
                       .Select(chunk => stripTags(chunk))
                       .ToArray();

        var vectors = Encode(chunks.Select(c => c.Text).ToArray());

        return chunks.Zip(vectors, (c, v) => new TaggedEncodedChunk(c.Text, v, c.Tag)).ToArray();
    }

    public static List<string> ChunkText(string text, char separator = ' ', int chunkLength = 500, int chunkOverlap = 100)
    {
        return MergeSplits(text.Split(new char[] { '\n', '\r', ' ' }, StringSplitOptions.RemoveEmptyEntries), separator, chunkLength, chunkOverlap);
    }

    private static List<string> MergeSplits(IEnumerable<string> splits, char separator, int chunkLength, int chunkOverlap)
    {
        const int separatorLength = 1;
        var docs = new List<string>();
        var currentDoc = new List<string>();
        int total = 0;
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
        }

        string final_doc = string.Join(separator, currentDoc);

        if (!string.IsNullOrWhiteSpace(final_doc))
        {
            docs.Add(final_doc);
        }

        return docs;
    }

    public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default)
    {
        var numSentences = sentences.Length;

        var encoded = _tokenizer.Encode(sentences);
        var tokenCount = encoded.First().InputIds.Length;

        long[] flattenIDs           = new long[encoded.Sum(s => s.InputIds.Length)];
        long[] flattenAttentionMask = new long[encoded.Sum(s => s.AttentionMask.Length)];
        long[] flattenTokenTypeIds  = new long[encoded.Sum(s => s.TokenTypeIds.Length)];
        
        var flattenIDsSpan           = flattenIDs.AsSpan();
        var flattenAttentionMaskSpan = flattenAttentionMask.AsSpan();
        var flattenTokenTypeIdsSpan  = flattenTokenTypeIds.AsSpan();

        foreach (var (InputIds, TokenTypeIds, AttentionMask) in encoded)
        {
            InputIds.AsSpan().CopyTo(flattenIDsSpan);
            flattenIDsSpan = flattenIDsSpan.Slice(InputIds.Length);
            
            AttentionMask.AsSpan().CopyTo(flattenAttentionMaskSpan);
            flattenAttentionMaskSpan = flattenAttentionMaskSpan.Slice(AttentionMask.Length);
            
            TokenTypeIds.AsSpan().CopyTo(flattenTokenTypeIdsSpan);
            flattenTokenTypeIdsSpan = flattenTokenTypeIdsSpan.Slice(TokenTypeIds.Length);
        }

        var dimensions = new[] { numSentences, tokenCount };

        var input = new NamedOnnxValue[3]
        {
            NamedOnnxValue.CreateFromTensor("input_ids",      new DenseTensor<long>(flattenIDs,          dimensions)),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(flattenAttentionMask,dimensions)),
            NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(flattenTokenTypeIds, dimensions))
        };

        using var runOptions = new RunOptions();
        using var registration = cancellationToken.Register(() => runOptions.Terminate = true);

        using var output = _session.Run(input, _outputNames, runOptions);
        
        cancellationToken.ThrowIfCancellationRequested();

        var output_pooled = MeanPooling((DenseTensor<float>)output.First().Value, encoded);
        var output_pooled_normalized = Normalize(output_pooled);
        
        const int embDim = 384;

        var outputFlatten = new float[sentences.Length][];

        for(int s = 0; s < sentences.Length; s++)
        {
            var emb = new float[embDim];
            outputFlatten[s] = emb;

            for (int i = 0; i < embDim; i++)
            {
                emb[i] = output_pooled_normalized[s, i];
            }
        }

        return outputFlatten;
    }
}