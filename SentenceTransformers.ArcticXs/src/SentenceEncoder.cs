using BERTTokenizers;
using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static SentenceTransformers.ArcticXs.DenseTensorHelpers;
using System.Linq;
using System.Collections.Generic;
using SentenceTransformers;

namespace SentenceTransformers.ArcticXs;

public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
{
    private readonly SessionOptions   _sessionOptions;
    private readonly InferenceSession _session;
    private readonly TokenizerBase    _tokenizer;
    private readonly string[]         _outputNames;

    private const int MAX_TOKEN_LENGTH = 512;

    public SentenceEncoder(SessionOptions sessionOptions = null)
    {
        _sessionOptions = sessionOptions ?? new SessionOptions();
        _session        = new InferenceSession(ResourceLoader.GetResource(typeof(SentenceEncoder).Assembly, "model.onnx"), _sessionOptions);
        _tokenizer      = new ArcticTokenizer();
        _tokenizer.SetMaxTokens(MAX_TOKEN_LENGTH);
        _outputNames = _session.OutputMetadata.Keys.ToArray();
    }

    public EncodedChunk[] ChunkAndEncode(string text, int chunkLength = MAX_TOKEN_LENGTH, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, CancellationToken cancellationToken = default)
    {
        if (chunkLength <= 0 || chunkLength > MAX_TOKEN_LENGTH) throw new ArgumentException("ArcticXs only supports a chunk length up to " + MAX_TOKEN_LENGTH);
        return ((ISentenceEncoder)this).ChunkAndEncode(text, chunkLength: chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, cancellationToken);
    }

    public TaggedEncodedChunk[] ChunkAndEncodeTagged(string text, Func<string, TaggedChunk> stripTags, int chunkLength = MAX_TOKEN_LENGTH, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, CancellationToken cancellationToken = default)
    {
        if (chunkLength <= 0 || chunkLength > MAX_TOKEN_LENGTH) throw new ArgumentException("ArcticXs only supports a chunk length up to " + MAX_TOKEN_LENGTH);

        return ((ISentenceEncoder)this).ChunkAndEncodeTagged(text, stripTags, chunkLength: chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, cancellationToken);
    }

    public void Dispose()
    {
        _sessionOptions.Dispose();
        _session.Dispose();
    }

    public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default)
    {
        var numSentences = sentences.Length;

        var encoded    = _tokenizer.Encode(sentences);
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
            NamedOnnxValue.CreateFromTensor("input_ids",      new DenseTensor<long>(flattenIDs,           dimensions)),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(flattenAttentionMask, dimensions)),
            NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(flattenTokenTypeIds,  dimensions))
        };

        using var runOptions   = new RunOptions();
        using var registration = cancellationToken.Register(() => runOptions.Terminate = true);

        using var output = _session.Run(input, _outputNames, runOptions);

        var outputValue = (DenseTensor<float>)output.First().Value;

        cancellationToken.ThrowIfCancellationRequested();

        return Normalize(outputValue);
    }
}