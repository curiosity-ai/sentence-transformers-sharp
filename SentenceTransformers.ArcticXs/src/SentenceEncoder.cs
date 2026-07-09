using BERTTokenizers;
using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static SentenceTransformers.ArcticXs.DenseTensorHelpers;
using System.Linq;
using System.Collections.Generic;
using SentenceTransformers;
using UID;

namespace SentenceTransformers.ArcticXs;

/// <summary>
/// Sentence encoder for the Snowflake arctic-embed-xs model. Loads the embedded ONNX model, tokenizes
/// inputs through <see cref="ArcticTokenizer"/>, runs ONNX inference, and returns L2-normalized
/// embeddings produced directly from the model's pooled output (no additional pooling is applied).
/// Implements <see cref="ISentenceEncoder"/>, so the shared chunking helpers
/// (<c>ChunkTokens</c>, <c>ChunkAndEncodeAsync</c>, <c>ChunkAndEncodeTaggedAsync</c>, and their aligned
/// counterparts) are available on instances cast to that interface.
/// </summary>
public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
{
    private readonly SessionOptions   _sessionOptions;
    private readonly InferenceSession _session;

    /// <summary>WordPiece tokenizer used to convert raw text into model input ids/attention/token-type tensors.</summary>
    public           TokenizerBase    Tokenizer { get; }
    private readonly string[]         _outputNames;

    /// <summary>LRU cache of the last 16 encoded vectors, keyed by each input's <c>Hash128()</c> UID.</summary>
    private readonly VectorCache       _vectorCache = new(16);

    /// <summary>Maximum number of tokens the model can process per call (512 for arctic-embed-xs).</summary>
    public static int GetMaxChunkLength() => 512;

    /// <inheritdoc cref="GetMaxChunkLength"/>
    public        int MaxChunkLength      => GetMaxChunkLength();

    /// <summary>Creates a new encoder, loading the embedded ONNX model and the arctic-embed vocabulary.</summary>
    /// <param name="sessionOptions">Optional ONNX runtime session options. A new default instance is used when null.</param>
    public SentenceEncoder(SessionOptions sessionOptions = null)
    {
        _sessionOptions = sessionOptions ?? new SessionOptions();
        _session        = new InferenceSession(ResourceLoader.GetResource(typeof(SentenceEncoder).Assembly, "model.onnx"), _sessionOptions);
        Tokenizer       = new ArcticTokenizer();
        Tokenizer.SetMaxTokens(MaxChunkLength);
        _outputNames = _session.OutputMetadata.Keys.ToArray();
    }

    /// <summary>Disposes the ONNX session and its options.</summary>
    public void Dispose()
    {
        _sessionOptions.Dispose();
        _session.Dispose();
    }

    /// <summary>
    /// Encodes a batch of sentences. Each sentence is tokenized and padded to the longest sequence
    /// in the batch (truncated at <see cref="MaxChunkLength"/>), the model is run, and its output
    /// vectors are L2-normalized.
    /// </summary>
    /// <param name="sentences">Texts to embed. Each entry produces one vector in the result.</param>
    /// <param name="cancellationToken">Token used to terminate the ONNX run early.</param>
    /// <returns>Array of embedding vectors, one per input sentence in the same order.</returns>
    public async Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
    {
        if (sentences is null || sentences.Length == 0)
        {
            return Array.Empty<float[]>();
        }

        var       results = new float[sentences.Length][];
        var       keys    = new UID128[sentences.Length];
        List<int> misses  = null;

        // Serve any inputs whose Hash128() UID is already cached, and collect the cache misses.
        for (int i = 0; i < sentences.Length; i++)
        {
            keys[i] = (sentences[i] ?? string.Empty).Hash128();

            if (_vectorCache.TryGet(keys[i], out var cached))
            {
                results[i] = cached;
            }
            else
            {
                (misses ??= new List<int>()).Add(i);
            }
        }

        if (misses is null)
        {
            return results;
        }

        var toEncode = new string[misses.Count];
        for (int i = 0; i < misses.Count; i++)
        {
            toEncode[i] = sentences[misses[i]];
        }

        var encoded = await EncodeCoreAsync(toEncode, cancellationToken);

        // Place each freshly encoded vector back in source order and add it to the cache.
        for (int i = 0; i < misses.Count; i++)
        {
            results[misses[i]] = encoded[i];
            _vectorCache.Set(keys[misses[i]], encoded[i]);
        }

        return results;
    }

    /// <inheritdoc/>
    public async Task<float[]> EncodeAsync(string sentence, CancellationToken cancellationToken = default)
    {
        var vectors = await EncodeAsync(new[] { sentence ?? string.Empty }, cancellationToken);
        return vectors.Length > 0 ? vectors[0] : null;
    }

    private async Task<float[][]> EncodeCoreAsync(string[] sentences, CancellationToken cancellationToken)
    {
        var numSentences = sentences.Length;

        var encoded    = Tokenizer.Encode(sentences);
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

        try
        {
            using var output      = _session.Run(input, _outputNames, runOptions);
            var       outputValue = (DenseTensor<float>)output.First().Value;

            cancellationToken.ThrowIfCancellationRequested();

            return Normalize(outputValue);
        }
        catch (Microsoft.ML.OnnxRuntime.OnnxRuntimeException e)
        {
            cancellationToken.ThrowIfCancellationRequested();
            throw;
        }
    }
}