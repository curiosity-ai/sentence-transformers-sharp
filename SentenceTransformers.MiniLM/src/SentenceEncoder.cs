using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static SentenceTransformers.MiniLM.DenseTensorHelpers;
using SentenceTransformers;
using UID;

namespace SentenceTransformers.MiniLM;

/// <summary>
/// Sentence encoder for the all-MiniLM-L6-v2 model. Loads the embedded ONNX model, applies WordPiece
/// tokenization through <see cref="MiniLMTokenizer"/>, runs the ONNX inference session, and returns
/// 384-dimensional L2-normalized embeddings produced by mean-pooling the token outputs with the
/// attention mask. Implements <see cref="ISentenceEncoder"/>, so the shared chunking helpers
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

    /// <summary>
    /// Maximum number of tokens the model can process per call.
    /// MiniLM-L6-v2 is trained with a 256-token context window despite the often-quoted 512;
    /// see https://stackoverflow.com/questions/75901231/max-seq-length-for-transformer-sentence-bert.
    /// </summary>
    public static int GetMaxChunkLength() => 256;

    /// <inheritdoc cref="GetMaxChunkLength"/>
    public        int MaxChunkLength      => GetMaxChunkLength();

    /// <summary>Creates a new encoder, loading the embedded ONNX model and the MiniLM vocabulary.</summary>
    /// <param name="sessionOptions">Optional ONNX runtime session options. A new default instance is used when null.</param>
    public SentenceEncoder(SessionOptions sessionOptions = null)
    {
        _sessionOptions = sessionOptions ?? new SessionOptions();
        _session        = new InferenceSession(ResourceLoader.GetResource(typeof(SentenceEncoder).Assembly, "model.onnx"), _sessionOptions);
        Tokenizer       = new MiniLMTokenizer();
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
    /// Encodes a batch of sentences to 384-dimensional embeddings. Each sentence is tokenized and
    /// padded to the longest sequence in the batch (truncated at <see cref="MaxChunkLength"/>),
    /// then mean-pooled with the attention mask and L2-normalized.
    /// </summary>
    /// <param name="sentences">Texts to embed. Each entry produces one vector in the result.</param>
    /// <param name="cancellationToken">Token used to terminate the ONNX run early.</param>
    /// <returns>Array of <c>float[384]</c> vectors, one per input sentence in the same order.</returns>
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
            using var output = _session.Run(input, _outputNames, runOptions);

            cancellationToken.ThrowIfCancellationRequested();

            var output_pooled            = MeanPooling((DenseTensor<float>)output.First().Value, encoded);
            var output_pooled_normalized = Normalize(output_pooled);

            const int embDim = 384;

            var outputFlatten = new float[sentences.Length][];

            for (int s = 0; s < sentences.Length; s++)
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
        catch (Microsoft.ML.OnnxRuntime.OnnxRuntimeException e)
        {
            cancellationToken.ThrowIfCancellationRequested();
            throw;
        }
    }
}