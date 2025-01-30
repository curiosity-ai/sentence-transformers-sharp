using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static SentenceTransformers.MiniLM.DenseTensorHelpers;
using SentenceTransformers;

namespace SentenceTransformers.MiniLM;

public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
{
    private readonly SessionOptions   _sessionOptions;
    private readonly InferenceSession _session;
    public           TokenizerBase    Tokenizer { get; }
    private readonly string[]         _outputNames;

    public static int GetMaxChunkLength() => 256; //The documentation is incorrect for MiniLM - the context window size is 256 - see https://stackoverflow.com/questions/75901231/max-seq-length-for-transformer-sentence-bert
    public        int MaxChunkLength      => GetMaxChunkLength();

    public SentenceEncoder(SessionOptions sessionOptions = null)
    {
        _sessionOptions = sessionOptions ?? new SessionOptions();
        _session        = new InferenceSession(ResourceLoader.GetResource(typeof(SentenceEncoder).Assembly, "model.onnx"), _sessionOptions);
        Tokenizer       = new MiniLMTokenizer();
        Tokenizer.SetMaxTokens(MaxChunkLength);
        _outputNames = _session.OutputMetadata.Keys.ToArray();
    }

    public void Dispose()
    {
        _sessionOptions.Dispose();
        _session.Dispose();
    }

    public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default)
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
}