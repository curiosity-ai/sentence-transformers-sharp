using BERTTokenizers.Base;
using SentenceTransformers;

namespace SentenceTransformers.Tests.Support;

/// <summary>
/// In-memory <see cref="ISentenceEncoder"/> that captures every batch passed to
/// <see cref="EncodeAsync"/> and returns deterministic dummy vectors. Lets us exercise the
/// chunk+encode pipeline (including marker injection / stripping flows) without loading an ONNX
/// model or downloading anything.
/// </summary>
internal sealed class FakeSentenceEncoder : ISentenceEncoder
{
    private readonly int _dim;

    public FakeSentenceEncoder(TokenizerBase tokenizer, int maxChunkLength, int dim = 4)
    {
        Tokenizer = tokenizer;
        MaxChunkLength = maxChunkLength;
        _dim = dim;
    }

    public int MaxChunkLength { get; }

    public TokenizerBase Tokenizer { get; }

    public List<string[]> ReceivedBatches { get; } = new();

    public int CallCount => ReceivedBatches.Count;

    public Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ReceivedBatches.Add(sentences.ToArray());

        var vectors = new float[sentences.Length][];
        for (int i = 0; i < sentences.Length; i++)
        {
            var v = new float[_dim];
            var s = sentences[i] ?? string.Empty;
            v[0] = s.Length;
            v[1] = s.Length == 0 ? 0 : s[0];
            v[2] = s.GetHashCode();
            v[3] = i;
            vectors[i] = v;
        }
        return Task.FromResult(vectors);
    }

    public void Dispose() { }
}
