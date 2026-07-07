using BERTTokenizers.Base;

namespace SentenceTransformers.Training;

/// <summary>
/// Wraps any <see cref="ISentenceEncoder"/> with a trained <see cref="LoraAdapter"/>, so a fine-tuned
/// model is a drop-in replacement for the base one: it exposes the same <see cref="ISentenceEncoder"/>
/// surface (including all the shared chunking helpers) and simply applies the adapter to every vector
/// the base encoder produces.
///
/// <code>
/// using var baseEncoder = new SentenceTransformers.MiniLM.SentenceEncoder();
/// var adapter           = LoraAdapter.Load("support-faq.lora");
/// using var encoder     = new AdaptedSentenceEncoder(baseEncoder, adapter);
///
/// float[][] vectors = await encoder.EncodeAsync(new[] { "how do I reset my password" });
/// </code>
/// </summary>
public sealed class AdaptedSentenceEncoder : ISentenceEncoder
{
    private readonly ISentenceEncoder _base;
    private readonly LoraAdapter      _adapter;
    private readonly bool             _ownsBase;

    /// <summary>The frozen base encoder being adapted.</summary>
    public ISentenceEncoder BaseEncoder => _base;

    /// <summary>The adapter applied on top of the base encoder's embeddings.</summary>
    public LoraAdapter Adapter => _adapter;

    /// <inheritdoc/>
    public int MaxChunkLength => _base.MaxChunkLength;

    /// <inheritdoc/>
    public TokenizerBase Tokenizer => _base.Tokenizer;

    /// <summary>Creates an adapted encoder.</summary>
    /// <param name="baseEncoder">The frozen encoder to wrap.</param>
    /// <param name="adapter">The trained adapter. Its <see cref="LoraAdapter.Dimension"/> must match the base encoder's output width.</param>
    /// <param name="disposeBaseEncoder">When true (default), disposing this encoder also disposes the wrapped base encoder.</param>
    public AdaptedSentenceEncoder(ISentenceEncoder baseEncoder, LoraAdapter adapter, bool disposeBaseEncoder = true)
    {
        _base     = baseEncoder ?? throw new ArgumentNullException(nameof(baseEncoder));
        _adapter  = adapter     ?? throw new ArgumentNullException(nameof(adapter));
        _ownsBase = disposeBaseEncoder;
    }

    /// <inheritdoc/>
    public async Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
    {
        var baseVectors = await _base.EncodeAsync(sentences, cancellationToken);

        var adapted = new float[baseVectors.Length][];
        for (int i = 0; i < baseVectors.Length; i++)
        {
            var v = baseVectors[i];
            if (v is null)
            {
                adapted[i] = null;
                continue;
            }

            if (v.Length != _adapter.Dimension)
            {
                throw new InvalidOperationException(
                    $"Adapter dimension ({_adapter.Dimension}) does not match the base encoder's output ({v.Length}). " +
                    "The adapter must be trained on the same model it is applied to.");
            }

            adapted[i] = _adapter.Apply(v);
        }

        return adapted;
    }

    /// <inheritdoc/>
    public void Dispose()
    {
        if (_ownsBase)
        {
            _base.Dispose();
        }
    }
}
