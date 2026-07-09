namespace SentenceTransformers.Bert.Pure.Model;

/// <summary>How a BERT sentence embedding is pooled from the token hidden states.</summary>
public enum PoolingMode
{
    /// <summary>Mean of all (non-special-token-agnostic) token vectors — used by all-MiniLM-L6-v2.</summary>
    Mean,

    /// <summary>The <c>[CLS]</c> (first) token vector — used by snowflake-arctic-embed-xs.</summary>
    Cls,
}

/// <summary>
/// Architecture hyper-parameters for a HuggingFace <c>BertModel</c> encoder. Defaults match both
/// all-MiniLM-L6-v2 and snowflake-arctic-embed-xs (they are the same shape); only <see cref="Pooling"/>
/// differs between them.
/// </summary>
public sealed class BertConfig
{
    public int VocabSize            { get; init; } = 30522;
    public int HiddenSize           { get; init; } = 384;
    public int NumLayers            { get; init; } = 6;
    public int NumHeads             { get; init; } = 12;
    public int IntermediateSize     { get; init; } = 1536;
    public int MaxPositionEmbeddings{ get; init; } = 512;
    public float LayerNormEps       { get; init; } = 1e-12f;

    /// <summary>Pooling strategy applied to the final token hidden states.</summary>
    public PoolingMode Pooling      { get; init; } = PoolingMode.Mean;

    public int HeadDim => HiddenSize / NumHeads;

    /// <summary>all-MiniLM-L6-v2: 6-layer BERT with mean pooling.</summary>
    public static BertConfig MiniLM => new() { Pooling = PoolingMode.Mean };

    /// <summary>snowflake-arctic-embed-xs: same shape, CLS pooling.</summary>
    public static BertConfig ArcticXs => new() { Pooling = PoolingMode.Cls };
}
