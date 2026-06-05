namespace SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// Architecture hyper-parameters for harrier-oss-v1-270m, a <c>Gemma3TextModel</c>. Values mirror the
/// upstream <c>config.json</c>. The defaults are hard-coded because this package targets exactly this
/// one checkpoint; nothing here is meant to be a general Gemma3 loader.
/// </summary>
internal sealed class Gemma3Config
{
    public int HiddenSize { get; init; } = 640;
    public int NumLayers { get; init; } = 18;
    public int NumHeads { get; init; } = 4;
    public int NumKvHeads { get; init; } = 1;
    public int HeadDim { get; init; } = 256;
    public int IntermediateSize { get; init; } = 2048;
    public int VocabSize { get; init; } = 262144;
    public float RmsNormEps { get; init; } = 1e-6f;
    public float RopeTheta { get; init; } = 1_000_000f;

    /// <summary>Attention logit scale = <c>query_pre_attn_scalar ** -0.5</c> = 256^-0.5 = 1/16.</summary>
    public float QueryPreAttnScalar { get; init; } = 256f;

    public int MaxPositionEmbeddings { get; init; } = 32768;

    public int QProjOut => NumHeads * HeadDim;   // 1024
    public int KvProjOut => NumKvHeads * HeadDim; // 256
    public float AttentionScale => 1f / MathF.Sqrt(QueryPreAttnScalar);
}
