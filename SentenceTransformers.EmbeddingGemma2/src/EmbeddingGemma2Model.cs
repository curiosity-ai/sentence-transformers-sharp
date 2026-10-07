namespace SentenceTransformers.EmbeddingGemma2;

/// <summary>The three published EmbeddingGemma 2 LiteRT-LM bundles. All share the same text tower
/// (tokenizer, embedder and transformer are byte-identical), so text embeddings are identical across
/// them; the larger bundles add the vision and audio encoders.</summary>
public enum EmbeddingGemma2Model
{
    /// <summary><c>embeddinggemma-2-text-270m</c>: text only (~165 MB download).</summary>
    Text270M,

    /// <summary><c>embeddinggemma-2-text-vision-440m</c>: text + images (~388 MB download).</summary>
    TextVision440M,

    /// <summary><c>embeddinggemma-2-740m</c>: text + images + audio (~485 MB download).</summary>
    Multimodal740M,
}

/// <summary>Download locations and capabilities of the <see cref="EmbeddingGemma2Model"/> variants.</summary>
public static class EmbeddingGemma2Models
{
    private const string ModelsBaseUrl = "https://models.curiosity.ai/embeddinggemma-2/";

    /// <summary>Hugging Face repository id the bundle is published under (<c>litert-community/&lt;id&gt;</c>).</summary>
    public static string GetRepository(EmbeddingGemma2Model model) => model switch
    {
        EmbeddingGemma2Model.Text270M => "embeddinggemma-2-text-270m-litert-lm",
        EmbeddingGemma2Model.TextVision440M => "embeddinggemma-2-text-vision-440m-litert-lm",
        EmbeddingGemma2Model.Multimodal740M => "embeddinggemma-2-740m-litert-lm",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>File name of the generic CPU/GPU <c>.litertlm</c> bundle.</summary>
    public static string GetFileName(EmbeddingGemma2Model model) => model switch
    {
        EmbeddingGemma2Model.Text270M => "embeddinggemma-2-text-270m.litertlm",
        EmbeddingGemma2Model.TextVision440M => "embeddinggemma-2-text-vision-440m.litertlm",
        EmbeddingGemma2Model.Multimodal740M => "embeddinggemma-2-740m.litertlm",
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    /// <summary>Default download URL: an unmodified copy of the <c>litert-community</c> Hugging Face bundle
    /// (same bytes, see <see cref="GetFileSize"/>) hosted on <c>models.curiosity.ai</c>.</summary>
    public static string GetDownloadUrl(EmbeddingGemma2Model model) => ModelsBaseUrl + GetFileName(model);

    /// <summary>Exact size in bytes of the published bundle (used to validate cached downloads).</summary>
    public static long GetFileSize(EmbeddingGemma2Model model) => model switch
    {
        EmbeddingGemma2Model.Text270M => 164_626_432,
        EmbeddingGemma2Model.TextVision440M => 387_710_976,
        EmbeddingGemma2Model.Multimodal740M => 484_622_336,
        _ => throw new ArgumentOutOfRangeException(nameof(model)),
    };

    public static bool SupportsImages(EmbeddingGemma2Model model) => model != EmbeddingGemma2Model.Text270M;

    public static bool SupportsAudio(EmbeddingGemma2Model model) => model == EmbeddingGemma2Model.Multimodal740M;

    /// <summary>Default local cache path: <c>%TEMP%/SentenceTransformers.EmbeddingGemma2/&lt;file&gt;</c>.</summary>
    public static string GetDefaultCachePath(EmbeddingGemma2Model model)
        => Path.Combine(Path.GetTempPath(), "SentenceTransformers.EmbeddingGemma2", GetFileName(model));
}
