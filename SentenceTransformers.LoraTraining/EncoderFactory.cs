using SentenceTransformers;
using SentenceTransformers.Bert.Pure;
using SentenceTransformers.Bert.Pure.Model;

namespace SentenceTransformers.LoraTraining;

/// <summary>
/// Builds the pure-C# BERT encoders that support real weight-space LoRA fine-tuning. Only the
/// BertModel-family encoders qualify (they run a full forward <b>and backward</b> pass in managed code);
/// the ONNX-only models (Qwen3, Harrier) are inference-only and are not trainable here.
///
/// <para>The full-precision weights are read directly from the fp32 ONNX graphs already embedded in the
/// MiniLM / ArcticXs packages — no download and no ONNX Runtime — so training is fully self-contained.</para>
/// </summary>
internal static class EncoderFactory
{
    public static readonly string[] Names = { "minilm", "arctic" };

    private static (BertConfig cfg, int maxTokens, System.Reflection.Assembly asm) Spec(string name) =>
        (name ?? "").Trim().ToLowerInvariant() switch
        {
            "minilm"              => (BertConfig.MiniLM,   256, typeof(SentenceTransformers.MiniLM.SentenceEncoder).Assembly),
            "arctic" or "arcticxs"=> (BertConfig.ArcticXs, 512, typeof(SentenceTransformers.ArcticXs.SentenceEncoder).Assembly),
            _ => throw new ArgumentException($"Unknown or unsupported model '{name}'. Trainable models: {string.Join(", ", Names)}."),
        };

    /// <summary>Creates the pure BERT encoder for <paramref name="name"/>, optionally with an adapter applied.</summary>
    public static SentenceEncoder Create(string name, LoraAdapter adapter = null)
    {
        var (cfg, maxTokens, asm) = Spec(name);
        var onnx = ResourceLoader.GetResource(asm, "model.onnx"); // fp32 weights, extracted with no ORT
        return SentenceEncoder.LoadFromOnnx(onnx, cfg, maxTokens, adapter);
    }
}
