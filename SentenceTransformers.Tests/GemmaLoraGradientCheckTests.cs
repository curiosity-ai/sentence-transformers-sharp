using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Harrier.Small.Pure.Training;
using SentenceTransformers.Training.Autograd;

namespace SentenceTransformers.Tests;

/// <summary>
/// Numerical gradient check for the pure-C# Gemma3 backward pass. A tiny random Gemma3 is run through the
/// autograd graph and the analytic gradients for every trainable LoRA parameter (A, B for each of the
/// seven targeted projections, plus the output-centering bias) must match central finite differences of a
/// scalar loss. This validates the backprop through RoPE, per-head + layer RMSNorm, grouped-query causal
/// attention, the GeGLU MLP and last-token pooling — i.e. that real weight-space LoRA training on Harrier
/// Small has correct gradients.
/// </summary>
public class GemmaLoraGradientCheckTests
{
    [Fact]
    public void AnalyticGradients_MatchFiniteDifferences()
    {
        var cfg = new Gemma3Config
        {
            HiddenSize = 8, NumLayers = 2, NumHeads = 2, NumKvHeads = 1, HeadDim = 4,
            IntermediateSize = 16, VocabSize = 20, QueryPreAttnScalar = 4f,
        };
        var model = Gemma3LoraModel.CreateRandom(cfg, seed: 5);

        var adapter = GemmaLoraAdapter.CreateInitialized(cfg, rank: 3, alpha: 3f, targets: GemmaLoraTargets.All, seed: 11);
        adapter.OutputBias = new float[cfg.HiddenSize];
        var rt = adapter.ToRuntime();

        var prng = new Random(7);
        var parameters = rt.Parameters().ToArray();
        foreach (var t in parameters)
            for (int i = 0; i < t.Data.Length; i++) t.Data[i] = (float)(prng.NextDouble() * 2 - 1) * 0.2f;

        int[] ids = { 1, 5, 3, 2 };
        var target = new float[cfg.HiddenSize];
        for (int i = 0; i < target.Length; i++) target[i] = (float)(prng.NextDouble() * 2 - 1);

        float Loss()
        {
            var g = new Graph();
            var z = model.Forward(g, ids, rt);
            var u = g.L2Normalize(z);
            float l = 0f;
            for (int j = 0; j < u.Cols; j++) l += u.Data[j] * target[j];
            return l;
        }

        foreach (var t in parameters) t.ClearGrad();
        {
            var g = new Graph();
            var z = model.Forward(g, ids, rt);
            var u = g.L2Normalize(z);
            Array.Copy(target, u.Grad, u.Cols);
            g.Backward();
        }

        const float eps = 1e-3f;
        int checkedCount = 0;
        foreach (var t in parameters)
        {
            for (int i = 0; i < t.Data.Length; i += Math.Max(1, t.Data.Length / 5))
            {
                float orig = t.Data[i];
                t.Data[i] = orig + eps; float lp = Loss();
                t.Data[i] = orig - eps; float lm = Loss();
                t.Data[i] = orig;

                float numeric = (lp - lm) / (2f * eps);
                float analytic = t.Grad[i];
                float tol = 3e-2f * (1f + MathF.Abs(numeric) + MathF.Abs(analytic));
                Assert.True(MathF.Abs(numeric - analytic) < tol,
                    $"grad mismatch at param entry {i}: analytic {analytic:0.00000}, numeric {numeric:0.00000}");
                checkedCount++;
            }
        }
        Assert.True(checkedCount > 50, "expected to check many parameter entries");
    }
}
