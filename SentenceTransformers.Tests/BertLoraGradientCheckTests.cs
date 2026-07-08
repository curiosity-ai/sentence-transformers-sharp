using SentenceTransformers.Bert.Pure.Model;
using SentenceTransformers.Training.Autograd;

namespace SentenceTransformers.Tests;

/// <summary>
/// Numerical gradient check for the pure-C# BERT backward pass. A tiny random BERT is run through the
/// autograd graph and the analytic gradients accumulated for every trainable LoRA parameter (A, B for
/// each targeted linear, plus the output-centering bias) must match central finite differences of a
/// scalar loss. This validates the hand-written vector-Jacobian products for LoRA-linear, matmul,
/// LayerNorm, GELU, softmax, attention (slice/concat), pooling and L2 normalization end-to-end — i.e.
/// that real weight-space LoRA training has correct gradients.
/// </summary>
public class BertLoraGradientCheckTests
{
    [Theory]
    [InlineData(PoolingMode.Mean)]
    [InlineData(PoolingMode.Cls)]
    public void AnalyticGradients_MatchFiniteDifferences(PoolingMode pooling)
    {
        var cfg = new BertConfig
        {
            VocabSize = 24, HiddenSize = 12, NumLayers = 2, NumHeads = 3,
            IntermediateSize = 16, MaxPositionEmbeddings = 8, Pooling = pooling,
        };
        var model = BertModel.CreateRandom(cfg, seed: 5);

        var adapter = LoraAdapter.CreateInitialized(cfg, rank: 3, alpha: 3f, targets: LoraTargets.All, seed: 11);
        adapter.OutputBias = new float[cfg.HiddenSize]; // enable the centering bias so it is checked too
        var rt = adapter.ToRuntime();

        // Perturb every trainable tensor to a non-trivial operating point (B initializes to zero, which
        // would make the low-rank gradients vanish and give a degenerate check).
        var prng = new Random(7);
        var parameters = Params(rt).ToArray();
        foreach (var t in parameters)
            for (int i = 0; i < t.Data.Length; i++) t.Data[i] = (float)(prng.NextDouble() * 2 - 1) * 0.2f;

        int[] ids = { 1, 5, 3, 9, 2 };                 // one sequence of length 5
        var target = new float[cfg.HiddenSize];        // fixed direction so loss = <u, target> exercises all dims
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

        // Analytic gradients: forward, seed dL/du = target, backward.
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
            // Check a spread of entries per tensor (checking every entry is slow but unnecessary).
            for (int i = 0; i < t.Data.Length; i += Math.Max(1, t.Data.Length / 7))
            {
                float orig = t.Data[i];
                t.Data[i] = orig + eps; float lp = Loss();
                t.Data[i] = orig - eps; float lm = Loss();
                t.Data[i] = orig;

                float numeric  = (lp - lm) / (2f * eps);
                float analytic = t.Grad[i];
                float tol = 2e-2f * (1f + MathF.Abs(numeric) + MathF.Abs(analytic));
                Assert.True(MathF.Abs(numeric - analytic) < tol,
                    $"grad mismatch at param entry {i}: analytic {analytic:0.00000}, numeric {numeric:0.00000}");
                checkedCount++;
            }
        }
        Assert.True(checkedCount > 50, "expected to check many parameter entries");
    }

    private static IEnumerable<Tensor> Params(LoraRuntime rt)
    {
        foreach (var p in rt.Parameters()) yield return p;
    }
}
