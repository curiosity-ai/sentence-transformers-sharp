using SentenceTransformers.Training;

namespace SentenceTransformers.Tests;

/// <summary>
/// Numerical gradient check for the contrastive trainer: the analytic gradients accumulated by
/// <see cref="LoraTrainer.DebugContrastiveBatch"/> must match central finite differences of the loss
/// with respect to every adapter parameter. This validates the hand-derived backprop through the
/// symmetric InfoNCE loss, the L2 normalization, and the low-rank residual.
/// </summary>
public class LoraGradientCheckTests
{
    [Fact]
    public void AnalyticGradient_MatchesFiniteDifferences()
    {
        const int dim  = 6;
        const int rank = 3;
        const int n    = 4;
        const float temperature = 0.2f;

        var adapter = LoraAdapter.CreateInitialized(dim, rank, seed: 17);

        // Give A and B non-trivial values so we exercise a generic (non-identity) operating point.
        var rng = new Random(9);
        for (int i = 0; i < adapter.A.Length; i++) adapter.A[i] = (float)(rng.NextDouble() - 0.5);
        for (int i = 0; i < adapter.B.Length; i++) adapter.B[i] = (float)(rng.NextDouble() - 0.5);

        var anchors   = new float[n][];
        var positives = new float[n][];
        for (int i = 0; i < n; i++)
        {
            anchors[i]   = Enumerable.Range(0, dim).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            positives[i] = Enumerable.Range(0, dim).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
        }

        var gradA = new float[adapter.A.Length];
        var gradB = new float[adapter.B.Length];
        LoraTrainer.DebugContrastiveBatch(adapter, anchors, positives, temperature, gradA, gradB);

        const float eps = 1e-3f;

        CheckParam(adapter.A, gradA, "A", adapter, anchors, positives, temperature, eps);
        CheckParam(adapter.B, gradB, "B", adapter, anchors, positives, temperature, eps);
    }

    private static void CheckParam(float[] param, float[] analyticGrad, string name, LoraAdapter adapter, float[][] anchors, float[][] positives, float temperature, float eps)
    {
        var dummyA = new float[adapter.A.Length];
        var dummyB = new float[adapter.B.Length];

        for (int i = 0; i < param.Length; i++)
        {
            float original = param[i];

            param[i] = original + eps;
            Array.Clear(dummyA); Array.Clear(dummyB);
            float lossPlus = LoraTrainer.DebugContrastiveBatch(adapter, anchors, positives, temperature, dummyA, dummyB);

            param[i] = original - eps;
            Array.Clear(dummyA); Array.Clear(dummyB);
            float lossMinus = LoraTrainer.DebugContrastiveBatch(adapter, anchors, positives, temperature, dummyA, dummyB);

            param[i] = original;

            float numeric = (lossPlus - lossMinus) / (2f * eps);
            float analytic = analyticGrad[i];

            float tol = 2e-2f * (1f + Math.Abs(analytic) + Math.Abs(numeric));
            Assert.True(Math.Abs(numeric - analytic) < tol,
                $"{name}[{i}] gradient mismatch: analytic {analytic:0.00000}, numeric {numeric:0.00000}");
        }
    }
}
