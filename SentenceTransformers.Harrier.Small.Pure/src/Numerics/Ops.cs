using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using SentenceTransformers;

namespace SentenceTransformers.Harrier.Small.Pure.Numerics;

/// <summary>
/// The small set of dense-tensor operators needed to run the Gemma3 forward pass, implemented on top
/// of <see cref="TensorPrimitives"/> (pure-managed, SIMD-accelerated) with multi-threading over the
/// outer dimension. Everything is row-major <see cref="float"/>; there is no tensor object model -
/// callers pass flat arrays plus shapes, which keeps allocations and indirection to a minimum.
/// </summary>
internal static class Ops
{
    /// <summary>Threshold (in output rows) below which a matmul runs single-threaded; spinning up
    /// the thread pool for tiny GEMMs costs more than it saves.</summary>
    private const int ParallelThreshold = 32;

    /// <summary>
    /// Row-major linear projection without bias: <c>y[s, o] = sum_i x[s, i] * w[o, i]</c>.
    /// This matches PyTorch's <c>nn.Linear</c> weight layout <c>[outDim, inDim]</c>, so each output
    /// channel's weights are contiguous and feed straight into a SIMD dot product. Parallelized over
    /// the output channels via <see cref="ParallelExecution.ForAsync"/> (which dispatches with
    /// <c>Parallel.ForAsync</c> when <see cref="ParallelExecution.Enabled"/>, else runs single-threaded)
    /// for anything but the smallest matrices.
    /// </summary>
    /// <param name="x">Input activations, length <c>seq * inDim</c>.</param>
    /// <param name="w">Weight matrix, length <c>outDim * inDim</c>, row-major <c>[outDim, inDim]</c>.</param>
    /// <param name="y">Output buffer, length <c>seq * outDim</c>.</param>
    public static Task LinearAsync(float[] x, float[] w, float[] y, int seq, int inDim, int outDim, CancellationToken ct = default)
    {
        if (outDim < ParallelThreshold)
        {
            for (int o = 0; o < outDim; o++)
            {
                LinearColumn(x, w, y, seq, inDim, outDim, o);
            }
            return Task.CompletedTask;
        }
        return ParallelExecution.ForAsync(0, outDim, ct, (o, _) =>
        {
            LinearColumn(x, w, y, seq, inDim, outDim, o);
            return ValueTask.CompletedTask;
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void LinearColumn(float[] x, float[] w, float[] y, int seq, int inDim, int outDim, int o)
    {
        // TensorPrimitives.Dot is already FMA-optimized with internal unrolling, so a hand-rolled
        // register-blocked kernel does not beat it for float32 here - keep the simple dot per (o, s).
        var wRow = new ReadOnlySpan<float>(w, o * inDim, inDim);
        for (int s = 0; s < seq; s++)
        {
            y[s * outDim + o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(x, s * inDim, inDim), wRow);
        }
    }

    /// <summary>
    /// Gemma RMSNorm over the last dimension: <c>y = x / sqrt(mean(x^2) + eps) * (1 + weight)</c>.
    /// The reference computes this in float32 and uses the <c>(1 + weight)</c> formulation; the
    /// reciprocal-sqrt statistics are computed per row.
    /// </summary>
    public static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, Span<float> y, int rows, int dim, float eps)
    {
        for (int r = 0; r < rows; r++)
        {
            var xr = x.Slice(r * dim, dim);
            var yr = y.Slice(r * dim, dim);
            float sumSq = TensorPrimitives.SumOfSquares(xr);
            float invRms = 1f / MathF.Sqrt(sumSq / dim + eps);
            for (int i = 0; i < dim; i++)
            {
                yr[i] = xr[i] * invRms * (1f + weight[i]);
            }
        }
    }

    /// <summary>In-place L2 normalization of a single vector to unit length.</summary>
    public static void L2NormalizeInPlace(Span<float> v, float eps = 1e-12f)
    {
        float norm = MathF.Sqrt(TensorPrimitives.SumOfSquares(v));
        float inv = 1f / MathF.Max(norm, eps);
        TensorPrimitives.Multiply(v, inv, v);
    }

    /// <summary>
    /// Gemma's GeGLU feed-forward combination: <c>out = gelu_tanh(gate) * up</c>, written into
    /// <paramref name="gate"/> in place. <paramref name="gate"/> and <paramref name="up"/> have the
    /// same length.
    /// </summary>
    public static void GeGluInPlace(Span<float> gate, ReadOnlySpan<float> up)
    {
        const float c = 0.7978845608028654f; // sqrt(2/pi)
        for (int i = 0; i < gate.Length; i++)
        {
            float v = gate[i];
            float inner = c * (v + 0.044715f * v * v * v);
            float gelu = 0.5f * v * (1f + MathF.Tanh(inner));
            gate[i] = gelu * up[i];
        }
    }

    /// <summary>Numerically-stable softmax over the first <paramref name="length"/> entries of
    /// <paramref name="scores"/>, in place.</summary>
    public static void SoftmaxInPlace(Span<float> scores, int length)
    {
        var s = scores.Slice(0, length);
        float max = TensorPrimitives.Max(s);
        TensorPrimitives.Subtract(s, max, s);
        TensorPrimitives.Exp(s, s);
        float sum = TensorPrimitives.Sum(s);
        TensorPrimitives.Multiply(s, 1f / sum, s);
    }
}
