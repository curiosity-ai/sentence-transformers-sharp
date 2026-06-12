using System.Diagnostics;
using System.Numerics.Tensors;
using SentenceTransformers;

/// <summary>
/// Micro-benchmark for the Pure inference kernels' fork/join primitive,
/// <see cref="ParallelExecution.ForAsync"/>. It runs the exact shape that dominates the Pure forward
/// pass - a float dot product over the output channels of a row-major weight matrix - single-threaded
/// (<see cref="ParallelExecution.Enabled"/> = <c>false</c>, the default) and multi-threaded
/// (<see cref="ParallelExecution.Enabled"/> = <c>true</c>, which dispatches with
/// <see cref="Parallel.ForAsync(int, int, CancellationToken, Func{int, CancellationToken, ValueTask})"/>),
/// so the speed-up of enabling parallelism for a single matmul layer is visible.
/// </summary>
public static class ParallelMicrobench
{
    public static void Run(int seq, int inDim, int outDim, int iterations, int warmup)
    {
        var rng = new Random(1234);
        var x = new float[seq * inDim];
        var w = new float[outDim * inDim];
        var y = new float[seq * outDim];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        for (int i = 0; i < w.Length; i++)
        {
            w[i] = (float)(rng.NextDouble() * 2 - 1);
        }

        // Warm up both paths.
        for (int i = 0; i < warmup; i++)
        {
            Matmul(x, w, y, seq, inDim, outDim, parallel: false).GetAwaiter().GetResult();
            Matmul(x, w, y, seq, inDim, outDim, parallel: true).GetAwaiter().GetResult();
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            Matmul(x, w, y, seq, inDim, outDim, parallel: false).GetAwaiter().GetResult();
        }
        sw.Stop();
        double seqMs = sw.Elapsed.TotalMilliseconds / iterations;

        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            Matmul(x, w, y, seq, inDim, outDim, parallel: true).GetAwaiter().GetResult();
        }
        sw.Stop();
        double parMs = sw.Elapsed.TotalMilliseconds / iterations;

        Console.WriteLine($"[matmul seq={seq} inDim={inDim} outDim={outDim}, iters={iterations}]");
        Console.WriteLine($"  Single-threaded     : {seqMs,8:F3} ms/iter");
        Console.WriteLine($"  Parallel.ForAsync   : {parMs,8:F3} ms/iter   ({seqMs / parMs:F2}x)");
    }

    private static Task Matmul(float[] x, float[] w, float[] y, int seq, int inDim, int outDim, bool parallel)
    {
        ParallelExecution.Enabled = parallel;
        return ParallelExecution.ForAsync(0, outDim, CancellationToken.None, (o, _) =>
        {
            LinearColumn(x, w, y, seq, inDim, outDim, o);
            return ValueTask.CompletedTask;
        });
    }

    private static void LinearColumn(float[] x, float[] w, float[] y, int seq, int inDim, int outDim, int o)
    {
        var wRow = new ReadOnlySpan<float>(w, o * inDim, inDim);
        for (int s = 0; s < seq; s++)
        {
            y[s * outDim + o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(x, s * inDim, inDim), wRow);
        }
    }
}
