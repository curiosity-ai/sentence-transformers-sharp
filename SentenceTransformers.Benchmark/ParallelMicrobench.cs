using System.Diagnostics;
using System.Numerics.Tensors;
using SentenceTransformers;

/// <summary>
/// Micro-benchmark that compares <see cref="Parallel.ForAsync"/> (the previous parallelism back-end
/// used by the Pure inference kernels) against <see cref="GlobalThreadPool.ForAsync"/> (the new
/// fixed-worker, contiguous-bucket pool). It runs the exact shape that dominates the Pure forward
/// pass - parallel float dot products over the output channels of a row-major weight matrix - so the
/// results are representative of a single matmul layer.
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
            ParallelForAsyncMatmul(x, w, y, seq, inDim, outDim).GetAwaiter().GetResult();
            GlobalThreadPoolMatmul(x, w, y, seq, inDim, outDim).GetAwaiter().GetResult();
        }

        var sw = Stopwatch.StartNew();
        for (int i = 0; i < iterations; i++)
        {
            ParallelForAsyncMatmul(x, w, y, seq, inDim, outDim).GetAwaiter().GetResult();
        }
        sw.Stop();
        double parMs = sw.Elapsed.TotalMilliseconds / iterations;

        sw.Restart();
        for (int i = 0; i < iterations; i++)
        {
            GlobalThreadPoolMatmul(x, w, y, seq, inDim, outDim).GetAwaiter().GetResult();
        }
        sw.Stop();
        double poolMs = sw.Elapsed.TotalMilliseconds / iterations;

        Console.WriteLine($"[matmul seq={seq} inDim={inDim} outDim={outDim}, iters={iterations}]");
        Console.WriteLine($"  Parallel.ForAsync   : {parMs,8:F3} ms/iter");
        Console.WriteLine($"  GlobalThreadPool    : {poolMs,8:F3} ms/iter   ({parMs / poolMs:F2}x)");
    }

    private static Task ParallelForAsyncMatmul(float[] x, float[] w, float[] y, int seq, int inDim, int outDim)
    {
        return Parallel.ForAsync(0, outDim, (o, _) =>
        {
            LinearColumn(x, w, y, seq, inDim, outDim, o);
            return ValueTask.CompletedTask;
        });
    }

    private static Task GlobalThreadPoolMatmul(float[] x, float[] w, float[] y, int seq, int inDim, int outDim)
    {
        return GlobalThreadPool.ForAsync(0, outDim, (x, w, y, seq, inDim, outDim), static (start, end, st) =>
        {
            for (int o = start; o < end; o++)
            {
                LinearColumn(st.x, st.w, st.y, st.seq, st.inDim, st.outDim, o);
            }
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
