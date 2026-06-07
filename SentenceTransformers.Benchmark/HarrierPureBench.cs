using System.Diagnostics;
using SentenceTransformers.Harrier.Small.Pure;
using SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// End-to-end timing for the Pure encoder on a short batch. Uses the cached weights at the standard
/// path (see SentenceEncoder.CreateAsync), so it does not redownload on each invocation. Reports
/// ms/iter for each quantization, which is the metric to watch for the parallelism migration: an
/// encode is dominated by the per-layer matmul fork/joins now running on
/// <see cref="SentenceTransformers.GlobalThreadPool"/>.
/// </summary>
public static class HarrierPureBench
{
    public static async Task RunAsync()
    {
        var texts = new[]
        {
            "Curiosity is the wick in the candle of learning.",
            "Quantization shrinks weights and trades a little accuracy for a lot of memory.",
            "Vector search retrieves the documents whose embedding is closest to the query's embedding.",
            "The harrier small model is a decoder-only transformer with grouped-query attention.",
        };

        foreach (var quant in new[] { Quantization.None, Quantization.Int8, Quantization.Int4 })
        {
            using var enc = await SentenceEncoder.CreateAsync(quantization: quant);

            // Warm up (downloads / quantizes happen here).
            await enc.EncodeAsync(texts);
            await enc.EncodeAsync(texts);

            const int iterations = 5;
            var sw = Stopwatch.StartNew();
            for (int i = 0; i < iterations; i++)
            {
                await enc.EncodeAsync(texts);
            }
            sw.Stop();

            double ms = sw.Elapsed.TotalMilliseconds / iterations;
            Console.WriteLine($"[Harrier.Small.Pure {quant}]");
            Console.WriteLine($"  Batch:    {texts.Length}");
            Console.WriteLine($"  Per iter: {ms,8:F1} ms");
            Console.WriteLine($"  Throughput: {1000.0 * texts.Length / ms,6:F2} embeddings/s");
        }
    }
}
