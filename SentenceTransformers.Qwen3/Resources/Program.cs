using SentenceTransformers.Qwen3;
using System.Diagnostics;
using System.Text;

class Program
{
    static async Task Main()
    {
        using var enc = await SentenceEncoder.CreateAsync();
        
        var sentences = new[]
        {
            "query: What is the capital of France?",
            "passage: Paris is the capital and most populous city of France."
        };

        // Warmup (JIT + ORT initialization effects)
        var warm = enc.Encode(sentences);
        Console.WriteLine($"Warmup OK: {warm.Length} x {warm[0].Length}");
        Console.WriteLine($"First 5 dims: {string.Join(", ", warm[0].Take(5).Select(v => v.ToString("0.0000")))}");

        // Simple throughput test
        int iters = 50;
        var sw = Stopwatch.StartNew();

        long totalUtf8Bytes = 0;
        for (int i = 0; i < iters; i++)
        {
            var embs = enc.Encode(sentences);
            totalUtf8Bytes += sentences.Sum(s => Encoding.UTF8.GetByteCount(s));
        }

        sw.Stop();

        double seconds = sw.Elapsed.TotalSeconds;
        double bytesPerSec = totalUtf8Bytes / seconds;

        // Output bytes/sec (float embeddings)
        int embDim = warm[0].Length;
        long totalOutputBytes = (long)iters * sentences.Length * embDim * sizeof(float);
        double outBytesPerSec = totalOutputBytes / seconds;

        Console.WriteLine($"Time: {sw.Elapsed.TotalMilliseconds:0.0} ms for {iters} iterations");
        Console.WriteLine($"Input text throughput (UTF-8 bytes/s): {bytesPerSec:0}");
        Console.WriteLine($"Output tensor throughput (bytes/s):   {outBytesPerSec:0}");
    }
}