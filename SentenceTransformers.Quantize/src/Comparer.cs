using System.Diagnostics;
using SentenceTransformers.Harrier.Small.Pure;
using SentenceTransformers.Harrier.Small.Pure.Model;

namespace SentenceTransformers.Quantize;

/// <summary>
/// Puts the ternary file next to the quantization modes the package already offers, on one sentence
/// set, so "is ternary worth it here?" is answered with numbers rather than by analogy to what
/// ternary does for a 27B model.
///
/// <para>Every mode is scored against the same fp32 baseline: mean and minimum cosine between each
/// sentence's fp32 embedding and its quantized one, and the Spearman correlation between the two
/// pairwise-similarity matrices - the ranking property retrieval actually depends on.</para>
/// </summary>
public static class Comparer
{
    public static async Task<int> RunAsync(string originalPath, string ternaryPath, string[] sentences, TextWriter log, CancellationToken ct = default)
    {
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount, CancellationToken = ct };

        log.WriteLine($"Baseline: fp32 from {Path.GetFileName(originalPath)}, {sentences.Length} sentences");
        using var reference = await SentenceEncoder.LoadAsync(originalPath, quantization: Quantization.None, parallelOptions: parallelOptions).ConfigureAwait(false);
        var refVectors = await reference.EncodeAsync(sentences, parallelOptions).ConfigureAwait(false);
        var refSim = Validator.PairwiseSimilarities(refVectors);

        log.WriteLine();
        log.WriteLine($"  {"mode",-22} {"mean cos",9} {"min cos",9} {"rho",9} {"encode s",9}");

        foreach (var quantization in new[] { Quantization.Int8, Quantization.Int4 })
        {
            ct.ThrowIfCancellationRequested();
            using var encoder = await SentenceEncoder.LoadAsync(originalPath, quantization: quantization, parallelOptions: parallelOptions).ConfigureAwait(false);
            await ReportAsync($"safetensors {quantization}", encoder, refVectors, refSim, sentences, parallelOptions, log).ConfigureAwait(false);
        }

        if (ternaryPath is not null)
        {
            using var ternary = await SentenceEncoder.LoadTernaryAsync(ternaryPath, parallelOptions: parallelOptions).ConfigureAwait(false);
            await ReportAsync($"ternary {Path.GetFileNameWithoutExtension(ternaryPath)}", ternary, refVectors, refSim, sentences, parallelOptions, log).ConfigureAwait(false);
        }

        return 0;
    }

    private static async Task ReportAsync(string label, SentenceEncoder encoder, float[][] refVectors, double[] refSim,
                                          string[] sentences, ParallelOptions parallelOptions, TextWriter log)
    {
        var sw = Stopwatch.StartNew();
        var vectors = await encoder.EncodeAsync(sentences, parallelOptions).ConfigureAwait(false);
        double seconds = sw.Elapsed.TotalSeconds;

        double sum = 0, min = 1;
        for (int i = 0; i < vectors.Length; i++)
        {
            double c = Validator.Cosine(refVectors[i], vectors[i]);
            sum += c;
            min = Math.Min(min, c);
        }
        double rho = Validator.Spearman(refSim, Validator.PairwiseSimilarities(vectors));

        log.WriteLine($"  {label,-22} {sum / vectors.Length,9:F5} {min,9:F5} {rho,9:F5} {seconds,9:F2}");
    }
}
