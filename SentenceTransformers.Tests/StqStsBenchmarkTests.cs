using SentenceTransformers.Harrier.Small.Pure;
using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Quantize;
using SentenceTransformers.Stq;
using SentenceTransformers.Training;

namespace SentenceTransformers.Tests;

/// <summary>
/// Scores a quantized build on a real task instead of only comparing it with the fp32 encoder.
///
/// <para>The other <c>.stq</c> tests answer "does the codec round-trip and does the kernel match the
/// reference decode" - both exact, both fast, both offline. Neither can tell you whether the model
/// still <i>works</i>: a conversion can be perfectly self-consistent and still destroy the
/// embeddings, which is exactly what whole-model ternarization does here (see QUANTIZATION.md §4).
/// This test closes that gap by running the STS Benchmark through the converted model and comparing
/// Spearman correlation against the same checkpoint in fp32.</para>
///
/// <para><b>Opt-in.</b> It needs the ~540 MB safetensors checkpoint and downloads the STS-B splits
/// (~1 MB), so, like <see cref="PureEncoderEndToEndTests"/>, it runs only when asked:</para>
/// <code>
///   HARRIER_STQ_STSB=/path/to/harrier-oss-v1-270m.safetensors dotnet test
/// </code>
/// <para>Optional knobs: <c>HARRIER_STQ_STSB_PAIRS</c> caps how many pairs are scored (default 250,
/// which keeps a run to a couple of minutes per configuration; the full split is 1379) and
/// <c>HARRIER_STQ_STSB_DATA</c> chooses where the dataset is cached.</para>
/// </summary>
public class StqStsBenchmarkTests
{
    private const string WeightsVar = "HARRIER_STQ_STSB";
    private const string PairsVar   = "HARRIER_STQ_STSB_PAIRS";
    private const string DataVar    = "HARRIER_STQ_STSB_DATA";

    /// <summary>
    /// The default conversion (4-bit projections and embedding table) must stay close to fp32 on STS-B.
    ///
    /// <para>The bar is a drop of at most 0.02 Spearman. That is wide enough not to trip on the noise
    /// of a few hundred pairs and far tighter than a broken conversion could sneak through: whole-model
    /// ternarization, which the second case pins, costs roughly ten times that.</para>
    /// </summary>
    [Fact]
    public async Task DefaultConversionMatchesFp32OnStsBenchmark()
    {
        string weights = Environment.GetEnvironmentVariable(WeightsVar);
        if (string.IsNullOrWhiteSpace(weights) || !File.Exists(weights))
        {
            return; // opt-in; xUnit 2.x has no dynamic skip, so return early like the other heavy tests
        }

        var data = await LoadPairsAsync();
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

        float fp32 = await ScoreSafetensorsAsync(weights, Quantization.None, data, parallelOptions);
        float quantized = await ScoreConvertedAsync(weights, data, parallelOptions, options => options);

        Assert.True(quantized >= fp32 - 0.02f,
            $"STS-B Spearman fell from {fp32:F4} (fp32) to {quantized:F4} after the default conversion.");
    }

    /// <summary>
    /// Pins the finding that whole-model ternarization is not usable, so nobody "fixes" the default
    /// back to it on the strength of the file size. If a future quantizer ever makes ternary
    /// projections work post-training, this test fails loudly and should be rewritten - that would be
    /// a genuine result, not a regression.
    /// </summary>
    [Fact]
    public async Task WholeModelTernarizationLosesTheModel()
    {
        string weights = Environment.GetEnvironmentVariable(WeightsVar);
        if (string.IsNullOrWhiteSpace(weights) || !File.Exists(weights))
        {
            return;
        }

        var data = await LoadPairsAsync();
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };

        float fp32 = await ScoreSafetensorsAsync(weights, Quantization.None, data, parallelOptions);
        float ternary = await ScoreConvertedAsync(weights, data, parallelOptions, options => options with
        {
            Band = StqBand.TQ1_0,
            EmbeddingBand = StqBand.TQ1_0,
        });

        Assert.True(ternary < fp32 - 0.05f,
            $"Whole-model ternarization scored {ternary:F4} against fp32's {fp32:F4}. QUANTIZATION.md §4 " +
            "records that it should collapse; if it no longer does, that finding needs revisiting.");
    }

    private static async Task<SentencePairDataset> LoadPairsAsync()
    {
        string dir = Environment.GetEnvironmentVariable(DataVar)
                     ?? Path.Combine(Path.GetTempPath(), "SentenceTransformers.Tests", "stsb");
        await StsbDataset.DownloadAsync(dir);

        var full = StsbDataset.Load(StsbDataset.TestPath(dir));
        int cap = int.TryParse(Environment.GetEnvironmentVariable(PairsVar), out int n) && n > 0 ? n : 250;
        return full.Count <= cap ? full : new SentencePairDataset(full.Pairs.Take(cap));
    }

    private static async Task<float> ScoreSafetensorsAsync(string weights, Quantization quantization,
                                                           SentencePairDataset data, ParallelOptions parallelOptions)
    {
        using var encoder = await SentenceEncoder.LoadAsync(weights, quantization: quantization, parallelOptions: parallelOptions);
        return await EmbeddingEvaluation.SpearmanAsync(encoder, data);
    }

    /// <summary>Converts the checkpoint with the real <see cref="Converter"/> - so the test covers the
    /// tensor policy the CLI actually applies - then scores the result.</summary>
    private static async Task<float> ScoreConvertedAsync(string weights, SentencePairDataset data,
                                                         ParallelOptions parallelOptions,
                                                         Func<ConversionOptions, ConversionOptions> configure)
    {
        string stq = Path.Combine(Path.GetTempPath(), $"stq-stsb-{Guid.NewGuid():N}.stq");
        try
        {
            var options = configure(new ConversionOptions
            {
                InputPath = weights,
                OutputPath = stq,
                SkipErrorReport = true,   // the per-tensor report is covered elsewhere and doubles conversion time
                MaxDegreeOfParallelism = Environment.ProcessorCount,
            });

            int exit = await Converter.RunAsync(options, TextWriter.Null);
            Assert.Equal(0, exit);

            using var encoder = await SentenceEncoder.LoadQuantizedAsync(stq, parallelOptions: parallelOptions);
            return await EmbeddingEvaluation.SpearmanAsync(encoder, data);
        }
        finally
        {
            if (File.Exists(stq)) File.Delete(stq);
        }
    }
}
