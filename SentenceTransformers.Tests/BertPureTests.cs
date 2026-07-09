using SentenceTransformers;
using SentenceTransformers.Bert.Pure;
using SentenceTransformers.Bert.Pure.Model;
using SentenceTransformers.Bert.Pure.Training;
using SentenceTransformers.Training;

namespace SentenceTransformers.Tests;

/// <summary>
/// End-to-end tests for the pure-C# BERT encoder and its real weight-space LoRA training. All weights are
/// read from the fp32 ONNX graphs already embedded in the MiniLM / ArcticXs packages (no download), so the
/// tests are self-contained.
/// </summary>
public class BertPureTests
{
    private static byte[] MiniLmOnnx() => ResourceLoader.GetResource(typeof(SentenceTransformers.MiniLM.SentenceEncoder).Assembly, "model.onnx");
    private static byte[] ArcticOnnx() => ResourceLoader.GetResource(typeof(SentenceTransformers.ArcticXs.SentenceEncoder).Assembly, "model.onnx");

    private static readonly string[] Samples =
    {
        "how do I reset my password",
        "The quick brown fox jumps over the lazy dog.",
        "Neural networks learn representations from data.",
    };

    private static float Cos(float[] a, float[] b)
    {
        double d = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
        return (float)(d / (Math.Sqrt(na) * Math.Sqrt(nb) + 1e-12));
    }

    [Fact]
    public async Task PureMiniLM_MatchesOnnx()
    {
        using var pure = SentenceEncoder.LoadFromOnnx(MiniLmOnnx(), BertConfig.MiniLM, 256);
        using var onnx = new SentenceTransformers.MiniLM.SentenceEncoder();
        var p = await pure.EncodeAsync(Samples);
        var o = await onnx.EncodeAsync(Samples);
        for (int i = 0; i < Samples.Length; i++)
            Assert.True(Cos(p[i], o[i]) > 0.9999f, $"MiniLM parity failed on sample {i}: cos={Cos(p[i], o[i])}");
    }

    [Fact]
    public async Task PureArctic_MatchesOnnx()
    {
        using var pure = SentenceEncoder.LoadFromOnnx(ArcticOnnx(), BertConfig.ArcticXs, 512);
        using var onnx = new SentenceTransformers.ArcticXs.SentenceEncoder();
        var p = await pure.EncodeAsync(Samples);
        var o = await onnx.EncodeAsync(Samples);
        for (int i = 0; i < Samples.Length; i++)
            Assert.True(Cos(p[i], o[i]) > 0.9999f, $"Arctic parity failed on sample {i}: cos={Cos(p[i], o[i])}");
    }

    [Fact]
    public async Task SingleEncode_MatchesBatch()
    {
        using var enc = SentenceEncoder.LoadFromOnnx(MiniLmOnnx(), BertConfig.MiniLM, 256);
        var single = await enc.EncodeAsync("how do I reset my password");
        var batch  = await enc.EncodeAsync(new[] { "how do I reset my password" });
        Assert.Equal(batch[0], single);
    }

    [Fact]
    public void Adapter_SaveLoad_RoundTrips()
    {
        var cfg = BertConfig.MiniLM;
        var adapter = LoraAdapter.CreateInitialized(cfg, rank: 6, alpha: 12f, targets: LoraTargets.All, seed: 3);
        var rng = new Random(1);
        // Give B non-zero values so the round-trip is non-trivial.
        adapter.OutputBias = new float[cfg.HiddenSize];
        for (int i = 0; i < adapter.OutputBias.Length; i++) adapter.OutputBias[i] = (float)rng.NextDouble();

        using var ms = new MemoryStream();
        adapter.Save(ms);
        ms.Position = 0;
        var loaded = LoraAdapter.Load(ms);

        Assert.Equal(adapter.Dimension, loaded.Dimension);
        Assert.Equal(adapter.Rank, loaded.Rank);
        Assert.Equal(adapter.Alpha, loaded.Alpha);
        Assert.Equal(adapter.Targets, loaded.Targets);
        Assert.Equal(adapter.ParameterCount, loaded.ParameterCount);
        Assert.Equal(adapter.OutputBias, loaded.OutputBias);
    }

    [Fact]
    public async Task Trainer_ImprovesGradedSpearman()
    {
        // Paraphrase pairs (high score) + unrelated pairs (low score): a low-rank adapter should raise the
        // graded (Spearman) alignment above the frozen baseline.
        (string a, string b, float s)[] data =
        {
            ("how do I reset my password", "steps to recover a forgotten password", 0.95f),
            ("cancel my subscription", "how to end my membership plan", 0.9f),
            ("the weather is sunny today", "it is a bright clear day outside", 0.9f),
            ("a dog runs in the park", "a canine is running across the field", 0.88f),
            ("i love this movie", "this film is fantastic and enjoyable", 0.9f),
            ("update my billing address", "change the address on my invoice", 0.9f),
            ("reset password", "the cat sleeps on the sofa", 0.05f),
            ("cancel subscription", "a rocket launched into space", 0.05f),
            ("sunny weather", "quarterly financial report", 0.05f),
            ("dog in the park", "install the software update", 0.05f),
            ("i love this movie", "the train arrives at noon", 0.05f),
            ("billing address", "photosynthesis in plants", 0.05f),
        };
        var pairs = Enumerable.Range(0, 4).SelectMany(_ => data).Select(d => new SentencePair(d.a, d.b, d.s)).ToList();
        var ds = new SentencePairDataset(pairs);

        using var enc = SentenceEncoder.LoadFromOnnx(MiniLmOnnx(), BertConfig.MiniLM, 64);
        var options = new BertLoraTrainingOptions
        {
            Objective = BertTrainingObjective.CoSent,
            Rank = 8, Alpha = 16, Targets = LoraTargets.Attention,
            Epochs = 8, BatchSize = 16, LearningRate = 1e-3f, Temperature = 0.05f,
            ValidationFraction = 0.25f, MaxTokens = 32, Seed = 1,
        };
        var report = await BertLoraTrainer.TrainAsync(enc, ds, options);

        Assert.True(report.BestSpearman > report.BaselineSpearman + 0.02f,
            $"Adapter did not improve Spearman: base {report.BaselineSpearman:0.000} -> tuned {report.BestSpearman:0.000}");
    }
}
