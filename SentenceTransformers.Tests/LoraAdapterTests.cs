using BERTTokenizers.Base;
using SentenceTransformers;
using SentenceTransformers.Training;

namespace SentenceTransformers.Tests;

/// <summary>
/// Tests for the model-agnostic LoRA adapter, its serialization, the <see cref="AdaptedSentenceEncoder"/>
/// wrapper, and the contrastive trainer. The trainer test uses a synthetic in-memory encoder (no ONNX
/// model, no downloads) whose embeddings hide a shared signal under heavy noise; a correctly-implemented
/// low-rank adapter + gradient must learn to amplify that signal and separate positive pairs, which is a
/// strong end-to-end check that the analytic gradients are right.
/// </summary>
public class LoraAdapterTests
{
    // ----- adapter forward / init -----------------------------------------------------------------

    [Fact]
    public void FreshAdapter_IsIdentityUpToNormalization()
    {
        var adapter = LoraAdapter.CreateInitialized(dimension: 8, rank: 4, seed: 3);

        var e       = new float[] { 3f, 0f, 4f, 0f, 0f, 0f, 0f, 0f }; // norm 5
        var adapted = adapter.Apply(e);

        // B initializes to zero, so the residual is zero and Apply just L2-normalizes the input.
        Assert.Equal(0.6f, adapted[0], 5);
        Assert.Equal(0.8f, adapted[2], 5);

        float norm = MathF.Sqrt(adapted.Sum(x => x * x));
        Assert.Equal(1f, norm, 4);
    }

    [Fact]
    public void Apply_AlwaysProducesUnitVectors()
    {
        var adapter = LoraAdapter.CreateInitialized(16, 8, seed: 7);

        // Force a non-trivial residual by writing into B.
        var rng = new Random(1);
        for (int i = 0; i < adapter.B.Length; i++) adapter.B[i] = (float)(rng.NextDouble() - 0.5);

        var e = Enumerable.Range(0, 16).Select(i => (float)Math.Sin(i)).ToArray();
        var u = adapter.Apply(e);

        Assert.Equal(1f, MathF.Sqrt(u.Sum(x => x * x)), 4);
    }

    // ----- serialization --------------------------------------------------------------------------

    [Fact]
    public void SaveLoad_RoundTrips()
    {
        var adapter = LoraAdapter.CreateInitialized(12, 5, alpha: 10f, seed: 11);
        var rng     = new Random(2);
        for (int i = 0; i < adapter.B.Length; i++) adapter.B[i] = (float)rng.NextDouble();

        using var ms = new MemoryStream();
        adapter.Save(ms);
        ms.Position = 0;
        var loaded = LoraAdapter.Load(ms);

        Assert.Equal(adapter.Dimension, loaded.Dimension);
        Assert.Equal(adapter.Rank, loaded.Rank);
        Assert.Equal(adapter.Alpha, loaded.Alpha);
        Assert.Equal(adapter.A, loaded.A);
        Assert.Equal(adapter.B, loaded.B);

        // Applying the loaded adapter matches the original exactly.
        var e = Enumerable.Range(0, 12).Select(i => (float)i - 6f).ToArray();
        Assert.Equal(adapter.Apply(e), loaded.Apply(e));
    }

    [Fact]
    public void Load_RejectsGarbage()
    {
        using var ms = new MemoryStream(new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18 });
        Assert.Throws<InvalidDataException>(() => LoraAdapter.Load(ms));
    }

    // ----- adapted encoder wrapper ----------------------------------------------------------------

    [Fact]
    public async Task AdaptedEncoder_AppliesAdapterToBaseVectors()
    {
        var vectors = new Dictionary<string, float[]>
        {
            ["a"] = new[] { 1f, 2f, 3f, 4f },
            ["b"] = new[] { -1f, 0f, 2f, 1f },
        };
        using var baseEnc = new DictionaryEncoder(vectors, dim: 4);
        var       adapter = LoraAdapter.CreateInitialized(4, 2, seed: 5);
        using var adapted = new AdaptedSentenceEncoder(baseEnc, adapter);

        var outVecs = await adapted.EncodeAsync(new[] { "a", "b" });

        Assert.Equal(2, outVecs.Length);
        for (int i = 0; i < outVecs.Length; i++)
        {
            var expected = adapter.Apply(vectors[i == 0 ? "a" : "b"]);
            Assert.Equal(expected, outVecs[i]);
            Assert.Equal(1f, MathF.Sqrt(outVecs[i].Sum(x => x * x)), 4);
        }
    }

    [Fact]
    public async Task AdaptedEncoder_ThrowsOnDimensionMismatch()
    {
        var vectors = new Dictionary<string, float[]> { ["x"] = new[] { 1f, 2f, 3f, 4f } };
        using var baseEnc = new DictionaryEncoder(vectors, dim: 4);
        var       adapter = LoraAdapter.CreateInitialized(8, 2); // wrong dimension
        using var adapted = new AdaptedSentenceEncoder(baseEnc, adapter);

        await Assert.ThrowsAsync<InvalidOperationException>(() => adapted.EncodeAsync(new[] { "x" }));
    }

    // ----- dataset split --------------------------------------------------------------------------

    [Fact]
    public void Split_IsDeterministicAndDisjoint()
    {
        var pairs   = Enumerable.Range(0, 100).Select(i => new SentencePair($"a{i}", $"p{i}")).ToArray();
        var dataset = new SentencePairDataset(pairs);

        var (t1, v1) = dataset.Split(0.2f, seed: 99);
        var (t2, v2) = dataset.Split(0.2f, seed: 99);

        Assert.Equal(20, v1.Count);
        Assert.Equal(80, t1.Count);

        // Same seed -> identical split.
        Assert.Equal(v1.Pairs.Select(p => p.Anchor), v2.Pairs.Select(p => p.Anchor));

        // Train and validation are disjoint and cover everything.
        var all = t1.Pairs.Concat(v1.Pairs).Select(p => p.Anchor).OrderBy(x => x).ToArray();
        Assert.Equal(pairs.Select(p => p.Anchor).OrderBy(x => x), all);
    }

    // ----- end-to-end training --------------------------------------------------------------------

    [Fact]
    public async Task Trainer_LearnsToSeparatePositivePairs()
    {
        const int dim        = 32;
        const int signalDims = 6;
        const int pairCount  = 80;

        var (encoder, dataset) = BuildSyntheticProblem(dim, signalDims, pairCount, seed: 123);

        var epochLosses = new List<float>();
        var options = new LoraTrainingOptions
        {
            Rank               = 8,
            Epochs             = 200,
            BatchSize          = 16,
            LearningRate       = 0.05f,
            WeightDecay        = 0f,
            Temperature        = 0.07f,
            ValidationFraction = 0.25f,
            Seed               = 7,
            OnEpoch            = m => epochLosses.Add(m.TrainLoss),
        };

        using (encoder)
        {
            var report = await LoraTrainer.TrainAsync(encoder, dataset, options);

            // Training loss must fall substantially — wrong gradients would leave it flat or growing.
            Assert.True(epochLosses[^1] < epochLosses[0] * 0.7f,
                $"Training loss did not decrease enough: {epochLosses[0]:0.000} -> {epochLosses[^1]:0.000}");

            // The adapter must clearly beat the frozen baseline on validation retrieval.
            Assert.True(report.BestAccuracy > report.BaselineAccuracy + 0.15f,
                $"Adapter did not improve retrieval: baseline {report.BaselineAccuracy:0.000}, tuned {report.BestAccuracy:0.000}");

            Assert.True(report.BestAccuracy > 0.6f,
                $"Tuned retrieval accuracy too low: {report.BestAccuracy:0.000}");
        }
    }

    /// <summary>
    /// Builds embeddings that carry a shared per-pair signal in the first <paramref name="signalDims"/>
    /// coordinates, buried under heavy full-dimension noise. Positive pairs share the signal; different
    /// pairs have independent signals. A low-rank residual adapter can learn to amplify the signal
    /// subspace, which is exactly what should raise positive-pair cosine above cross-pair cosine.
    /// </summary>
    private static (DictionaryEncoder encoder, SentencePairDataset dataset) BuildSyntheticProblem(int dim, int signalDims, int pairCount, int seed)
    {
        var rng     = new Random(seed);
        var vectors = new Dictionary<string, float[]>(StringComparer.Ordinal);
        var pairs   = new List<SentencePair>(pairCount);

        // Signal dims carry the shared per-pair signal with only light noise; the remaining dims are
        // pure heavy noise. At baseline the heavy-noise dims dominate the norm, so positive-pair cosine
        // is swamped and retrieval is near chance. A low-rank adapter can learn to amplify the signal
        // subspace, which effectively drops the noise dims out of the (normalized) comparison.
        const float SignalScale     = 1.5f;
        const float SignalNoiseStd  = 0.3f;
        const float BackgroundStd   = 2.0f;

        float[] Make(float[] signal)
        {
            var v = new float[dim];
            for (int i = 0; i < signalDims; i++) v[i] = signal[i] + (float)(SignalNoiseStd * Gaussian(rng));
            for (int i = signalDims; i < dim; i++) v[i] = (float)(BackgroundStd * Gaussian(rng));
            return v;
        }

        for (int i = 0; i < pairCount; i++)
        {
            var signal = new float[signalDims];
            for (int k = 0; k < signalDims; k++) signal[k] = (float)(SignalScale * Gaussian(rng));

            string a = $"anchor-{i}";
            string p = $"positive-{i}";
            vectors[a] = Make(signal);
            vectors[p] = Make(signal);
            pairs.Add(new SentencePair(a, p)); // no score -> treated as a positive pair
        }

        return (new DictionaryEncoder(vectors, dim), new SentencePairDataset(pairs));
    }

    private static double Gaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }

    /// <summary>Minimal in-memory <see cref="ISentenceEncoder"/> returning a fixed vector per text.</summary>
    private sealed class DictionaryEncoder : ISentenceEncoder
    {
        private readonly Dictionary<string, float[]> _vectors;
        public DictionaryEncoder(Dictionary<string, float[]> vectors, int dim) { _vectors = vectors; MaxChunkLength = dim; }

        public int           MaxChunkLength { get; }
        public TokenizerBase Tokenizer      => null;

        public Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
        {
            var result = new float[sentences.Length][];
            for (int i = 0; i < sentences.Length; i++) result[i] = (float[])_vectors[sentences[i]].Clone();
            return Task.FromResult(result);
        }

        public void Dispose() { }
    }
}
