using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Stq;

namespace SentenceTransformers.Tests;

/// <summary>
/// Checks the runtime side of the ternary path: that <see cref="StqMatrix"/> and
/// <see cref="StqEmbedding"/> reproduce what the format's own reference decode says the weights
/// are.
///
/// <para>These are the tests that tell a kernel bug apart from quantization loss. Both look the same
/// from the outside - embeddings that drift from the fp32 reference - but only one is fixable in
/// code, so the runner is held to the decoded weights rather than to the original checkpoint.</para>
/// </summary>
public class StqRunnerTests
{
    private const int GroupSize = 128;   // the ternary default; the 4-bit tests use 32

    private static async Task<(StqFile File, string Path, float[] Weights)> WriteMatrixAsync(
        int rows, int inDim, StqBand band, bool rotate, int seed = 31)
    {
        var rnd = new Random(seed);
        var weights = new float[rows * inDim];
        for (int i = 0; i < weights.Length; i++)
        {
            // A heavy-tailed draw, so the rotation has outliers to spread - a uniform draw would
            // flatter the quantizer and hide any basis mistake.
            double u = rnd.NextDouble() * 2 - 1;
            weights[i] = (float)(u * u * u);
        }

        var rotation = rotate ? HadamardRotation.Create(inDim, HadamardRotation.ChooseBlock(inDim, 1024), 77) : null;
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 1 };
        int groupSize = StqFormat.DefaultGroupSizeFor(band);
        var built = await StqTensorBuilder.BuildAsync("w", weights, rows, inDim, band, groupSize, rotation, TernaryMethod.Optimal, parallelOptions);

        var writer = new StqWriter();
        if (rotation is not null) writer.AddRotation("r", rotation);
        writer.AddPacked("w", band, [rows, inDim], groupSize, rotation is null ? null : "r", built.Codes, built.Scales);

        var path = Path.Combine(Path.GetTempPath(), $"stq-runner-{Guid.NewGuid():N}.stq");
        await writer.WriteAsync(path);
        return (await StqFile.LoadAsync(path), path, weights);
    }

    [Theory]
    [InlineData(StqBand.TQ1_0, 640, true)]
    [InlineData(StqBand.TQ2_0, 640, true)]
    [InlineData(StqBand.TQ1_0, 1024, true)]
    [InlineData(StqBand.TQ1_0, 2048, true)]
    [InlineData(StqBand.TQ1_0, 640, false)]
    public async Task TernaryMatrixReproducesTheReferenceDecode(StqBand band, int inDim, bool rotate)
    {
        const int Rows = 96, Seq = 4;
        var (file, path, _) = await WriteMatrixAsync(Rows, inDim, band, rotate);
        try
        {
            var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 1 };
            var info = file.Info("w");

            var ternary = await StqMatrix.CreateAsync(file, info, parallelOptions);
            // FloatMatrix over the decoded weights takes a completely different route to the same
            // answer: it un-rotates the weights, where the kernel rotates the activation instead.
            var reference = new FloatMatrix(file.ReadFloat("w"), Rows, inDim);

            var rnd = new Random(5);
            var x = new float[Seq * inDim];
            for (int i = 0; i < x.Length; i++) x[i] = (float)(rnd.NextDouble() * 2 - 1);

            var yTernary = new float[Seq * Rows];
            var yReference = new float[Seq * Rows];
            await ternary.MultiplyAsync(x, yTernary, Seq, parallelOptions);
            await reference.MultiplyAsync(x, yReference, Seq, parallelOptions);

            double num = 0, den = 0;
            for (int i = 0; i < yReference.Length; i++)
            {
                double d = yTernary[i] - yReference[i];
                num += d * d;
                den += (double)yReference[i] * yReference[i];
            }
            double relative = Math.Sqrt(num / den);

            // On a VNNI host the kernel quantizes activations to int8, which costs ~1e-2 relative;
            // without VNNI the float path should agree far more tightly. Either way, anything above
            // this is a basis or packing mistake, not arithmetic noise.
            Assert.True(relative < 2e-2, $"kernel disagrees with the reference decode by {relative:E3}");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Theory]
    [InlineData(StqBand.TQ1_0, true)]
    [InlineData(StqBand.TQ2_0, true)]
    [InlineData(StqBand.TQ1_0, false)]
    public async Task TernaryEmbeddingLookupMatchesTheReferenceDecode(StqBand band, bool rotate)
    {
        const int Vocab = 256, Hidden = 640;
        var (file, path, _) = await WriteMatrixAsync(Vocab, Hidden, band, rotate, seed: 64);
        try
        {
            var info = file.Info("w");
            var embedding = new StqEmbedding(file, info);
            var decoded = file.ReadFloat("w");

            var row = new float[Hidden];
            const float Scale = 25.3f;   // stands in for the sqrt(hidden) embedding scale
            for (int token = 0; token < Vocab; token += 7)
            {
                embedding.Lookup(token, row, Scale);
                for (int i = 0; i < Hidden; i++)
                {
                    float expected = decoded[token * Hidden + i] * Scale;
                    Assert.True(Math.Abs(expected - row[i]) < 1e-3f,
                        $"token {token}, element {i}: expected {expected}, got {row[i]}");
                }
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public async Task AlreadyTernaryWeightsSurviveAConversionUnchanged()
    {
        // The path a quantization-aware-trained checkpoint takes: weights that are already {-s, 0, +s}
        // per group convert losslessly, provided the rotation is off (rotating would destroy exactly
        // the structure that makes them lossless).
        const int Rows = 32, InDim = 640;
        var rnd = new Random(19);
        var weights = new float[Rows * InDim];
        for (int r = 0; r < Rows; r++)
        {
            for (int g = 0; g < InDim / GroupSize; g++)
            {
                float s = 0.005f + (float)rnd.NextDouble() * 0.02f;
                s = (float)(Half)s;   // a scale the FP16 group field can hold exactly
                for (int i = 0; i < GroupSize; i++)
                {
                    weights[r * InDim + g * GroupSize + i] = (rnd.Next(3) - 1) * s;
                }
            }
        }

        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 1 };
        var built = await StqTensorBuilder.BuildAsync("w", weights, Rows, InDim, StqBand.TQ1_0, GroupSize,
                                                          rotation: null, TernaryMethod.Optimal, parallelOptions);

        Assert.True(built.Stats.RelativeError < 1e-6, $"expected a lossless conversion, got relative error {built.Stats.RelativeError:E3}");

        var writer = new StqWriter();
        writer.AddPacked("w", StqBand.TQ1_0, [Rows, InDim], GroupSize, null, built.Codes, built.Scales);
        var path = Path.Combine(Path.GetTempPath(), $"stq-qat-{Guid.NewGuid():N}.stq");
        try
        {
            await writer.WriteAsync(path);
            var file = await StqFile.LoadAsync(path);
            var decoded = file.ReadFloat("w");
            for (int i = 0; i < weights.Length; i++)
            {
                Assert.Equal(weights[i], decoded[i], 6);
            }
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }
}
