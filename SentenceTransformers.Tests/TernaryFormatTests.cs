using SentenceTransformers.Ternary;

namespace SentenceTransformers.Tests;

/// <summary>
/// Covers the ternary container end to end without needing a real checkpoint: the packings, the
/// Hadamard rotation, the quantizer's optimality claim, and a write/read round trip.
/// </summary>
public class TernaryFormatTests
{
    [Theory]
    [InlineData(TernaryBand.TQ1_0, 128, 28, 1.75)]
    [InlineData(TernaryBand.TQ2_0, 128, 34, 2.125)]
    public void BandSizesMatchTheDocumentedRates(TernaryBand band, int groupSize, int expectedBytes, double expectedBpw)
    {
        // The whole point of the two bands is these numbers; if the arithmetic drifts, a file written
        // by one build stops lining up with what another expects.
        Assert.Equal(expectedBytes, TernaryFormat.CodeBytesPerGroup(band, groupSize) + 2);
        Assert.Equal(expectedBpw, TernaryFormat.BitsPerWeight(band, groupSize), 6);
    }

    [Theory]
    [InlineData(TernaryBand.TQ1_0, 128)]
    [InlineData(TernaryBand.TQ2_0, 128)]
    [InlineData(TernaryBand.TQ1_0, 64)]
    [InlineData(TernaryBand.TQ2_0, 64)]
    [InlineData(TernaryBand.TQ1_0, 32)]
    [InlineData(TernaryBand.TQ2_0, 32)]
    public void PackingRoundTripsEveryCode(TernaryBand band, int groupSize)
    {
        var rnd = new Random(7);
        var codes = new sbyte[groupSize];
        var packed = new byte[TernaryFormat.CodeBytesPerGroup(band, groupSize)];
        var unpacked = new sbyte[groupSize];

        for (int trial = 0; trial < 200; trial++)
        {
            for (int i = 0; i < groupSize; i++)
            {
                codes[i] = (sbyte)(rnd.Next(3) - 1);
            }
            TernaryPacking.PackGroup(band, codes, packed);
            TernaryPacking.UnpackGroup(band, packed, unpacked, groupSize);
            Assert.Equal(codes, unpacked);
        }
    }

    [Fact]
    public void Tq1PacksFiveTritsPerByteInBaseThree()
    {
        // Pin the on-disk byte layout, not just its round trip: a reader in another language has to be
        // able to reproduce it from the spec alone.
        sbyte[] codes = [1, 0, -1, 1, 1];   // -> 2 + 1*3 + 0*9 + 2*27 + 2*81 = 221
        var packed = new byte[1];
        TernaryPacking.PackGroup(TernaryBand.TQ1_0, codes, packed);
        Assert.Equal(221, packed[0]);
    }

    [Fact]
    public void Tq2PacksFourCodesPerByteTwoBitsEach()
    {
        sbyte[] codes = [1, 0, -1, 1];      // -> 2 | 1<<2 | 0<<4 | 2<<6 = 2 + 4 + 0 + 128 = 134
        var packed = new byte[1];
        TernaryPacking.PackGroup(TernaryBand.TQ2_0, codes, packed);
        Assert.Equal(134, packed[0]);
    }

    [Theory]
    [InlineData(640, 1024, 128)]    // Harrier's hidden width: 640 = 2^7 * 5, so 128 is the largest block
    [InlineData(1024, 1024, 1024)]
    [InlineData(2048, 1024, 1024)]  // capped
    [InlineData(2048, 4096, 2048)]
    [InlineData(768, 1024, 256)]
    public void ChooseBlockPicksTheLargestDividingPowerOfTwo(int dim, int max, int expected)
        => Assert.Equal(expected, TernaryRotation.ChooseBlock(dim, max));

    [Fact]
    public void FwhtAppliedTwiceScalesByN()
    {
        var rnd = new Random(11);
        var v = new float[64];
        for (int i = 0; i < v.Length; i++) v[i] = (float)(rnd.NextDouble() * 2 - 1);
        var original = (float[])v.Clone();

        TernaryRotation.Fwht(v);
        TernaryRotation.Fwht(v);

        for (int i = 0; i < v.Length; i++)
        {
            Assert.True(Math.Abs(original[i] * 64 - v[i]) < 1e-3, $"element {i}");
        }
    }

    [Theory]
    [InlineData(640, 128)]
    [InlineData(1024, 1024)]
    [InlineData(2048, 1024)]
    public void RotationIsItsOwnInverse(int dim, int block)
    {
        var rotation = TernaryRotation.Create(dim, block, 99);
        var rnd = new Random(3);
        var v = new float[dim];
        for (int i = 0; i < dim; i++) v[i] = (float)(rnd.NextDouble() * 2 - 1);
        var original = (float[])v.Clone();

        rotation.Apply(v);
        rotation.ApplyInverse(v);

        // A 1024-wide transform is ten levels of float adds deep each way, so compare with an
        // absolute tolerance rather than by rounding to a decimal place.
        for (int i = 0; i < dim; i++)
        {
            Assert.True(Math.Abs(original[i] - v[i]) < 1e-5, $"element {i}: {original[i]} -> {v[i]}");
        }
    }

    [Fact]
    public void RotationPreservesTheDotProduct()
    {
        // The identity the whole format rests on: W x = (W R^T)(R x). Applying the same rotation to
        // both operands must leave their dot product alone, because R is orthogonal.
        var rotation = TernaryRotation.Create(640, 128, 5);
        var rnd = new Random(21);
        var w = new float[640];
        var x = new float[640];
        for (int i = 0; i < 640; i++)
        {
            w[i] = (float)(rnd.NextDouble() * 2 - 1);
            x[i] = (float)(rnd.NextDouble() * 2 - 1);
        }

        double before = 0;
        for (int i = 0; i < 640; i++) before += (double)w[i] * x[i];

        rotation.Apply(w);
        rotation.Apply(x);

        double after = 0;
        for (int i = 0; i < 640; i++) after += (double)w[i] * x[i];

        Assert.True(Math.Abs(before - after) < 1e-3, $"dot product moved from {before} to {after}");
    }

    [Fact]
    public void OptimalQuantizerBeatsTheAlternativesAndMatchesBruteForce()
    {
        // The optimal method claims to be exactly optimal, not just good. Check it against an
        // exhaustive search over all 3^n sign/support assignments on groups small enough to enumerate.
        const int N = 7;
        var rnd = new Random(1234);
        var w = new float[N];
        var codes = new sbyte[N];

        for (int trial = 0; trial < 300; trial++)
        {
            for (int i = 0; i < N; i++)
            {
                w[i] = (float)(rnd.NextDouble() * 2 - 1);
            }

            float scale = TernaryQuantizer.QuantizeGroup(w, codes, TernaryMethod.Optimal);
            double ours = SquaredError(w, codes, scale);
            double best = BruteForceBestError(w);

            Assert.True(ours <= best + 1e-6, $"optimal method found {ours:E4}, brute force found {best:E4}");
        }
    }

    [Fact]
    public void OptimalQuantizerRecoversAlreadyTernaryWeightsExactly()
    {
        // The path that matters for a quantization-aware-trained checkpoint: if the weights are
        // already {-s, 0, +s} per group, converting them must be lossless (with the rotation off, since
        // rotating would destroy exactly that structure).
        var rnd = new Random(55);
        const int N = 128;
        var w = new float[N];
        var codes = new sbyte[N];
        const float s = 0.0137f;
        for (int i = 0; i < N; i++)
        {
            w[i] = (rnd.Next(3) - 1) * s;
        }

        float scale = TernaryQuantizer.QuantizeGroup(w, codes, TernaryMethod.Optimal);

        Assert.Equal(s, scale, 6);
        for (int i = 0; i < N; i++)
        {
            Assert.Equal(w[i], scale * codes[i], 6);
        }
    }

    [Fact]
    public void QuantizerHandlesAnAllZeroGroup()
    {
        var w = new float[128];
        var codes = new sbyte[128];
        foreach (var method in new[] { TernaryMethod.Optimal, TernaryMethod.AbsMean, TernaryMethod.Twn })
        {
            float scale = TernaryQuantizer.QuantizeGroup(w, codes, method);
            Assert.Equal(0f, scale);
            Assert.All(codes, c => Assert.Equal(0, c));
        }
    }

    [Theory]
    [InlineData(TernaryBand.TQ1_0, true)]
    [InlineData(TernaryBand.TQ2_0, true)]
    [InlineData(TernaryBand.TQ1_0, false)]
    public async Task ContainerRoundTripsThroughDisk(TernaryBand band, bool rotate)
    {
        const int Rows = 40, InDim = 640, GroupSize = 128;
        var rnd = new Random(8);
        var weights = new float[Rows * InDim];
        for (int i = 0; i < weights.Length; i++)
        {
            weights[i] = (float)(rnd.NextDouble() * 2 - 1);
        }
        var norm = new float[InDim];
        for (int i = 0; i < InDim; i++) norm[i] = (float)rnd.NextDouble();

        var rotation = rotate ? TernaryRotation.Create(InDim, 128, 17) : null;
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = 1 };
        var built = await TernaryTensorBuilder.BuildAsync("w", weights, Rows, InDim, band, GroupSize, rotation, TernaryMethod.Optimal, parallelOptions);

        var writer = new TernaryModelWriter().SetMetadata("architecture", "test");
        if (rotation is not null) writer.AddRotation("r", rotation);
        writer.AddTernary("w", band, [Rows, InDim], GroupSize, rotation is null ? null : "r", built.Codes, built.Scales);
        writer.AddRaw("n", TernaryBand.F32, [InDim], norm);

        var path = Path.Combine(Path.GetTempPath(), $"stq-test-{Guid.NewGuid():N}.stq");
        try
        {
            await writer.WriteAsync(path);
            var file = await TernaryModelFile.LoadAsync(path);

            Assert.Equal("test", file.Metadata["architecture"]);
            Assert.Equal(TernaryFormat.FormatVersion.ToString(), file.Metadata["format_version"]);

            // The float band is stored verbatim, so it must come back bit-identical.
            Assert.Equal(norm, file.ReadFloat("n"));

            // The ternary tensor comes back with exactly the error the builder reported - the reader
            // and the builder must agree on the reconstruction, whatever the quantization cost was.
            var decoded = file.ReadFloat("w");
            double se = 0, mag = 0;
            for (int i = 0; i < weights.Length; i++)
            {
                double d = weights[i] - decoded[i];
                se += d * d;
                mag += (double)weights[i] * weights[i];
            }
            double rel = Math.Sqrt(se / mag);
            Assert.Equal(built.Stats.RelativeError, rel, 4);

            // Rotated storage is what makes ternary survive; it should measurably beat unrotated here.
            Assert.True(rel < (rotate ? 0.47 : 0.60), $"relative error {rel:F4} is worse than expected for rotate={rotate}");
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    [Fact]
    public async Task ReaderRejectsAFileFromANewerFormat()
    {
        var writer = new TernaryModelWriter().SetMetadata("format_version", (TernaryFormat.FormatVersion + 1).ToString());
        writer.AddRaw("n", TernaryBand.F32, [4], new float[4]);
        var path = Path.Combine(Path.GetTempPath(), $"stq-test-{Guid.NewGuid():N}.stq");
        try
        {
            await writer.WriteAsync(path);
            var ex = await Assert.ThrowsAsync<InvalidDataException>(() => TernaryModelFile.LoadAsync(path));
            Assert.Contains("newer format", ex.Message);
        }
        finally
        {
            if (File.Exists(path)) File.Delete(path);
        }
    }

    private static double SquaredError(ReadOnlySpan<float> w, ReadOnlySpan<sbyte> codes, float scale)
    {
        double e = 0;
        for (int i = 0; i < w.Length; i++)
        {
            double d = w[i] - scale * codes[i];
            e += d * d;
        }
        return e;
    }

    /// <summary>Enumerates every ternary assignment, taking the best scale for each.</summary>
    private static double BruteForceBestError(float[] w)
    {
        int n = w.Length;
        int total = 1;
        for (int i = 0; i < n; i++) total *= 3;

        double best = double.MaxValue;
        var codes = new sbyte[n];
        for (int mask = 0; mask < total; mask++)
        {
            int m = mask;
            double dot = 0, norm = 0;
            for (int i = 0; i < n; i++)
            {
                sbyte c = (sbyte)(m % 3 - 1);
                m /= 3;
                codes[i] = c;
                dot += (double)w[i] * c;
                norm += (double)c * c;
            }
            if (norm == 0)
            {
                continue;
            }
            double scale = dot / norm;
            best = Math.Min(best, SquaredError(w, codes, (float)scale));
        }
        return best;
    }
}
