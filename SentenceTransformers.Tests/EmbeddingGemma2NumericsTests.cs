using SentenceTransformers.EmbeddingGemma2.Numerics;

namespace SentenceTransformers.Tests;

/// <summary>
/// Model-free checks of the EmbeddingGemma 2 SIMD kernels against naive scalar references: the integer
/// GEMM must be bit-exact for every code path (int4-range and full int8 weights, ragged tiles, the
/// static requantization epilogue), the float GEMM and attention must agree to rounding, and the
/// activation quantizer must reproduce XNNPACK's <c>qd8</c> parameters.
/// </summary>
public class EmbeddingGemma2NumericsTests
{
    private static readonly ParallelOptions Parallel4 = new() { MaxDegreeOfParallelism = 4 };
    private static readonly ParallelOptions Serial = new() { MaxDegreeOfParallelism = 1 };

    private static float[] RandomFloats(Random rng, int n, float scale = 1f)
    {
        var x = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = (float)(rng.NextDouble() * 2 - 1) * scale;
        }
        return x;
    }

    private static sbyte[] RandomWeights(Random rng, int n, int min, int max)
    {
        var w = new sbyte[n];
        for (int i = 0; i < n; i++)
        {
            w[i] = (sbyte)rng.Next(min, max + 1);
        }
        return w;
    }

    [Theory]
    [InlineData(1, 1, 1, true)]
    [InlineData(3, 5, 7, true)]
    [InlineData(37, 70, 300, true)]
    [InlineData(64, 128, 512, true)]
    [InlineData(65, 96, 520, true)]
    [InlineData(33, 129, 1536, true)]
    [InlineData(5, 9, 77, false)]
    [InlineData(37, 70, 300, false)]
    [InlineData(64, 64, 768, false)]
    public void QGemm_MatchesNaiveIntegerReference(int n, int m, int k, bool smallRange)
    {
        var rng = new Random(n * 1000 + m * 10 + k);
        var x = RandomFloats(rng, n * k, 3f);
        var w = smallRange ? RandomWeights(rng, m * k, -8, 7) : RandomWeights(rng, m * k, -127, 127);
        var scale = RandomFloats(rng, m, 0.01f).Select(s => MathF.Abs(s) + 1e-3f).ToArray();
        var matrix = new QuantizedMatrix(m, k, w, scale) { Bias = RandomFloats(rng, m) };
        Assert.Equal(smallRange, matrix.SmallRange);

        var qa = new QuantizedActivations();
        qa.Quantize(x, n, k, k);

        var expected = new float[n * m];
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < m; c++)
            {
                long acc = 0;
                for (int i = 0; i < k; i++)
                {
                    acc += (qa.Data[r * qa.Stride + i] - 128 - qa.ZeroPoint[r]) * w[c * k + i];
                }
                expected[r * m + c] = (float)acc * qa.Scale[r] * scale[c] + matrix.Bias[c];
            }
        }

        foreach (var po in new[] { Serial, Parallel4 })
        {
            var y = new float[n * m];
            QGemm.Multiply(qa, matrix, y, m, po);
            Assert.Equal(expected, y);
        }

        // And the dynamically quantized product approximates the float one.
        double err = 0, mag = 0;
        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < m; c++)
            {
                double f = matrix.Bias[c];
                for (int i = 0; i < k; i++)
                {
                    f += x[r * k + i] * w[c * k + i] * (double)scale[c];
                }
                err = Math.Max(err, Math.Abs(f - expected[r * m + c]));
                mag = Math.Max(mag, Math.Abs(f));
            }
        }
        Assert.True(err <= 0.02 * mag + 1e-6, $"quantized GEMM error {err} vs magnitude {mag}");
    }

    [Theory]
    [InlineData(10, 12, 64, false)]
    [InlineData(37, 70, 300, true)]
    public void QGemm_StaticRequantization_MatchesNaive(int n, int m, int k, bool smallRange)
    {
        var rng = new Random(42 + n);
        var x = RandomFloats(rng, n * k, 2f);
        var w = smallRange ? RandomWeights(rng, m * k, -8, 7) : RandomWeights(rng, m * k, -127, 127);
        var wScale = Enumerable.Range(0, m).Select(i => 0.002f + 0.0001f * i).ToArray();
        var matrix = new QuantizedMatrix(m, k, w, wScale);
        float inScale = 2f / 127f, outScale = 0.05f;
        int inZero = 3;

        var qa = new QuantizedActivations();
        qa.QuantizeStatic(x, n, k, k, inScale, inZero, Parallel4);
        var rq = new QGemm.Requantization(inScale, wScale, outScale);
        var y = new float[n * m];
        QGemm.Multiply(qa, matrix, y, m, Parallel4, rq);

        for (int r = 0; r < n; r++)
        {
            for (int c = 0; c < m; c++)
            {
                long acc = 0;
                for (int i = 0; i < k; i++)
                {
                    int q = Math.Clamp((int)MathF.Round(x[r * k + i] * (1f / inScale), MidpointRounding.ToEven) + inZero, -128, 127);
                    acc += (q - inZero) * w[c * k + i];
                }
                float expected = Math.Clamp(MathF.Round((float)acc * rq.Scale[c], MidpointRounding.ToEven), -128f, 127f) * outScale;
                Assert.Equal(expected, y[r * m + c]);
            }
        }
    }

    [Theory]
    [InlineData(1)]
    [InlineData(31)]
    [InlineData(32)]
    [InlineData(33)]
    [InlineData(512)]
    [InlineData(1000)]
    public void QuantizeRow_MatchesXnnpackQd8(int length)
    {
        var rng = new Random(length);
        var x = RandomFloats(rng, length, 5f);
        x[0] = 4.5f;   // make sure both signs are present
        if (length > 40)
        {
            // Products that land exactly on .5 must round half to even before the zero point is added.
            float m = 255f / (MathF.Max(0, x.Max()) - MathF.Min(0, x.Min()));
            for (int i = 1; i < 40; i++)
            {
                x[i] = (i - 20 + 0.5f) / m;
            }
        }
        var dst = new byte[length];
        QuantizedActivations.QuantizeRow(x, dst, out float scale, out int zp);

        // xnn_f32_qd8_asymmetric_quantization_params, in float as XNNPACK evaluates it.
        float min = MathF.Min(0, x.Min()), max = MathF.Max(0, x.Max());
        float mult = 255f / (max - min);
        float dmin = min * mult, dmax = max * mult;
        float zpf = (-128f + dmin) + (127f + dmax) > 0 ? -128f - dmin : 127f - dmax;
        Assert.Equal(1f / mult, scale);
        Assert.Equal((int)MathF.Round(Math.Clamp(zpf, -128f, 127f), MidpointRounding.ToEven), zp);
        for (int i = 0; i < length; i++)
        {
            // f32-qs8-vcvt: q = sat8(sat16(rint(x · mult)) + zp).
            int q = Math.Clamp((int)MathF.Round(x[i] * mult, MidpointRounding.ToEven) + zp, -128, 127);
            Assert.Equal(q + 128, dst[i]);
            Assert.True(MathF.Abs((q - zp) * scale - x[i]) <= scale * 0.5f + 1e-6f);
        }
    }

    [Fact]
    public void QuantizeRow_AllZerosUsesUnitScale()
    {
        var dst = new byte[40];
        QuantizedActivations.QuantizeRow(new float[40], dst, out float scale, out int zp);
        Assert.Equal(1f, scale);
        Assert.All(dst, b => Assert.Equal(128 + zp, b));
    }

    [Theory]
    [InlineData(1, 1, 1)]
    [InlineData(6, 32, 128)]
    [InlineData(7, 33, 129)]
    [InlineData(50, 70, 300)]
    [InlineData(100, 17, 513)]
    public void SGemm_MatchesNaive(int m, int n, int k)
    {
        var rng = new Random(m + n + k);
        var a = RandomFloats(rng, m * k);
        var b = RandomFloats(rng, k * n);
        foreach (var po in new[] { Serial, Parallel4 })
        {
            var c = new float[m * n];
            SGemm.Multiply(a, k, b, n, c, n, m, n, k, 0.5f, po);
            for (int i = 0; i < m; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    double e = 0;
                    for (int p = 0; p < k; p++)
                    {
                        e += a[i * k + p] * (double)b[p * n + j];
                    }
                    Assert.True(Math.Abs(0.5 * e - c[i * n + j]) <= 1e-4 * Math.Sqrt(k), $"C[{i},{j}] = {c[i * n + j]}, expected {0.5 * e}");
                }
            }
        }
    }

    [Fact]
    public void SGemm_Transpose_PadsDestination()
    {
        var src = Enumerable.Range(0, 6).Select(i => (float)i).ToArray();   // 2×3
        var dst = Enumerable.Repeat(-1f, 3 * 4).ToArray();
        SGemm.Transpose(src, 3, 2, 3, dst, 4);
        Assert.Equal(new float[] { 0, 3, 0, 0, 1, 4, 0, 0, 2, 5, 0, 0 }, dst);
    }

    [Theory]
    [InlineData(4, 2, 64, false)]
    [InlineData(4, 1, 32, true)]
    public void Attention_MatchesNaiveGroupedQueryAttention(int heads, int kvHeads, int hd, bool twoSequences)
    {
        var rng = new Random(heads * 7 + hd);
        int[] offsets = twoSequences ? new[] { 0, 70, 75 } : new[] { 0, 130 };
        int total = offsets[^1];
        var q = RandomFloats(rng, total * heads * hd);
        var k = RandomFloats(rng, total * kvHeads * hd);
        var v = RandomFloats(rng, total * kvHeads * hd);
        var output = new float[total * heads * hd];
        const float Scale = 0.3f;
        Attention.Run(q, k, v, output, offsets, heads, kvHeads, hd, Scale, Parallel4);

        for (int s = 0; s + 1 < offsets.Length; s++)
        {
            for (int h = 0; h < heads; h++)
            {
                int g = h * kvHeads / heads;
                for (int i = offsets[s]; i < offsets[s + 1]; i++)
                {
                    var scores = new double[offsets[s + 1] - offsets[s]];
                    for (int j = offsets[s]; j < offsets[s + 1]; j++)
                    {
                        double d = 0;
                        for (int e = 0; e < hd; e++)
                        {
                            d += q[(i * heads + h) * hd + e] * (double)k[(j * kvHeads + g) * hd + e];
                        }
                        scores[j - offsets[s]] = d * Scale;
                    }
                    double mx = scores.Max(), sum = 0;
                    for (int j = 0; j < scores.Length; j++)
                    {
                        scores[j] = Math.Exp(scores[j] - mx);
                        sum += scores[j];
                    }
                    for (int e = 0; e < hd; e++)
                    {
                        double o = 0;
                        for (int j = 0; j < scores.Length; j++)
                        {
                            o += scores[j] / sum * v[((offsets[s] + j) * kvHeads + g) * hd + e];
                        }
                        Assert.True(Math.Abs(o - output[(i * heads + h) * hd + e]) < 1e-5, $"seq {s} head {h} row {i} dim {e}");
                    }
                }
            }
        }
    }

    [Fact]
    public void RmsNorm_And_Gelu_MatchScalarFormulas()
    {
        var rng = new Random(5);
        const int Dim = 77;
        var x = RandomFloats(rng, Dim, 4f);
        var w = RandomFloats(rng, Dim);
        var dst = new float[Dim];
        Ops.RmsNorm(x, w, dst, Dim, 1e-6f);
        double ms = x.Select(v => (double)v * v).Average();
        for (int i = 0; i < Dim; i++)
        {
            Assert.Equal(x[i] / Math.Sqrt(ms + 1e-6) * w[i], dst[i], 5);
        }

        var g = (float[])x.Clone();
        Ops.GeluTanh(g.AsSpan());
        for (int i = 0; i < Dim; i++)
        {
            double v = x[i];
            double expected = 0.5 * v * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
            Assert.Equal(expected, g[i], 5);
        }
    }

    [Fact]
    public void Rope_RotatesHalves()
    {
        var head = new float[] { 1, 2, 3, 4, 10, 20, 30, 40 };
        var angles = new[] { 0.1, 0.2, 0.3, 0.4 };
        var cos = angles.Select(a => (float)Math.Cos(a)).ToArray();
        var sin = angles.Select(a => (float)Math.Sin(a)).ToArray();
        var original = (float[])head.Clone();
        Ops.Rope(head, cos, sin);
        for (int i = 0; i < 4; i++)
        {
            Assert.Equal(original[i] * cos[i] - original[i + 4] * sin[i], head[i], 5);
            Assert.Equal(original[i + 4] * cos[i] + original[i] * sin[i], head[i + 4], 5);
        }
    }
}
