using SentenceTransformers.EmbeddingGemma2.Audio;
using SentenceTransformers.EmbeddingGemma2.Numerics;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SentenceTransformers.Tests;

/// <summary>
/// Model-free checks of the ports that make EmbeddingGemma 2 bit-exact with the LiteRT-LM engine: the XNNPACK
/// activation kernels, the single-precision KISS FFT and the correctly rounded <c>logf</c> of the audio
/// front-end. Bit-exactness itself is checked end to end by <see cref="EmbeddingGemma2ModelTests"/>; these
/// tests pin down that each kernel computes the right function to its approximation error.
/// </summary>
public class EmbeddingGemma2KernelTests
{
    private static float[] Range(float from, float to, int n)
    {
        var x = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = from + (to - from) * i / (n - 1);
        }
        return x;
    }

    [Fact]
    public void Gelu_ApproximatesTanhGelu()
    {
        var x = Range(-12f, 12f, 4001);
        var y = (float[])x.Clone();
        Xnn.Gelu(y);
        for (int i = 0; i < x.Length; i++)
        {
            double v = x[i];
            double expected = 0.5 * v * (1 + Math.Tanh(Math.Sqrt(2 / Math.PI) * (v + 0.044715 * v * v * v)));
            Assert.True(Math.Abs(y[i] - expected) <= 2e-6 * Math.Max(1, Math.Abs(v)), $"gelu({v}) = {y[i]}, expected {expected}");
        }
    }

    [Fact]
    public void SigmoidAndTanh_MatchMath()
    {
        var x = Range(-30f, 30f, 6001);
        var s = new float[x.Length];
        var t = new float[x.Length];
        Xnn.Sigmoid(x, s);
        Xnn.Tanh(x, t);
        for (int i = 0; i < x.Length; i++)
        {
            double sig = 1 / (1 + Math.Exp(-x[i]));
            Assert.True(Math.Abs(s[i] - sig) <= 1e-6 * sig + 1e-38, $"sigmoid({x[i]}) = {s[i]}, expected {sig}");
            Assert.True(Math.Abs(t[i] - Math.Tanh(x[i])) <= 3e-7, $"tanh({x[i]}) = {t[i]}");
        }
    }

    [Fact]
    public void Softmax_IsNormalizedAndMasksPrefix()
    {
        var rng = new Random(7);
        foreach (int n in new[] { 1, 15, 16, 63, 64, 65, 200 })
        {
            var row = new float[n];
            for (int i = 0; i < n; i++)
            {
                row[i] = (float)(rng.NextDouble() * 20 - 10);
            }
            var expected = new double[n];
            double max = row.Max(), sum = 0;
            for (int i = 0; i < n; i++)
            {
                sum += expected[i] = Math.Exp(row[i] - max);
            }
            Xnn.Softmax(row);
            for (int i = 0; i < n; i++)
            {
                Assert.True(Math.Abs(row[i] - expected[i] / sum) <= 1e-6, $"n={n} i={i}: {row[i]} vs {expected[i] / sum}");
            }
        }
    }

    [Fact]
    public void SinCos_MatchMath()
    {
        foreach (float x in Range(-200f, 200f, 20001))
        {
            Assert.True(Math.Abs(Xnn.Sin(x) - Math.Sin(x)) <= 1e-6, $"sin({x}) = {Xnn.Sin(x)}, expected {Math.Sin(x)}");
            Assert.True(Math.Abs(Xnn.Cos(x) - Math.Cos(x)) <= 1e-6, $"cos({x}) = {Xnn.Cos(x)}, expected {Math.Cos(x)}");
        }
    }

    [Fact]
    public void RmsNorm_MatchesDefinition()
    {
        var rng = new Random(3);
        const int dim = 640;
        var x = new float[2 * dim];
        var w = new float[dim];
        for (int i = 0; i < x.Length; i++)
        {
            x[i] = (float)(rng.NextDouble() * 4 - 2);
        }
        for (int i = 0; i < dim; i++)
        {
            w[i] = (float)rng.NextDouble();
        }
        var y = new float[x.Length];
        Xnn.RmsNorm(x, w, y, dim, 1e-6f);
        for (int r = 0; r < 2; r++)
        {
            double ms = 0;
            for (int i = 0; i < dim; i++)
            {
                ms += (double)x[r * dim + i] * x[r * dim + i];
            }
            double inv = 1 / Math.Sqrt(ms / dim + 1e-6);
            for (int i = 0; i < dim; i++)
            {
                double e = x[r * dim + i] * inv * w[i];
                Assert.True(Math.Abs(y[r * dim + i] - e) <= 1e-6 * Math.Max(1, Math.Abs(e)), $"row {r} [{i}]");
            }
        }
    }

    [Fact]
    public void Rsqrt14_MatchesHardwareEstimate()
    {
        var inputs = new List<float> { 0f, -0f, float.PositiveInfinity, float.NegativeInfinity, float.NaN, -1f, 1f, 4f, 0.25f, 2f, 0.5f,
            float.Epsilon, 1e-40f, float.MaxValue, 1.17549435E-38f, BitConverter.UInt32BitsToSingle(0x7FC12345) };
        var rng = new Random(5);
        for (int i = 0; i < 1_000_000; i++)
        {
            inputs.Add(BitConverter.UInt32BitsToSingle((uint)rng.NextInt64(0, 1L << 32)));
        }
        foreach (float a in inputs)
        {
            float emu = Rsqrt14.Estimate(a);
            if (Avx512F.VL.IsSupported)
            {
                float hw = Avx512F.VL.ReciprocalSqrt14(Vector128.CreateScalarUnsafe(a)).ToScalar();
                Assert.True(BitConverter.SingleToUInt32Bits(hw) == BitConverter.SingleToUInt32Bits(emu), $"rsqrt14(0x{BitConverter.SingleToUInt32Bits(a):X8})");
            }
            else if (a > 0 && float.IsFinite(a))
            {
                // vrsqrt14 guarantees a relative error below 2^-14.
                Assert.True(Math.Abs(emu * Math.Sqrt(a) - 1) < 1.0 / 16384, $"rsqrt14({a:R}) = {emu:R}");
            }
        }
    }

    [Theory]
    [InlineData(64)]
    [InlineData(400)]
    [InlineData(512)]
    [InlineData(90)]   // radix 5 and 3 butterflies
    [InlineData(14)]   // generic (radix 7) butterfly
    public void KissFftr_MatchesDft(int n)
    {
        var rng = new Random(n);
        var x = new float[n];
        for (int i = 0; i < n; i++)
        {
            x[i] = (float)(rng.NextDouble() * 2 - 1);
        }
        var re = new float[n / 2 + 1];
        var im = new float[n / 2 + 1];
        new KissFftr(n).Forward(x, re, im);
        for (int k = 0; k <= n / 2; k++)
        {
            double er = 0, ei = 0;
            for (int t = 0; t < n; t++)
            {
                double a = -2 * Math.PI * k * t / n;
                er += x[t] * Math.Cos(a);
                ei += x[t] * Math.Sin(a);
            }
            Assert.True(Math.Abs(re[k] - er) <= 1e-5 * n && Math.Abs(im[k] - ei) <= 1e-5 * n, $"n={n} bin {k}: ({re[k]}, {im[k]}) vs ({er}, {ei})");
        }
    }

    [Fact]
    public void LogF_IsCorrectlyRounded()
    {
        // The five inputs where the double logarithm rounded to float is off by one ulp.
        foreach (var (x, y) in new (uint, uint)[] { (0x3C413D3A, 0xC08E158F), (0x41178FEB, 0x400FE5E7), (0x4C5D65A5, 0x418F034B), (0x65D890D3, 0x4254D1F9), (0x6F31A8EC, 0x42845A89) })
        {
            Assert.Equal(y, BitConverter.SingleToUInt32Bits(LogMelFrontend.LogF(BitConverter.UInt32BitsToSingle(x))));
        }
        var rng = new Random(11);
        for (int i = 0; i < 200_000; i++)
        {
            float x = BitConverter.UInt32BitsToSingle((uint)rng.Next(1, 0x7F800000));
            float y = LogMelFrontend.LogF(x);
            double exact = Math.Log(x);
            // Correct rounding: no float is closer to the true value than the result.
            Assert.True(Math.Abs(y - exact) <= Math.Abs(MathF.BitIncrement(y) - exact) && Math.Abs(y - exact) <= Math.Abs(MathF.BitDecrement(y) - exact), $"logf({x:R}) = {y:R}");
        }
    }
}
