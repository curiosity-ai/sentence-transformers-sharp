using System.Buffers.Binary;
using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Harrier.Small.Pure.Numerics;

namespace SentenceTransformers.Tests;

/// <summary>
/// Unit tests for the pure-managed numeric kernels and the safetensors reader that back the
/// dependency-free Harrier Small encoder. These run without downloading model weights.
/// </summary>
public class PureNumericsTests
{
    [Theory]
    [InlineData(0f)]
    [InlineData(1f)]
    [InlineData(-2.5f)]
    [InlineData(3.1415927f)]
    [InlineData(1e-3f)]
    public void BFloat16_RoundTrip_IsCloseToOriginal(float value)
    {
        var bf = FloatConversions.SingleToBFloat16(value);
        var back = FloatConversions.BFloat16ToSingle(bf);
        // bfloat16 keeps 8 bits of mantissa => relative error <= ~2^-8.
        float tol = MathF.Max(1e-6f, MathF.Abs(value) * (1f / 128f));
        Assert.True(MathF.Abs(back - value) <= tol, $"{value} -> {back}");
    }

    [Fact]
    public void RmsNorm_MatchesReferenceFormula()
    {
        int dim = 8;
        var x = new float[dim];
        var w = new float[dim];
        var rng = new Random(42);
        for (int i = 0; i < dim; i++)
        {
            x[i] = (float)(rng.NextDouble() * 2 - 1);
            w[i] = (float)(rng.NextDouble() * 0.5);
        }

        float eps = 1e-6f;
        var got = new float[dim];
        Ops.RmsNorm(x, w, got, 1, dim, eps);

        // reference: y = x / sqrt(mean(x^2)+eps) * (1 + w)
        double meanSq = 0;
        for (int i = 0; i < dim; i++) meanSq += (double)x[i] * x[i];
        meanSq /= dim;
        double inv = 1.0 / Math.Sqrt(meanSq + eps);
        for (int i = 0; i < dim; i++)
        {
            float expected = (float)(x[i] * inv * (1 + w[i]));
            Assert.Equal(expected, got[i], 4);
        }
    }

    [Fact]
    public void Softmax_SumsToOne_AndIsMonotonicWithInput()
    {
        var s = new float[] { 1f, 2f, 3f, 0.5f, -1f };
        var copy = (float[])s.Clone();
        Ops.SoftmaxInPlace(s, s.Length);

        Assert.Equal(1.0, s.Sum(), 4);
        Assert.All(s, v => Assert.True(v is >= 0f and <= 1f));
        // largest input -> largest probability
        int argmaxIn = Array.IndexOf(copy, copy.Max());
        int argmaxOut = Array.IndexOf(s, s.Max());
        Assert.Equal(argmaxIn, argmaxOut);
    }

    [Fact]
    public void GeGlu_MatchesTanhGeluTimesUp()
    {
        var gate = new float[] { -1f, 0f, 0.5f, 2f };
        var up = new float[] { 1f, 2f, -1f, 0.5f };
        var expected = new float[gate.Length];
        const float c = 0.7978845608028654f;
        for (int i = 0; i < gate.Length; i++)
        {
            float v = gate[i];
            float gelu = 0.5f * v * (1f + MathF.Tanh(c * (v + 0.044715f * v * v * v)));
            expected[i] = gelu * up[i];
        }

        var g = (float[])gate.Clone();
        Ops.GeGluInPlace(g, up);
        for (int i = 0; i < g.Length; i++)
        {
            Assert.Equal(expected[i], g[i], 5);
        }
    }

    [Fact]
    public void L2Normalize_ProducesUnitVector()
    {
        var v = new float[] { 3f, 4f, 0f, 12f };
        Ops.L2NormalizeInPlace(v);
        double norm = Math.Sqrt(v.Sum(x => (double)x * x));
        Assert.Equal(1.0, norm, 5);
    }

    [Fact]
    public async Task Linear_ComputesRowMajorMatVec()
    {
        // y[s,o] = sum_i x[s,i] * w[o,i]
        int seq = 2, inDim = 3, outDim = 2;
        var x = new float[] { 1, 2, 3, /*row1*/ 4, 5, 6 };
        var w = new float[] { 1, 0, 0, /*o0*/ 0, 1, 1 }; // o0 picks x[0], o1 picks x[1]+x[2]
        var y = new float[seq * outDim];
        await Ops.LinearAsync(x, w, y, seq, inDim, outDim);

        Assert.Equal(1f, y[0]);   // row0,o0 = 1
        Assert.Equal(5f, y[1]);   // row0,o1 = 2+3
        Assert.Equal(4f, y[2]);   // row1,o0 = 4
        Assert.Equal(11f, y[3]);  // row1,o1 = 5+6
    }

    [Theory]
    [InlineData(Quantization.Int8, 0.03f)]
    [InlineData(Quantization.Int4, 0.08f)]
    public async Task QuantizedMatrix_ApproximatesFloatMatMul(Quantization quant, float tolerance)
    {
        int seq = 3, inDim = 256, outDim = 64;
        var rng = new Random(123);
        var w = new float[outDim * inDim];
        for (int i = 0; i < w.Length; i++) w[i] = (float)(rng.NextDouble() * 2 - 1);
        var x = new float[seq * inDim];
        for (int i = 0; i < x.Length; i++) x[i] = (float)(rng.NextDouble() * 2 - 1);

        var reference = IWeightMatrix.Create((float[])w.Clone(), outDim, inDim, Quantization.None);
        var quantized = IWeightMatrix.Create((float[])w.Clone(), outDim, inDim, quant);

        var yRef = new float[seq * outDim];
        var yQ = new float[seq * outDim];
        await reference.MultiplyAsync(x, yRef, seq, default);
        await quantized.MultiplyAsync(x, yQ, seq, default);

        // Compare via relative error on the output vectors (the dot products), which is what matters.
        double num = 0, den = 0;
        for (int i = 0; i < yRef.Length; i++)
        {
            num += (yRef[i] - yQ[i]) * (yRef[i] - yQ[i]);
            den += yRef[i] * (double)yRef[i];
        }
        double relErr = Math.Sqrt(num / den);
        Assert.True(relErr < tolerance, $"{quant} relative error {relErr:F4} exceeded {tolerance}");
    }

    [Fact]
    public void SafeTensors_RoundTrips_F32_And_BF16()
    {
        // Build a tiny safetensors file with one F32 and one BF16 tensor.
        var f32 = new float[] { 1.0f, -2.0f, 3.5f, 0.25f };
        var bf16Vals = new float[] { 1.0f, 2.0f, -0.5f };

        // serialize bf16
        var bf16Bytes = new byte[bf16Vals.Length * 2];
        for (int i = 0; i < bf16Vals.Length; i++)
        {
            BinaryPrimitives.WriteUInt16LittleEndian(bf16Bytes.AsSpan(i * 2, 2), FloatConversions.SingleToBFloat16(bf16Vals[i]));
        }
        var f32Bytes = new byte[f32.Length * 4];
        for (int i = 0; i < f32.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(f32Bytes.AsSpan(i * 4, 4), f32[i]);
        }

        // header: "a" is F32 [4] at [0, 16); "b" is BF16 [3] at [16, 22)
        string header = "{\"a\":{\"dtype\":\"F32\",\"shape\":[4],\"data_offsets\":[0,16]}," +
                        "\"b\":{\"dtype\":\"BF16\",\"shape\":[3],\"data_offsets\":[16,22]}}";
        var headerBytes = System.Text.Encoding.UTF8.GetBytes(header);

        var path = Path.Combine(Path.GetTempPath(), $"st_test_{Guid.NewGuid():N}.safetensors");
        try
        {
            using (var fs = File.Create(path))
            {
                var len = new byte[8];
                BinaryPrimitives.WriteUInt64LittleEndian(len, (ulong)headerBytes.Length);
                fs.Write(len);
                fs.Write(headerBytes);
                fs.Write(f32Bytes);
                fs.Write(bf16Bytes);
            }

            var st = SafeTensors.Load(path);
            Assert.True(st.Contains("a"));
            Assert.True(st.Contains("b"));
            Assert.Equal(new[] { 4 }, st.Shape("a"));

            var a = st.ReadFloat("a");
            Assert.Equal(f32, a);

            var b = st.ReadFloat("b");
            for (int i = 0; i < bf16Vals.Length; i++)
            {
                Assert.Equal(FloatConversions.RoundToBFloat16(bf16Vals[i]), b[i]);
            }
        }
        finally
        {
            File.Delete(path);
        }
    }
}
