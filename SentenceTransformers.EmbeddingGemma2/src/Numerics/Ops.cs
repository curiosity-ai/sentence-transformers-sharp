using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>Vectorized elementwise kernels shared by the text, vision and audio encoders.</summary>
internal static class Ops
{
    /// <summary><c>odml.rms_norm</c>: <c>y = x · rsqrt(mean(x²) + eps) · w</c> applied to each
    /// <paramref name="dim"/>-sized row (in place allowed). The graphs fold Gemma's <c>1 + w</c> into <c>w</c>.</summary>
    public static void RmsNorm(ReadOnlySpan<float> x, ReadOnlySpan<float> weight, Span<float> dst, int dim, float eps = 1e-6f)
    {
        using var _ = Profiler.Measure("rmsnorm");
        int rows = x.Length / dim;
        for (int r = 0; r < rows; r++)
        {
            var row = x.Slice(r * dim, dim);
            var o = dst.Slice(r * dim, dim);
            float mean = TensorPrimitives.SumOfSquares(row) / dim;
            float inv = 1f / MathF.Sqrt(mean + eps);
            TensorPrimitives.Multiply(row, inv, o);
            if (!weight.IsEmpty)
            {
                TensorPrimitives.Multiply(o, weight, o);
            }
        }
    }

    /// <summary>Row-parallel <see cref="RmsNorm(ReadOnlySpan{float}, ReadOnlySpan{float}, Span{float}, int, float)"/> over
    /// <paramref name="rows"/> rows of <paramref name="dim"/> (src and dst may be the same array).</summary>
    public static void RmsNorm(float[] x, float[] weight, float[] dst, int rows, int dim, ParallelOptions po, float eps = 1e-6f)
        => ParallelRows.For(rows, dim, po, (r0, r1) => RmsNorm(x.AsSpan(r0 * dim, (r1 - r0) * dim), weight, dst.AsSpan(r0 * dim, (r1 - r0) * dim), dim, eps));

    /// <summary>Row-parallel <c>x += rms_norm(t)</c> (sandwich-norm residual), with <c>t</c> normalized in place.</summary>
    public static void AddRmsNorm(float[] x, float[] t, float[] weight, int rows, int dim, ParallelOptions po, float eps = 1e-6f)
        => ParallelRows.For(rows, dim, po, (r0, r1) =>
        {
            var ts = t.AsSpan(r0 * dim, (r1 - r0) * dim);
            RmsNorm(ts, weight, ts, dim, eps);
            var xs = x.AsSpan(r0 * dim, (r1 - r0) * dim);
            TensorPrimitives.Add(xs, ts, xs);
        });

    /// <summary>Element-parallel GELU (in place) over the first <paramref name="count"/> values.</summary>
    public static void GeluTanh(float[] x, int count, ParallelOptions po)
        => ParallelRows.For(count / 256 + 1, 256, po, (r0, r1) => GeluTanh(x.AsSpan(r0 * 256, Math.Min(count, r1 * 256) - Math.Min(count, r0 * 256))));

    /// <summary>Element-parallel GeGLU (<c>a = gelu(a)·b</c>) over the first <paramref name="count"/> values.</summary>
    public static void GeluMul(float[] a, float[] b, int count, ParallelOptions po)
        => ParallelRows.For(count / 256 + 1, 256, po, (r0, r1) =>
        {
            int s0 = Math.Min(count, r0 * 256), s1 = Math.Min(count, r1 * 256);
            GeluMul(a.AsSpan(s0, s1 - s0), b.AsSpan(s0, s1 - s0));
        });

    /// <summary>tanh-approximated GELU, in place: <c>0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))</c>,
    /// evaluated through the identity <c>0.5·(1 + tanh u) = σ(2u)</c> (no cancellation near zero).</summary>
    public static void GeluTanh(Span<float> x)
    {
        using var _ = Profiler.Measure("gelu");
        const float C = 0.7978845608028654f;   // sqrt(2/pi)
        const float A = 0.044715f;
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            ref float r = ref MemoryMarshal.GetReference(x);
            var vc2 = Vector256.Create(-2f * C);
            var va = Vector256.Create(A);
            var one = Vector256<float>.One;
            for (; i + 8 <= x.Length; i += 8)
            {
                var v = Vector256.LoadUnsafe(ref r, (nuint)i);
                var u = vc2 * (v + va * v * v * v);
                (v / (one + Vector256.Exp(u))).StoreUnsafe(ref r, (nuint)i);
            }
        }
        for (; i < x.Length; i++)
        {
            float v = x[i];
            x[i] = v / (1f + MathF.Exp(-2f * C * (v + A * v * v * v)));
        }
    }

    /// <summary><c>a[i] = gelu(a[i]) · b[i]</c> (GeGLU), in place on <paramref name="a"/>.</summary>
    public static void GeluMul(Span<float> a, ReadOnlySpan<float> b)
    {
        GeluTanh(a);
        TensorPrimitives.Multiply(a, b, a);
    }

    /// <summary>Numerically stable softmax over each row of length <paramref name="n"/> with stride <paramref name="ld"/>.</summary>
    public static void SoftmaxRows(Span<float> x, int rows, int n, int ld)
    {
        for (int r = 0; r < rows; r++)
        {
            var row = x.Slice(r * ld, n);
            float max = TensorPrimitives.Max(row);
            TensorPrimitives.Subtract(row, max, row);
            TensorPrimitives.Exp(row, row);
            float sum = TensorPrimitives.Sum(row);
            TensorPrimitives.Multiply(row, 1f / sum, row);
        }
    }

    /// <summary>Half-rotation RoPE on one head vector: <c>[a, b] -> [a·cos − b·sin, b·cos + a·sin]</c>.</summary>
    public static void Rope(Span<float> head, ReadOnlySpan<float> cos, ReadOnlySpan<float> sin)
    {
        int half = cos.Length;
        var a = head.Slice(0, half);
        var b = head.Slice(half, half);
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            ref float ar = ref MemoryMarshal.GetReference(a);
            ref float br = ref MemoryMarshal.GetReference(b);
            ref float cr = ref MemoryMarshal.GetReference(cos);
            ref float sr = ref MemoryMarshal.GetReference(sin);
            for (; i + 8 <= half; i += 8)
            {
                var va = Vector256.LoadUnsafe(ref ar, (nuint)i);
                var vb = Vector256.LoadUnsafe(ref br, (nuint)i);
                var vc = Vector256.LoadUnsafe(ref cr, (nuint)i);
                var vs = Vector256.LoadUnsafe(ref sr, (nuint)i);
                (va * vc - vb * vs).StoreUnsafe(ref ar, (nuint)i);
                (vb * vc + va * vs).StoreUnsafe(ref br, (nuint)i);
            }
        }
        for (; i < half; i++)
        {
            float x0 = a[i], x1 = b[i];
            a[i] = x0 * cos[i] - x1 * sin[i];
            b[i] = x1 * cos[i] + x0 * sin[i];
        }
    }

    public static void L2NormalizeInPlace(Span<float> v)
    {
        float norm = TensorPrimitives.Norm(v);
        if (norm > 0)
        {
            TensorPrimitives.Multiply(v, 1f / norm, v);
        }
    }
}

/// <summary>Precomputed RoPE cos/sin tables: angle = (float)position · inv_freq[i], as the graph computes it.</summary>
internal sealed class RopeTable
{
    private readonly float[] _invFreq;
    private float[] _cos = Array.Empty<float>();
    private float[] _sin = Array.Empty<float>();
    private int _positions;
    private readonly object _lock = new();

    public int Half => _invFreq.Length;

    public RopeTable(float[] invFreq)
    {
        _invFreq = invFreq;
    }

    public static float[] InverseFrequencies(double theta, int headDim)
    {
        int half = headDim / 2;
        var f = new float[half];
        for (int i = 0; i < half; i++)
        {
            f[i] = (float)Math.Pow(theta, -(double)i / half);
        }
        return f;
    }

    public void Ensure(int positions)
    {
        if (positions <= _positions)
        {
            return;
        }
        lock (_lock)
        {
            if (positions <= _positions)
            {
                return;
            }
            int n = Math.Max(positions, 128);
            var c = new float[n * Half];
            var s = new float[n * Half];
            for (int p = 0; p < n; p++)
            {
                for (int i = 0; i < Half; i++)
                {
                    float ang = (float)p * _invFreq[i];
                    c[p * Half + i] = MathF.Cos(ang);
                    s[p * Half + i] = MathF.Sin(ang);
                }
            }
            _cos = c;
            _sin = s;
            _positions = n;
        }
    }

    public ReadOnlySpan<float> Cos(int position) => _cos.AsSpan(position * Half, Half);
    public ReadOnlySpan<float> Sin(int position) => _sin.AsSpan(position * Half, Half);
}
