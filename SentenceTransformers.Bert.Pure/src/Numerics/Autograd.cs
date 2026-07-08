using System.Numerics.Tensors;

namespace SentenceTransformers.Bert.Pure.Numerics;

/// <summary>
/// A row-major 2-D tensor node in a reverse-mode automatic-differentiation graph. <see cref="Data"/>
/// holds the forward values; <see cref="Grad"/> accumulates <c>dLoss/dThis</c> during the backward pass.
///
/// <para>The whole point of this tiny engine is that the BERT forward pass can be written once, as a
/// sequence of <see cref="Graph"/> ops, and the exact gradients w.r.t. the (few) trainable LoRA
/// parameters fall out automatically — no hand-derived backprop through attention/LayerNorm/GELU, and
/// no autodiff framework dependency. It is deliberately tensor-level (nodes are whole matrices, not
/// scalars) so a full 6-layer forward is only a few hundred nodes.</para>
/// </summary>
internal sealed class Tensor
{
    public readonly float[] Data;
    public readonly float[] Grad;
    public readonly int     Rows;
    public readonly int     Cols;

    public int Length => Rows * Cols;

    public Tensor(int rows, int cols)
    {
        Rows = rows;
        Cols = cols;
        Data = new float[rows * cols];
        Grad = new float[rows * cols];
    }

    public Tensor(float[] data, int rows, int cols)
    {
        if (data.Length != rows * cols) throw new ArgumentException("data length does not match rows*cols");
        Rows = rows;
        Cols = cols;
        Data = data;
        Grad = new float[rows * cols];
    }

    public void ClearGrad() => Array.Clear(Grad);
}

/// <summary>
/// Records the ops that produced a set of <see cref="Tensor"/>s so <see cref="Backward"/> can replay
/// their local vector-Jacobian products in reverse. Frozen weights/biases are passed as plain
/// <c>float[]</c> and never enter the graph, so no gradient is computed or stored for them — only the
/// activations and the trainable LoRA tensors carry grad. That is exactly the LoRA efficiency win:
/// activations still backprop through the frozen network, but the expensive weight gradients are skipped.
/// </summary>
internal sealed class Graph
{
    private readonly List<Action> _tape = new(512);

    /// <summary>Runs every recorded backward closure in reverse creation order, accumulating grads.</summary>
    public void Backward()
    {
        for (int i = _tape.Count - 1; i >= 0; i--) _tape[i]();
    }

    private Tensor Record(Tensor y, Action backward)
    {
        _tape.Add(backward);
        return y;
    }

    /// <summary>A graph leaf (no parents); its grad is accumulated but never propagated further.</summary>
    public static Tensor Leaf(float[] data, int rows, int cols) => new Tensor(data, rows, cols);

    // ----- linear algebra --------------------------------------------------------------------------

    /// <summary>
    /// A LoRA-augmented linear layer: <c>y = x·Wᵀ + b + scale·(x·Aᵀ)·Bᵀ</c>. <paramref name="w"/> (shape
    /// <c>[outDim, inDim]</c>) and <paramref name="b"/> (<c>[outDim]</c>, may be null) are frozen. When
    /// <paramref name="a"/>/<paramref name="b_"/> are non-null they are the trainable low-rank factors
    /// (<c>A: [rank, inDim]</c>, <c>B: [outDim, rank]</c>) and receive gradients; when null the layer is a
    /// plain frozen projection (only the input gradient flows).
    /// </summary>
    public Tensor LoraLinear(Tensor x, float[] w, float[] bias, int outDim, int inDim, Tensor a, Tensor b_, float scale)
    {
        int m = x.Rows;
        if (x.Cols != inDim) throw new ArgumentException("LoraLinear input width mismatch");

        var y = new Tensor(m, outDim);

        // base: y = x·Wᵀ (+ bias). W row o is w[o*inDim .. ].
        for (int i = 0; i < m; i++)
        {
            var xi = x.Data.AsSpan(i * inDim, inDim);
            int yo = i * outDim;
            for (int o = 0; o < outDim; o++)
            {
                float v = TensorPrimitives.Dot(xi, w.AsSpan(o * inDim, inDim));
                if (bias != null) v += bias[o];
                y.Data[yo + o] = v;
            }
        }

        int rank = a?.Rows ?? 0;
        float[] h = null; // low-rank hidden [m, rank], kept for backward
        if (a != null)
        {
            h = new float[m * rank];
            for (int i = 0; i < m; i++)
            {
                var xi = x.Data.AsSpan(i * inDim, inDim);
                for (int r = 0; r < rank; r++)
                {
                    h[i * rank + r] = TensorPrimitives.Dot(xi, a.Data.AsSpan(r * inDim, inDim));
                }
                // low = h·Bᵀ ; y += scale·low
                int yo = i * outDim;
                var hi = h.AsSpan(i * rank, rank);
                for (int o = 0; o < outDim; o++)
                {
                    y.Data[yo + o] += scale * TensorPrimitives.Dot(hi, b_.Data.AsSpan(o * rank, rank));
                }
            }
        }

        return Record(y, () =>
        {
            // dx += dY·W   (base path)
            for (int i = 0; i < m; i++)
            {
                var dyi = y.Grad.AsSpan(i * outDim, outDim);
                var dxi = x.Grad.AsSpan(i * inDim, inDim);
                for (int o = 0; o < outDim; o++)
                {
                    float g = dyi[o];
                    if (g != 0f) TensorPrimitives.MultiplyAdd(w.AsSpan(o * inDim, inDim), g, dxi, dxi);
                }
            }

            if (a == null) return;

            // dh = scale·(dY·B) ; dB += scale·dYᵀ·h ; dA += dhᵀ·x ; dx += dh·A
            var dh = new float[m * rank];
            for (int i = 0; i < m; i++)
            {
                var dyi = y.Grad.AsSpan(i * outDim, outDim);
                var dhi = dh.AsSpan(i * rank, rank);
                for (int o = 0; o < outDim; o++)
                {
                    float g = scale * dyi[o];
                    if (g == 0f) continue;
                    var bo  = b_.Data.AsSpan(o * rank, rank);
                    var dbo = b_.Grad.AsSpan(o * rank, rank);
                    float hio;
                    var hi = h.AsSpan(i * rank, rank);
                    for (int r = 0; r < rank; r++)
                    {
                        dhi[r]  += g * bo[r];
                        hio      = hi[r];
                        dbo[r]  += g * hio;
                    }
                }
            }
            for (int i = 0; i < m; i++)
            {
                var dhi = dh.AsSpan(i * rank, rank);
                var xi  = x.Data.AsSpan(i * inDim, inDim);
                var dxi = x.Grad.AsSpan(i * inDim, inDim);
                for (int r = 0; r < rank; r++)
                {
                    float g = dhi[r];
                    if (g == 0f) continue;
                    TensorPrimitives.MultiplyAdd(xi, g, a.Grad.AsSpan(r * inDim, inDim), a.Grad.AsSpan(r * inDim, inDim)); // dA_r += g·x_i
                    TensorPrimitives.MultiplyAdd(a.Data.AsSpan(r * inDim, inDim), g, dxi, dxi);                            // dx_i += g·A_r
                }
            }
        });
    }

    /// <summary>Matrix product. When <paramref name="transposeB"/> is false: <c>[m,k]·[k,n]</c>; when true:
    /// <c>[m,k]·([n,k])ᵀ = [m,n]</c>. Both operands are differentiable activations.</summary>
    public Tensor MatMul(Tensor aT, Tensor bT, bool transposeB)
    {
        int m = aT.Rows, k = aT.Cols;
        int n = transposeB ? bT.Rows : bT.Cols;
        if (transposeB) { if (bT.Cols != k) throw new ArgumentException("MatMul dim mismatch (transB)"); }
        else            { if (bT.Rows != k) throw new ArgumentException("MatMul dim mismatch"); }

        var y = new Tensor(m, n);
        var A = aT.Data; var B = bT.Data; var Y = y.Data;

        if (transposeB)
        {
            for (int i = 0; i < m; i++)
            {
                var ai = A.AsSpan(i * k, k);
                for (int j = 0; j < n; j++)
                    Y[i * n + j] = TensorPrimitives.Dot(ai, B.AsSpan(j * k, k));
            }
        }
        else
        {
            for (int i = 0; i < m; i++)
            {
                int yb = i * n;
                for (int l = 0; l < k; l++)
                {
                    float av = A[i * k + l];
                    if (av == 0f) continue;
                    var brow = B.AsSpan(l * n, n);
                    var yr   = Y.AsSpan(yb, n);
                    TensorPrimitives.MultiplyAdd(brow, av, yr, yr);
                }
            }
        }

        return Record(y, () =>
        {
            var dY = y.Grad; var dA = aT.Grad; var dB = bT.Grad;
            if (transposeB)
            {
                // Y=A·Bᵀ ; dA += dY·B ; dB += dYᵀ·A
                for (int i = 0; i < m; i++)
                {
                    var dyi = dY.AsSpan(i * n, n);
                    var dai = dA.AsSpan(i * k, k);
                    for (int j = 0; j < n; j++)
                    {
                        float g = dyi[j];
                        if (g != 0f) TensorPrimitives.MultiplyAdd(B.AsSpan(j * k, k), g, dai, dai);
                    }
                }
                for (int j = 0; j < n; j++)
                {
                    var dbj = dB.AsSpan(j * k, k);
                    for (int i = 0; i < m; i++)
                    {
                        float g = dY[i * n + j];
                        if (g != 0f) TensorPrimitives.MultiplyAdd(A.AsSpan(i * k, k), g, dbj, dbj);
                    }
                }
            }
            else
            {
                // Y=A·B ; dA += dY·Bᵀ ; dB += Aᵀ·dY
                for (int i = 0; i < m; i++)
                {
                    var dyi = dY.AsSpan(i * n, n);
                    for (int l = 0; l < k; l++)
                    {
                        float g = TensorPrimitives.Dot(dyi, B.AsSpan(l * n, n));
                        dA[i * k + l] += g;
                    }
                    var ai = A.AsSpan(i * k, k);
                    for (int l = 0; l < k; l++)
                    {
                        float av = ai[l];
                        if (av != 0f)
                        {
                            var dbl = dB.AsSpan(l * n, n);
                            TensorPrimitives.MultiplyAdd(dyi, av, dbl, dbl);
                        }
                    }
                }
            }
        });
    }

    /// <summary>Elementwise sum of two equally-shaped tensors (used for residual connections).</summary>
    public Tensor Add(Tensor a, Tensor b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols) throw new ArgumentException("Add shape mismatch");
        var y = new Tensor(a.Rows, a.Cols);
        TensorPrimitives.Add(a.Data, b.Data, y.Data);
        return Record(y, () =>
        {
            TensorPrimitives.Add(a.Grad, y.Grad, a.Grad);
            TensorPrimitives.Add(b.Grad, y.Grad, b.Grad);
        });
    }

    /// <summary>Adds a broadcast row vector <paramref name="bias"/> (<c>[1, Cols]</c>, trainable) to every row of <paramref name="x"/>.</summary>
    public Tensor AddRowVector(Tensor x, Tensor bias)
    {
        if (bias.Rows != 1 || bias.Cols != x.Cols) throw new ArgumentException("AddRowVector shape mismatch");
        var y = new Tensor(x.Rows, x.Cols);
        for (int i = 0; i < x.Rows; i++)
            TensorPrimitives.Add(x.Data.AsSpan(i * x.Cols, x.Cols), bias.Data, y.Data.AsSpan(i * x.Cols, x.Cols));
        return Record(y, () =>
        {
            TensorPrimitives.Add(x.Grad, y.Grad, x.Grad);
            for (int i = 0; i < x.Rows; i++)
                TensorPrimitives.Add(bias.Grad, y.Grad.AsSpan(i * x.Cols, x.Cols), bias.Grad);
        });
    }

    /// <summary>Scales a tensor by a constant (used for the 1/sqrt(head_dim) attention factor).</summary>
    public Tensor Scale(Tensor x, float s)
    {
        var y = new Tensor(x.Rows, x.Cols);
        TensorPrimitives.Multiply(x.Data, s, y.Data);
        return Record(y, () => TensorPrimitives.MultiplyAdd(y.Grad, s, x.Grad, x.Grad));
    }

    // ----- nonlinearities & norms ------------------------------------------------------------------

    /// <summary>Exact (erf) GELU, matching HuggingFace BERT's default <c>hidden_act="gelu"</c>.</summary>
    public Tensor Gelu(Tensor x)
    {
        var y = new Tensor(x.Rows, x.Cols);
        for (int i = 0; i < x.Length; i++) y.Data[i] = (float)GeluScalar(x.Data[i]);
        return Record(y, () =>
        {
            for (int i = 0; i < x.Length; i++) x.Grad[i] += y.Grad[i] * (float)GeluPrime(x.Data[i]);
        });
    }

    /// <summary>Row-wise LayerNorm over the last dim with frozen <paramref name="gamma"/>/<paramref name="beta"/> (<c>[Cols]</c>).</summary>
    public Tensor LayerNorm(Tensor x, float[] gamma, float[] beta, float eps)
    {
        int m = x.Rows, c = x.Cols;
        var y     = new Tensor(m, c);
        var xhat  = new float[m * c]; // cached for backward
        var invStd = new float[m];
        for (int i = 0; i < m; i++)
        {
            var xi = x.Data.AsSpan(i * c, c);
            float mean = 0f;
            for (int j = 0; j < c; j++) mean += xi[j];
            mean /= c;
            float var = 0f;
            for (int j = 0; j < c; j++) { float d = xi[j] - mean; var += d * d; }
            var /= c;
            float inv = 1f / MathF.Sqrt(var + eps);
            invStd[i] = inv;
            int b = i * c;
            for (int j = 0; j < c; j++)
            {
                float xh = (xi[j] - mean) * inv;
                xhat[b + j]   = xh;
                y.Data[b + j] = gamma[j] * xh + beta[j];
            }
        }
        return Record(y, () =>
        {
            for (int i = 0; i < m; i++)
            {
                int b = i * c;
                var dy = y.Grad.AsSpan(b, c);
                // dxhat = dy*gamma ; then dx = inv*(dxhat - mean(dxhat) - xhat*mean(dxhat*xhat))
                float meanDxhat = 0f, meanDxhatXhat = 0f;
                for (int j = 0; j < c; j++)
                {
                    float dxh = dy[j] * gamma[j];
                    meanDxhat     += dxh;
                    meanDxhatXhat += dxh * xhat[b + j];
                }
                meanDxhat     /= c;
                meanDxhatXhat /= c;
                float inv = invStd[i];
                for (int j = 0; j < c; j++)
                {
                    float dxh = dy[j] * gamma[j];
                    x.Grad[b + j] += inv * (dxh - meanDxhat - xhat[b + j] * meanDxhatXhat);
                }
            }
        });
    }

    /// <summary>Row-wise softmax over the last dim.</summary>
    public Tensor Softmax(Tensor x)
    {
        int m = x.Rows, c = x.Cols;
        var y = new Tensor(m, c);
        for (int i = 0; i < m; i++)
        {
            var xi = x.Data.AsSpan(i * c, c);
            var yi = y.Data.AsSpan(i * c, c);
            float max = float.NegativeInfinity;
            for (int j = 0; j < c; j++) if (xi[j] > max) max = xi[j];
            float sum = 0f;
            for (int j = 0; j < c; j++) { float e = MathF.Exp(xi[j] - max); yi[j] = e; sum += e; }
            float inv = 1f / sum;
            for (int j = 0; j < c; j++) yi[j] *= inv;
        }
        return Record(y, () =>
        {
            for (int i = 0; i < m; i++)
            {
                var yi  = y.Data.AsSpan(i * c, c);
                var dyi = y.Grad.AsSpan(i * c, c);
                float dot = TensorPrimitives.Dot(yi, dyi);
                var dxi = x.Grad.AsSpan(i * c, c);
                for (int j = 0; j < c; j++) dxi[j] += yi[j] * (dyi[j] - dot);
            }
        });
    }

    // ----- shape ops ------------------------------------------------------------------------------

    /// <summary>Selects columns <c>[start, start+len)</c> of every row (per-head slicing of Q/K/V).</summary>
    public Tensor SliceCols(Tensor x, int start, int len)
    {
        int m = x.Rows, c = x.Cols;
        var y = new Tensor(m, len);
        for (int i = 0; i < m; i++)
            x.Data.AsSpan(i * c + start, len).CopyTo(y.Data.AsSpan(i * len, len));
        return Record(y, () =>
        {
            for (int i = 0; i < m; i++)
            {
                var dst = x.Grad.AsSpan(i * c + start, len);
                TensorPrimitives.Add(dst, y.Grad.AsSpan(i * len, len), dst);
            }
        });
    }

    /// <summary>Concatenates equally-tall tensors along columns (re-assembling attention heads).</summary>
    public Tensor ConcatCols(Tensor[] parts)
    {
        int m = parts[0].Rows;
        int total = 0;
        foreach (var p in parts) total += p.Cols;
        var y = new Tensor(m, total);
        int off = 0;
        foreach (var p in parts)
        {
            for (int i = 0; i < m; i++)
                p.Data.AsSpan(i * p.Cols, p.Cols).CopyTo(y.Data.AsSpan(i * total + off, p.Cols));
            off += p.Cols;
        }
        return Record(y, () =>
        {
            int o = 0;
            foreach (var p in parts)
            {
                for (int i = 0; i < m; i++)
                {
                    var dst = p.Grad.AsSpan(i * p.Cols, p.Cols);
                    TensorPrimitives.Add(dst, y.Grad.AsSpan(i * total + o, p.Cols), dst);
                }
                o += p.Cols;
            }
        });
    }

    // ----- pooling & normalization ----------------------------------------------------------------

    /// <summary>Mean-pools all rows into a single <c>[1, Cols]</c> row (BERT mean pooling; no padding rows are passed).</summary>
    public Tensor MeanRows(Tensor x)
    {
        int m = x.Rows, c = x.Cols;
        var y = new Tensor(1, c);
        for (int i = 0; i < m; i++)
            TensorPrimitives.Add(y.Data, x.Data.AsSpan(i * c, c), y.Data);
        TensorPrimitives.Divide(y.Data, m, y.Data);
        return Record(y, () =>
        {
            float inv = 1f / m;
            for (int i = 0; i < m; i++)
            {
                var dst = x.Grad.AsSpan(i * c, c);
                TensorPrimitives.MultiplyAdd(y.Grad, inv, dst, dst);
            }
        });
    }

    /// <summary>Returns row <paramref name="idx"/> as a <c>[1, Cols]</c> tensor (CLS pooling).</summary>
    public Tensor Row(Tensor x, int idx)
    {
        int c = x.Cols;
        var y = new Tensor(1, c);
        x.Data.AsSpan(idx * c, c).CopyTo(y.Data);
        return Record(y, () =>
        {
            var dst = x.Grad.AsSpan(idx * c, c);
            TensorPrimitives.Add(dst, y.Grad, dst);
        });
    }

    /// <summary>L2-normalizes a single row vector (<c>[1, Cols]</c>) to unit length.</summary>
    public Tensor L2Normalize(Tensor x, float eps = 1e-12f)
    {
        int c = x.Cols;
        var y = new Tensor(1, c);
        float norm = MathF.Max(MathF.Sqrt(TensorPrimitives.Dot(x.Data, x.Data)), eps);
        float inv = 1f / norm;
        TensorPrimitives.Multiply(x.Data, inv, y.Data);
        return Record(y, () =>
        {
            // dx = (dY - (dY·u) u) / norm
            float dyu = TensorPrimitives.Dot(y.Grad, y.Data);
            for (int j = 0; j < c; j++)
                x.Grad[j] += (y.Grad[j] - dyu * y.Data[j]) * inv;
        });
    }

    // ----- scalar helpers -------------------------------------------------------------------------

    private const double InvSqrt2   = 0.7071067811865476;
    private const double InvSqrt2Pi = 0.3989422804014327;

    private static double GeluScalar(double x) => 0.5 * x * (1.0 + Erf(x * InvSqrt2));

    private static double GeluPrime(double x)
        => 0.5 * (1.0 + Erf(x * InvSqrt2)) + x * InvSqrt2Pi * Math.Exp(-0.5 * x * x);

    /// <summary>Abramowitz &amp; Stegun 7.1.26 error-function approximation (|err| &lt; 1.5e-7).</summary>
    private static double Erf(double x)
    {
        int sign = x < 0 ? -1 : 1;
        x = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return sign * y;
    }
}
