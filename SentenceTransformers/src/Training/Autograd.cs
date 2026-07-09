using System.Numerics.Tensors;

namespace SentenceTransformers.Training.Autograd;

/// <summary>
/// A row-major 2-D tensor node in a reverse-mode automatic-differentiation graph. <see cref="Data"/>
/// holds the forward values; <see cref="Grad"/> accumulates <c>dLoss/dThis</c> during the backward pass.
///
/// <para>This tiny engine lets a transformer forward pass be written once, as a sequence of
/// <see cref="Graph"/> ops, so the exact gradients w.r.t. the (few) trainable LoRA parameters fall out
/// automatically — no hand-derived backprop, and no autodiff framework dependency. It is deliberately
/// tensor-level (nodes are whole matrices, not scalars) so a full forward is only a few hundred nodes.
/// It is shared by the pure-C# BERT and Gemma3 LoRA trainers.</para>
/// </summary>
public sealed class Tensor
{
    public readonly float[] Data;
    public readonly float[] Grad;
    public readonly int     Rows;
    public readonly int     Cols;

    public int Length => Rows * Cols;

    public Tensor(int rows, int cols)
    {
        Rows = rows; Cols = cols;
        Data = new float[rows * cols];
        Grad = new float[rows * cols];
    }

    public Tensor(float[] data, int rows, int cols)
    {
        if (data.Length != rows * cols) throw new ArgumentException("data length does not match rows*cols");
        Rows = rows; Cols = cols; Data = data;
        Grad = new float[rows * cols];
    }

    public void ClearGrad() => Array.Clear(Grad);
}

/// <summary>
/// Records the ops that produced a set of <see cref="Tensor"/>s so <see cref="Backward"/> can replay
/// their local vector-Jacobian products in reverse. Frozen weights are passed as plain <c>float[]</c> and
/// never enter the graph, so no gradient is computed or stored for them — only the activations and the
/// trainable LoRA tensors carry grad (the LoRA efficiency win).
/// </summary>
public sealed class Graph
{
    private readonly List<Action> _tape = new(512);

    public void Backward()
    {
        for (int i = _tape.Count - 1; i >= 0; i--) _tape[i]();
    }

    private Tensor Record(Tensor y, Action backward) { _tape.Add(backward); return y; }

    public static Tensor Leaf(float[] data, int rows, int cols) => new Tensor(data, rows, cols);

    // ----- linear algebra --------------------------------------------------------------------------

    /// <summary>LoRA-augmented linear: <c>y = x·Wᵀ + b + scale·(x·Aᵀ)·Bᵀ</c>. <paramref name="w"/>
    /// (<c>[outDim,inDim]</c>) and <paramref name="bias"/> (<c>[outDim]</c>, may be null) are frozen;
    /// <paramref name="a"/> (<c>[rank,inDim]</c>) / <paramref name="b_"/> (<c>[outDim,rank]</c>) are the
    /// trainable low-rank factors (null = plain frozen projection).</summary>
    public Tensor LoraLinear(Tensor x, float[] w, float[] bias, int outDim, int inDim, Tensor a, Tensor b_, float scale)
    {
        int m = x.Rows;
        if (x.Cols != inDim) throw new ArgumentException("LoraLinear input width mismatch");
        var y = new Tensor(m, outDim);

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
        float[] h = null;
        if (a != null)
        {
            h = new float[m * rank];
            for (int i = 0; i < m; i++)
            {
                var xi = x.Data.AsSpan(i * inDim, inDim);
                for (int r = 0; r < rank; r++) h[i * rank + r] = TensorPrimitives.Dot(xi, a.Data.AsSpan(r * inDim, inDim));
                int yo = i * outDim;
                var hi = h.AsSpan(i * rank, rank);
                for (int o = 0; o < outDim; o++) y.Data[yo + o] += scale * TensorPrimitives.Dot(hi, b_.Data.AsSpan(o * rank, rank));
            }
        }

        return Record(y, () =>
        {
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

            var dh = new float[m * rank];
            for (int i = 0; i < m; i++)
            {
                var dyi = y.Grad.AsSpan(i * outDim, outDim);
                var dhi = dh.AsSpan(i * rank, rank);
                var hi  = h.AsSpan(i * rank, rank);
                for (int o = 0; o < outDim; o++)
                {
                    float g = scale * dyi[o];
                    if (g == 0f) continue;
                    var bo  = b_.Data.AsSpan(o * rank, rank);
                    var dbo = b_.Grad.AsSpan(o * rank, rank);
                    for (int r = 0; r < rank; r++) { dhi[r] += g * bo[r]; dbo[r] += g * hi[r]; }
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
                    TensorPrimitives.MultiplyAdd(xi, g, a.Grad.AsSpan(r * inDim, inDim), a.Grad.AsSpan(r * inDim, inDim));
                    TensorPrimitives.MultiplyAdd(a.Data.AsSpan(r * inDim, inDim), g, dxi, dxi);
                }
            }
        });
    }

    /// <summary>Matrix product. transposeB=false: <c>[m,k]·[k,n]</c>; true: <c>[m,k]·([n,k])ᵀ=[m,n]</c>.</summary>
    public Tensor MatMul(Tensor aT, Tensor bT, bool transposeB)
    {
        int m = aT.Rows, k = aT.Cols;
        int n = transposeB ? bT.Rows : bT.Cols;
        if (transposeB) { if (bT.Cols != k) throw new ArgumentException("MatMul dim mismatch (transB)"); }
        else            { if (bT.Rows != k) throw new ArgumentException("MatMul dim mismatch"); }

        var y = new Tensor(m, n);
        var A = aT.Data; var B = bT.Data; var Y = y.Data;

        if (transposeB)
            for (int i = 0; i < m; i++)
            {
                var ai = A.AsSpan(i * k, k);
                for (int j = 0; j < n; j++) Y[i * n + j] = TensorPrimitives.Dot(ai, B.AsSpan(j * k, k));
            }
        else
            for (int i = 0; i < m; i++)
            {
                int yb = i * n;
                for (int l = 0; l < k; l++)
                {
                    float av = A[i * k + l];
                    if (av == 0f) continue;
                    var brow = B.AsSpan(l * n, n);
                    var yr = Y.AsSpan(yb, n);
                    TensorPrimitives.MultiplyAdd(brow, av, yr, yr);
                }
            }

        return Record(y, () =>
        {
            var dY = y.Grad; var dA = aT.Grad; var dB = bT.Grad;
            if (transposeB)
            {
                for (int i = 0; i < m; i++)
                {
                    var dyi = dY.AsSpan(i * n, n);
                    var dai = dA.AsSpan(i * k, k);
                    for (int j = 0; j < n; j++) { float g = dyi[j]; if (g != 0f) TensorPrimitives.MultiplyAdd(B.AsSpan(j * k, k), g, dai, dai); }
                }
                for (int j = 0; j < n; j++)
                {
                    var dbj = dB.AsSpan(j * k, k);
                    for (int i = 0; i < m; i++) { float g = dY[i * n + j]; if (g != 0f) TensorPrimitives.MultiplyAdd(A.AsSpan(i * k, k), g, dbj, dbj); }
                }
            }
            else
            {
                for (int i = 0; i < m; i++)
                {
                    var dyi = dY.AsSpan(i * n, n);
                    for (int l = 0; l < k; l++) dA[i * k + l] += TensorPrimitives.Dot(dyi, B.AsSpan(l * n, n));
                    var ai = A.AsSpan(i * k, k);
                    for (int l = 0; l < k; l++) { float av = ai[l]; if (av != 0f) { var dbl = dB.AsSpan(l * n, n); TensorPrimitives.MultiplyAdd(dyi, av, dbl, dbl); } }
                }
            }
        });
    }

    public Tensor Add(Tensor a, Tensor b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols) throw new ArgumentException("Add shape mismatch");
        var y = new Tensor(a.Rows, a.Cols);
        TensorPrimitives.Add(a.Data, b.Data, y.Data);
        return Record(y, () => { TensorPrimitives.Add(a.Grad, y.Grad, a.Grad); TensorPrimitives.Add(b.Grad, y.Grad, b.Grad); });
    }

    public Tensor AddRowVector(Tensor x, Tensor bias)
    {
        if (bias.Rows != 1 || bias.Cols != x.Cols) throw new ArgumentException("AddRowVector shape mismatch");
        var y = new Tensor(x.Rows, x.Cols);
        for (int i = 0; i < x.Rows; i++) TensorPrimitives.Add(x.Data.AsSpan(i * x.Cols, x.Cols), bias.Data, y.Data.AsSpan(i * x.Cols, x.Cols));
        return Record(y, () =>
        {
            TensorPrimitives.Add(x.Grad, y.Grad, x.Grad);
            for (int i = 0; i < x.Rows; i++) TensorPrimitives.Add(bias.Grad, y.Grad.AsSpan(i * x.Cols, x.Cols), bias.Grad);
        });
    }

    public Tensor Scale(Tensor x, float s)
    {
        var y = new Tensor(x.Rows, x.Cols);
        TensorPrimitives.Multiply(x.Data, s, y.Data);
        return Record(y, () => TensorPrimitives.MultiplyAdd(y.Grad, s, x.Grad, x.Grad));
    }

    // ----- nonlinearities & norms ------------------------------------------------------------------

    /// <summary>Exact (erf) GELU — HuggingFace BERT's default <c>hidden_act="gelu"</c>.</summary>
    public Tensor Gelu(Tensor x)
    {
        var y = new Tensor(x.Rows, x.Cols);
        for (int i = 0; i < x.Length; i++) y.Data[i] = (float)GeluErf(x.Data[i]);
        return Record(y, () => { for (int i = 0; i < x.Length; i++) x.Grad[i] += y.Grad[i] * (float)GeluErfPrime(x.Data[i]); });
    }

    /// <summary>Tanh-approximation GELU — Gemma's GeGLU activation.</summary>
    public Tensor GeluTanh(Tensor x)
    {
        var y = new Tensor(x.Rows, x.Cols);
        for (int i = 0; i < x.Length; i++) y.Data[i] = GeluTanhScalar(x.Data[i]);
        return Record(y, () => { for (int i = 0; i < x.Length; i++) x.Grad[i] += y.Grad[i] * GeluTanhPrime(x.Data[i]); });
    }

    /// <summary>Elementwise product of two equally-shaped tensors (GeGLU: gelu(gate) * up).</summary>
    public Tensor Mul(Tensor a, Tensor b)
    {
        if (a.Rows != b.Rows || a.Cols != b.Cols) throw new ArgumentException("Mul shape mismatch");
        var y = new Tensor(a.Rows, a.Cols);
        TensorPrimitives.Multiply(a.Data, b.Data, y.Data);
        return Record(y, () =>
        {
            for (int i = 0; i < a.Length; i++) { a.Grad[i] += y.Grad[i] * b.Data[i]; b.Grad[i] += y.Grad[i] * a.Data[i]; }
        });
    }

    /// <summary>Row-wise LayerNorm over the last dim with frozen <paramref name="gamma"/>/<paramref name="beta"/>.</summary>
    public Tensor LayerNorm(Tensor x, float[] gamma, float[] beta, float eps)
    {
        int m = x.Rows, c = x.Cols;
        var y = new Tensor(m, c);
        var xhat = new float[m * c];
        var invStd = new float[m];
        for (int i = 0; i < m; i++)
        {
            var xi = x.Data.AsSpan(i * c, c);
            float mean = 0f; for (int j = 0; j < c; j++) mean += xi[j]; mean /= c;
            float var = 0f; for (int j = 0; j < c; j++) { float d = xi[j] - mean; var += d * d; } var /= c;
            float inv = 1f / MathF.Sqrt(var + eps); invStd[i] = inv;
            int b = i * c;
            for (int j = 0; j < c; j++) { float xh = (xi[j] - mean) * inv; xhat[b + j] = xh; y.Data[b + j] = gamma[j] * xh + beta[j]; }
        }
        return Record(y, () =>
        {
            for (int i = 0; i < m; i++)
            {
                int b = i * c; var dy = y.Grad.AsSpan(b, c);
                float meanDxhat = 0f, meanDxhatXhat = 0f;
                for (int j = 0; j < c; j++) { float dxh = dy[j] * gamma[j]; meanDxhat += dxh; meanDxhatXhat += dxh * xhat[b + j]; }
                meanDxhat /= c; meanDxhatXhat /= c;
                float inv = invStd[i];
                for (int j = 0; j < c; j++) { float dxh = dy[j] * gamma[j]; x.Grad[b + j] += inv * (dxh - meanDxhat - xhat[b + j] * meanDxhatXhat); }
            }
        });
    }

    /// <summary>Gemma RMSNorm over the last dim: <c>y = x / sqrt(mean(x²)+eps) * (1 + weight)</c>
    /// (<paramref name="addOne"/> selects the <c>(1+w)</c> Gemma form; false uses plain <c>w</c>).</summary>
    public Tensor RmsNorm(Tensor x, float[] weight, float eps, bool addOne = true)
    {
        int m = x.Rows, c = x.Cols;
        var y = new Tensor(m, c);
        var invRms = new float[m];
        for (int i = 0; i < m; i++)
        {
            var xi = x.Data.AsSpan(i * c, c);
            float ss = 0f; for (int j = 0; j < c; j++) ss += xi[j] * xi[j];
            float inv = 1f / MathF.Sqrt(ss / c + eps); invRms[i] = inv;
            int b = i * c;
            for (int j = 0; j < c; j++) y.Data[b + j] = xi[j] * inv * (addOne ? 1f + weight[j] : weight[j]);
        }
        return Record(y, () =>
        {
            for (int i = 0; i < m; i++)
            {
                int b = i * c; var xi = x.Data.AsSpan(b, c); var dy = y.Grad.AsSpan(b, c);
                float inv = invRms[i];
                // g_j = dy_j * (1+w_j) ; let s = x/||rms||. dL/dx = inv*(g - (mean(g·x)/(rms²·... )) ) — derive:
                // y = x * inv * wj, inv = (ss/c+eps)^-1/2. d inv/d x_k = -inv³/c * x_k.
                // dL/dx_j = sum_k dy_k*wk*( inv*[j==k] + x_k * dinv/dx_j )
                //         = g_j*inv + (sum_k dy_k*wk*x_k) * (-inv³/c) * x_j
                float dotGX = 0f;
                for (int j = 0; j < c; j++) { float wj = addOne ? 1f + weight[j] : weight[j]; dotGX += dy[j] * wj * xi[j]; }
                float coef = -inv * inv * inv / c * dotGX;
                for (int j = 0; j < c; j++)
                {
                    float wj = addOne ? 1f + weight[j] : weight[j];
                    x.Grad[b + j] += dy[j] * wj * inv + coef * xi[j];
                }
            }
        });
    }

    /// <summary>Row-wise softmax over the last dim.</summary>
    public Tensor Softmax(Tensor x) => SoftmaxImpl(x, causal: false);

    /// <summary>Causal row-wise softmax: row i attends only to columns 0..i (upper triangle masked).</summary>
    public Tensor SoftmaxCausal(Tensor x) => SoftmaxImpl(x, causal: true);

    private Tensor SoftmaxImpl(Tensor x, bool causal)
    {
        int m = x.Rows, c = x.Cols;
        var y = new Tensor(m, c);
        for (int i = 0; i < m; i++)
        {
            int limit = causal ? Math.Min(i + 1, c) : c;
            var xi = x.Data.AsSpan(i * c, c);
            var yi = y.Data.AsSpan(i * c, c);
            float max = float.NegativeInfinity;
            for (int j = 0; j < limit; j++) if (xi[j] > max) max = xi[j];
            float sum = 0f;
            for (int j = 0; j < limit; j++) { float e = MathF.Exp(xi[j] - max); yi[j] = e; sum += e; }
            float inv = 1f / sum;
            for (int j = 0; j < limit; j++) yi[j] *= inv;
            // columns >= limit stay 0
        }
        return Record(y, () =>
        {
            for (int i = 0; i < m; i++)
            {
                int limit = causal ? Math.Min(i + 1, c) : c;
                var yi = y.Data.AsSpan(i * c, c);
                var dyi = y.Grad.AsSpan(i * c, c);
                float dot = 0f; for (int j = 0; j < limit; j++) dot += yi[j] * dyi[j];
                var dxi = x.Grad.AsSpan(i * c, c);
                for (int j = 0; j < limit; j++) dxi[j] += yi[j] * (dyi[j] - dot);
            }
        });
    }

    /// <summary>Applies rotary position embeddings to a <c>[seq, headDim]</c> tensor using the precomputed
    /// (frozen) <paramref name="cos"/>/<paramref name="sin"/> tables (each <c>[seq*headDim]</c>, halves duplicated).</summary>
    public Tensor Rope(Tensor x, float[] cos, float[] sin)
    {
        int seq = x.Rows, hd = x.Cols, half = hd / 2;
        var y = new Tensor(seq, hd);
        for (int p = 0; p < seq; p++)
        {
            int b = p * hd;
            for (int d = 0; d < half; d++)
            {
                float x1 = x.Data[b + d], x2 = x.Data[b + d + half];
                float c = cos[b + d], s = sin[b + d];
                y.Data[b + d] = x1 * c - x2 * s;
                y.Data[b + d + half] = x2 * c + x1 * s;
            }
        }
        return Record(y, () =>
        {
            for (int p = 0; p < seq; p++)
            {
                int b = p * hd;
                for (int d = 0; d < half; d++)
                {
                    float g1 = y.Grad[b + d], g2 = y.Grad[b + d + half];
                    float c = cos[b + d], s = sin[b + d];
                    // transpose of the rotation
                    x.Grad[b + d]        += g1 * c + g2 * s;
                    x.Grad[b + d + half] += -g1 * s + g2 * c;
                }
            }
        });
    }

    // ----- shape ops ------------------------------------------------------------------------------

    public Tensor SliceCols(Tensor x, int start, int len)
    {
        int m = x.Rows, c = x.Cols;
        var y = new Tensor(m, len);
        for (int i = 0; i < m; i++) x.Data.AsSpan(i * c + start, len).CopyTo(y.Data.AsSpan(i * len, len));
        return Record(y, () =>
        {
            for (int i = 0; i < m; i++) { var dst = x.Grad.AsSpan(i * c + start, len); TensorPrimitives.Add(dst, y.Grad.AsSpan(i * len, len), dst); }
        });
    }

    public Tensor ConcatCols(Tensor[] parts)
    {
        int m = parts[0].Rows, total = 0;
        foreach (var p in parts) total += p.Cols;
        var y = new Tensor(m, total);
        int off = 0;
        foreach (var p in parts) { for (int i = 0; i < m; i++) p.Data.AsSpan(i * p.Cols, p.Cols).CopyTo(y.Data.AsSpan(i * total + off, p.Cols)); off += p.Cols; }
        return Record(y, () =>
        {
            int o = 0;
            foreach (var p in parts) { for (int i = 0; i < m; i++) { var dst = p.Grad.AsSpan(i * p.Cols, p.Cols); TensorPrimitives.Add(dst, y.Grad.AsSpan(i * total + o, p.Cols), dst); } o += p.Cols; }
        });
    }

    // ----- pooling & normalization ----------------------------------------------------------------

    public Tensor MeanRows(Tensor x)
    {
        int m = x.Rows, c = x.Cols;
        var y = new Tensor(1, c);
        for (int i = 0; i < m; i++) TensorPrimitives.Add(y.Data, x.Data.AsSpan(i * c, c), y.Data);
        TensorPrimitives.Divide(y.Data, m, y.Data);
        return Record(y, () => { float inv = 1f / m; for (int i = 0; i < m; i++) { var dst = x.Grad.AsSpan(i * c, c); TensorPrimitives.MultiplyAdd(y.Grad, inv, dst, dst); } });
    }

    public Tensor Row(Tensor x, int idx)
    {
        int c = x.Cols;
        var y = new Tensor(1, c);
        x.Data.AsSpan(idx * c, c).CopyTo(y.Data);
        return Record(y, () => { var dst = x.Grad.AsSpan(idx * c, c); TensorPrimitives.Add(dst, y.Grad, dst); });
    }

    public Tensor L2Normalize(Tensor x, float eps = 1e-12f)
    {
        int c = x.Cols;
        var y = new Tensor(1, c);
        float norm = MathF.Max(MathF.Sqrt(TensorPrimitives.Dot(x.Data, x.Data)), eps);
        float inv = 1f / norm;
        TensorPrimitives.Multiply(x.Data, inv, y.Data);
        return Record(y, () =>
        {
            float dyu = TensorPrimitives.Dot(y.Grad, y.Data);
            for (int j = 0; j < c; j++) x.Grad[j] += (y.Grad[j] - dyu * y.Data[j]) * inv;
        });
    }

    // ----- scalar helpers -------------------------------------------------------------------------

    private const double InvSqrt2   = 0.7071067811865476;
    private const double InvSqrt2Pi = 0.3989422804014327;
    private const float  TanhC      = 0.7978845608028654f; // sqrt(2/pi)

    private static double GeluErf(double x) => 0.5 * x * (1.0 + Erf(x * InvSqrt2));
    private static double GeluErfPrime(double x) => 0.5 * (1.0 + Erf(x * InvSqrt2)) + x * InvSqrt2Pi * Math.Exp(-0.5 * x * x);

    private static float GeluTanhScalar(float x)
    {
        float inner = TanhC * (x + 0.044715f * x * x * x);
        return 0.5f * x * (1f + MathF.Tanh(inner));
    }

    private static float GeluTanhPrime(float x)
    {
        float x2 = x * x;
        float inner = TanhC * (x + 0.044715f * x * x2);
        float t = MathF.Tanh(inner);
        float dInner = TanhC * (1f + 3f * 0.044715f * x2);
        float sech2 = 1f - t * t;
        return 0.5f * (1f + t) + 0.5f * x * sech2 * dInner;
    }

    private static double Erf(double x)
    {
        int sign = x < 0 ? -1 : 1; x = Math.Abs(x);
        double t = 1.0 / (1.0 + 0.3275911 * x);
        double y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.Exp(-x * x);
        return sign * y;
    }
}
