using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Harrier.Small.Pure.Numerics;
using SentenceTransformers.Training.Autograd;

namespace SentenceTransformers.Harrier.Small.Pure.Training;

/// <summary>
/// An autograd-based Gemma3 forward pass (harrier-oss-v1-270m) built on the shared
/// <see cref="Graph"/> engine, so real weight-space LoRA adapters injected into the attention/MLP
/// projections get exact gradients by replaying the tape. This is the <b>trainable</b> counterpart to the
/// heavily-optimized inference <c>Gemma3Model</c>: it trades raw speed for a full, differentiable
/// forward+backward in pure C#. All base weights are frozen (attention/MLP projections widened to fp32 at
/// load; the embedding table stays bfloat16 and is widened per token); only the LoRA factors train.
/// </summary>
internal sealed class Gemma3LoraModel
{
    private readonly Gemma3Config _cfg;
    private readonly ushort[]     _embed;      // [vocab, hidden] bf16
    private readonly float        _embedScale;
    private readonly Layer[]      _layers;
    private readonly float[]      _finalNorm;

    public Gemma3Config Config => _cfg;
    public int Dimension => _cfg.HiddenSize;

    internal sealed class Layer
    {
        public float[] InputNorm, PostAttnNorm, PreFfNorm, PostFfNorm;
        public float[] Wq, Wk, Wv, Wo;   // projections (no bias in Gemma)
        public float[] QNorm, KNorm;     // per-head Q/K RMSNorm
        public float[] Wgate, Wup, Wdown;
    }

    private Gemma3LoraModel(Gemma3Config cfg, ushort[] embed, Layer[] layers, float[] finalNorm)
    {
        _cfg = cfg; _embed = embed; _layers = layers; _finalNorm = finalNorm;
        _embedScale = FloatConversions.RoundToBFloat16(MathF.Sqrt(cfg.HiddenSize));
    }

    public static async Task<Gemma3LoraModel> LoadAsync(string safetensorsPath, CancellationToken ct = default)
    {
        var cfg = new Gemma3Config();
        var st  = await SafeTensors.LoadAsync(safetensorsPath, ct).ConfigureAwait(false);
        string prefix = st.Contains("embed_tokens.weight") ? "" : "model.";
        float[] R(string n) => st.ReadFloat(prefix + n);

        var layers = new Layer[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            ct.ThrowIfCancellationRequested();
            string p = $"{prefix}layers.{i}.";
            layers[i] = new Layer
            {
                InputNorm    = R($"{prefix}layers.{i}.input_layernorm.weight"),
                PostAttnNorm = R($"{prefix}layers.{i}.post_attention_layernorm.weight"),
                PreFfNorm    = R($"{prefix}layers.{i}.pre_feedforward_layernorm.weight"),
                PostFfNorm   = R($"{prefix}layers.{i}.post_feedforward_layernorm.weight"),
                Wq    = st.ReadFloat($"{p}self_attn.q_proj.weight"),
                Wk    = st.ReadFloat($"{p}self_attn.k_proj.weight"),
                Wv    = st.ReadFloat($"{p}self_attn.v_proj.weight"),
                Wo    = st.ReadFloat($"{p}self_attn.o_proj.weight"),
                QNorm = st.ReadFloat($"{p}self_attn.q_norm.weight"),
                KNorm = st.ReadFloat($"{p}self_attn.k_norm.weight"),
                Wgate = st.ReadFloat($"{p}mlp.gate_proj.weight"),
                Wup   = st.ReadFloat($"{p}mlp.up_proj.weight"),
                Wdown = st.ReadFloat($"{p}mlp.down_proj.weight"),
            };
        }
        return new Gemma3LoraModel(cfg, st.ReadRaw16(prefix + "embed_tokens.weight"), layers, R("norm.weight"));
    }

    /// <summary>Tiny random model for gradient-checking (no download).</summary>
    internal static Gemma3LoraModel CreateRandom(Gemma3Config cfg, int seed)
    {
        var rng = new Random(seed);
        float[] Rand(int n, float s) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * s; return a; }
        ushort[] RandBf(int n, float s) { var a = new ushort[n]; for (int i = 0; i < n; i++) a[i] = FloatConversions.SingleToBFloat16((float)(rng.NextDouble() * 2 - 1) * s); return a; }

        int h = cfg.HiddenSize, inter = cfg.IntermediateSize, qo = cfg.QProjOut, kv = cfg.KvProjOut, hd = cfg.HeadDim;
        var layers = new Layer[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++)
            layers[i] = new Layer
            {
                InputNorm = Rand(h, 0.1f), PostAttnNorm = Rand(h, 0.1f), PreFfNorm = Rand(h, 0.1f), PostFfNorm = Rand(h, 0.1f),
                Wq = Rand(qo * h, 0.1f), Wk = Rand(kv * h, 0.1f), Wv = Rand(kv * h, 0.1f), Wo = Rand(h * qo, 0.1f),
                QNorm = Rand(hd, 0.1f), KNorm = Rand(hd, 0.1f),
                Wgate = Rand(inter * h, 0.1f), Wup = Rand(inter * h, 0.1f), Wdown = Rand(h * inter, 0.1f),
            };
        return new Gemma3LoraModel(cfg, RandBf(cfg.VocabSize * h, 0.1f), layers, Rand(h, 0.1f));
    }

    /// <summary>Full forward for one token sequence, returning the last-token pooled vector <b>before</b>
    /// L2 normalization. <paramref name="rt"/> = null runs the frozen base model.</summary>
    public Tensor Forward(Graph g, int[] tokenIds, GemmaLoraRuntime rt)
    {
        int seq = tokenIds.Length, h = _cfg.HiddenSize, hd = _cfg.HeadDim, heads = _cfg.NumHeads;
        float eps = _cfg.RmsNormEps, attnScale = _cfg.AttentionScale;

        // --- token embedding (* sqrt(hidden), bf16-rounded, matching the reference) ---
        var emb = new float[seq * h];
        for (int p = 0; p < seq; p++)
        {
            int tok = tokenIds[p], b = p * h, src = tok * h;
            for (int j = 0; j < h; j++) emb[b + j] = FloatConversions.BFloat16ToSingle(_embed[src + j]) * _embedScale;
        }
        var hidden = Graph.Leaf(emb, seq, h);

        // --- rotary tables ---
        var (cos, sin) = BuildRope(seq, hd, _cfg.RopeTheta);

        for (int l = 0; l < _cfg.NumLayers; l++)
        {
            var L = _layers[l];

            // ---- self attention (GQA: 4 query heads share 1 KV head) ----
            var normed = g.RmsNorm(hidden, L.InputNorm, eps);
            var q = Lin(g, normed, L.Wq, _cfg.QProjOut,  h, rt, l, 0);
            var k = Lin(g, normed, L.Wk, _cfg.KvProjOut, h, rt, l, 1);
            var v = Lin(g, normed, L.Wv, _cfg.KvProjOut, h, rt, l, 2);

            // shared KV head: per-head RMSNorm on K, RoPE on K; V as-is.
            var kh = g.Rope(g.RmsNorm(g.SliceCols(k, 0, hd), L.KNorm, eps), cos, sin);
            var vh = g.SliceCols(v, 0, hd);

            var ctxHeads = new Tensor[heads];
            for (int head = 0; head < heads; head++)
            {
                var qh = g.Rope(g.RmsNorm(g.SliceCols(q, head * hd, hd), L.QNorm, eps), cos, sin);
                var scores = g.Scale(g.MatMul(qh, kh, transposeB: true), attnScale); // [seq,seq]
                var probs  = g.SoftmaxCausal(scores);
                ctxHeads[head] = g.MatMul(probs, vh, transposeB: false);             // [seq,hd]
            }
            var ctx = g.ConcatCols(ctxHeads);                                        // [seq, QProjOut]

            var attn = Lin(g, ctx, L.Wo, h, _cfg.QProjOut, rt, l, 3);
            attn = g.RmsNorm(attn, L.PostAttnNorm, eps);                             // post-attention norm
            hidden = g.Add(hidden, attn);                                            // residual

            // ---- feed forward (GeGLU) ----
            var ffIn = g.RmsNorm(hidden, L.PreFfNorm, eps);
            var gate = Lin(g, ffIn, L.Wgate, _cfg.IntermediateSize, h, rt, l, 4);
            var up   = Lin(g, ffIn, L.Wup,   _cfg.IntermediateSize, h, rt, l, 5);
            var ff   = g.Mul(g.GeluTanh(gate), up);                                  // gelu_tanh(gate) * up
            var down = Lin(g, ff, L.Wdown, h, _cfg.IntermediateSize, rt, l, 6);
            down = g.RmsNorm(down, L.PostFfNorm, eps);
            hidden = g.Add(hidden, down);
        }

        // ---- final norm + last-token pooling ----
        var pooled = g.RmsNorm(g.Row(hidden, seq - 1), _finalNorm, eps);
        if (rt?.OutBias != null) pooled = g.AddRowVector(pooled, rt.OutBias);
        return pooled;
    }

    private static Tensor Lin(Graph g, Tensor x, float[] w, int outDim, int inDim, GemmaLoraRuntime rt, int layer, int t)
    {
        Tensor a = null, b = null; float scale = 1f;
        if (rt != null && (rt.Targets & GemmaLoraAdapter.Flag(t)) != 0) { a = rt.Aof(layer, t); b = rt.Bof(layer, t); scale = rt.Scale; }
        return g.LoraLinear(x, w, null, outDim, inDim, a, b, scale);
    }

    private static (float[] cos, float[] sin) BuildRope(int seq, int headDim, float theta)
    {
        int half = headDim / 2;
        var cos = new float[seq * headDim];
        var sin = new float[seq * headDim];
        Span<float> invFreq = half <= 512 ? stackalloc float[half] : new float[half];
        for (int d = 0; d < half; d++) invFreq[d] = 1f / MathF.Pow(theta, (2f * d) / headDim);
        for (int p = 0; p < seq; p++)
        {
            int b = p * headDim;
            for (int d = 0; d < half; d++)
            {
                float angle = p * invFreq[d];
                float c = MathF.Cos(angle), s = MathF.Sin(angle);
                cos[b + d] = c; cos[b + d + half] = c;
                sin[b + d] = s; sin[b + d + half] = s;
            }
        }
        return (cos, sin);
    }
}
