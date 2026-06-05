using SentenceTransformers.Harrier.Small.Pure.Numerics;

namespace SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// Pure-managed implementation of the harrier-oss-v1-270m (<c>Gemma3TextModel</c>) forward pass.
///
/// The model is a decoder-only transformer with causal attention. For embedding we run a single
/// forward pass per input sequence (no KV cache, no padding) and read the hidden state of the final
/// token - the reference uses last-token pooling, and because the tokenizer appends <c>&lt;eos&gt;</c>
/// the final token is always the EOS aggregation token.
///
/// Per Gemma3, each decoder layer is a four-norm sandwich:
/// <code>
///   h = h + post_attention_layernorm( attn( input_layernorm(h) ) )
///   h = h + post_feedforward_layernorm( mlp( pre_feedforward_layernorm(h) ) )
/// </code>
/// Attention uses grouped-query attention (4 query heads, 1 KV head), per-head Q/K RMSNorm, rotary
/// position embeddings (theta = 1e6) and a logit scale of <c>query_pre_attn_scalar^-0.5</c> = 1/16.
/// The MLP is a GeGLU with the tanh GELU approximation. All arithmetic is float32; weights are widened
/// from the checkpoint's bfloat16 at load time.
/// </summary>
internal sealed class Gemma3Model
{
    private readonly Gemma3Config _cfg;

    // Token embedding kept in its native bfloat16 storage (335 MB) - rows are widened one token at a
    // time, so there is no need to materialize the full 167M-parameter table as float32.
    private readonly ushort[] _embedTokens;
    private readonly float _embedScale;

    private readonly Layer[] _layers;
    private readonly float[] _finalNorm;

    private sealed class Layer
    {
        public required float[] InputLayerNorm;
        public required float[] PostAttentionLayerNorm;
        public required float[] PreFeedforwardLayerNorm;
        public required float[] PostFeedforwardLayerNorm;
        public required IWeightMatrix QProj;
        public required IWeightMatrix KProj;
        public required IWeightMatrix VProj;
        public required IWeightMatrix OProj;
        public required float[] QNorm;
        public required float[] KNorm;
        public required IWeightMatrix GateProj;
        public required IWeightMatrix UpProj;
        public required IWeightMatrix DownProj;
    }

    private Gemma3Model(Gemma3Config cfg, ushort[] embedTokens, Layer[] layers, float[] finalNorm)
    {
        _cfg = cfg;
        _embedTokens = embedTokens;
        _layers = layers;
        _finalNorm = finalNorm;
        // The reference scales embeddings by sqrt(hidden_size) rounded to the embedding's bfloat16
        // dtype; reproduce that rounding so our activations track the reference exactly.
        _embedScale = FloatConversions.RoundToBFloat16(MathF.Sqrt(cfg.HiddenSize));
    }

    public static Gemma3Model Load(string safetensorsPath, Gemma3Config cfg, Quantization quantization = Quantization.None)
    {
        var st = SafeTensors.Load(safetensorsPath);

        // Some exports prefix every tensor with "model."; tolerate both.
        string prefix = st.Contains("embed_tokens.weight") ? "" : "model.";

        var embed = st.ReadRaw16(prefix + "embed_tokens.weight");

        var layers = new Layer[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"{prefix}layers.{i}.";
            // The attention/MLP projections dominate weight size and compute, so they carry the
            // chosen quantization; the tiny per-channel norm vectors stay in float32.
            IWeightMatrix Proj(string name, int outDim, int inDim)
                => IWeightMatrix.Create(st.ReadFloat(p + name), outDim, inDim, quantization);

            layers[i] = new Layer
            {
                InputLayerNorm = st.ReadFloat(p + "input_layernorm.weight"),
                PostAttentionLayerNorm = st.ReadFloat(p + "post_attention_layernorm.weight"),
                PreFeedforwardLayerNorm = st.ReadFloat(p + "pre_feedforward_layernorm.weight"),
                PostFeedforwardLayerNorm = st.ReadFloat(p + "post_feedforward_layernorm.weight"),
                QProj = Proj("self_attn.q_proj.weight", cfg.QProjOut, cfg.HiddenSize),
                KProj = Proj("self_attn.k_proj.weight", cfg.KvProjOut, cfg.HiddenSize),
                VProj = Proj("self_attn.v_proj.weight", cfg.KvProjOut, cfg.HiddenSize),
                OProj = Proj("self_attn.o_proj.weight", cfg.HiddenSize, cfg.QProjOut),
                QNorm = st.ReadFloat(p + "self_attn.q_norm.weight"),
                KNorm = st.ReadFloat(p + "self_attn.k_norm.weight"),
                GateProj = Proj("mlp.gate_proj.weight", cfg.IntermediateSize, cfg.HiddenSize),
                UpProj = Proj("mlp.up_proj.weight", cfg.IntermediateSize, cfg.HiddenSize),
                DownProj = Proj("mlp.down_proj.weight", cfg.HiddenSize, cfg.IntermediateSize),
            };
        }

        var finalNorm = st.ReadFloat(prefix + "norm.weight");
        return new Gemma3Model(cfg, embed, layers, finalNorm);
    }

    /// <summary>Runs the full forward pass for one tokenized sequence and returns the (un-normalized)
    /// last-token hidden state - a <c>HiddenSize</c>-length embedding.</summary>
    public float[] Forward(ReadOnlySpan<int> tokenIds)
    {
        int seq = tokenIds.Length;
        int h = _cfg.HiddenSize;
        int headDim = _cfg.HeadDim;

        // --- token embedding (+ sqrt(hidden) scale) ---
        var hidden = new float[seq * h];
        for (int p = 0; p < seq; p++)
        {
            int tok = tokenIds[p];
            int srcBase = tok * h;
            int dstBase = p * h;
            for (int i = 0; i < h; i++)
            {
                hidden[dstBase + i] = FloatConversions.BFloat16ToSingle(_embedTokens[srcBase + i]) * _embedScale;
            }
        }

        // --- rotary tables ---
        var (cos, sin) = BuildRope(seq, headDim, _cfg.RopeTheta);

        // Scratch buffers reused across layers.
        var normed = new float[seq * h];
        var q = new float[seq * _cfg.QProjOut];
        var k = new float[seq * _cfg.KvProjOut];
        var v = new float[seq * _cfg.KvProjOut];
        var attnOut = new float[seq * _cfg.QProjOut];
        var attnProj = new float[seq * h];
        var gate = new float[seq * _cfg.IntermediateSize];
        var up = new float[seq * _cfg.IntermediateSize];
        var down = new float[seq * h];

        foreach (var layer in _layers)
        {
            // ---- self attention ----
            Ops.RmsNorm(hidden, layer.InputLayerNorm, normed, seq, h, _cfg.RmsNormEps);
            layer.QProj.Multiply(normed, q, seq);
            layer.KProj.Multiply(normed, k, seq);
            layer.VProj.Multiply(normed, v, seq);

            ApplyHeadNorm(q, layer.QNorm, seq, _cfg.NumHeads, headDim);
            ApplyHeadNorm(k, layer.KNorm, seq, _cfg.NumKvHeads, headDim);

            ApplyRope(q, cos, sin, seq, _cfg.NumHeads, headDim);
            ApplyRope(k, cos, sin, seq, _cfg.NumKvHeads, headDim);

            Attention(q, k, v, attnOut, seq, headDim);

            layer.OProj.Multiply(attnOut, attnProj, seq);
            // post-attention norm then residual add
            Ops.RmsNorm(attnProj, layer.PostAttentionLayerNorm, attnProj, seq, h, _cfg.RmsNormEps);
            for (int i = 0; i < hidden.Length; i++)
            {
                hidden[i] += attnProj[i];
            }

            // ---- feed forward ----
            Ops.RmsNorm(hidden, layer.PreFeedforwardLayerNorm, normed, seq, h, _cfg.RmsNormEps);
            layer.GateProj.Multiply(normed, gate, seq);
            layer.UpProj.Multiply(normed, up, seq);
            Ops.GeGluInPlace(gate, up);
            layer.DownProj.Multiply(gate, down, seq);
            Ops.RmsNorm(down, layer.PostFeedforwardLayerNorm, down, seq, h, _cfg.RmsNormEps);
            for (int i = 0; i < hidden.Length; i++)
            {
                hidden[i] += down[i];
            }
        }

        // ---- final norm + last-token pooling ----
        var lastNormed = new float[h];
        Ops.RmsNorm(hidden.AsSpan((seq - 1) * h, h), _finalNorm, lastNormed, 1, h, _cfg.RmsNormEps);
        return lastNormed;
    }

    /// <summary>Applies Gemma's per-head Q/K RMSNorm (over <paramref name="headDim"/>) in place.</summary>
    private void ApplyHeadNorm(float[] x, float[] weight, int seq, int heads, int headDim)
    {
        int stride = heads * headDim;
        for (int s = 0; s < seq; s++)
        {
            for (int hh = 0; hh < heads; hh++)
            {
                int off = s * stride + hh * headDim;
                Ops.RmsNorm(x.AsSpan(off, headDim), weight, x.AsSpan(off, headDim), 1, headDim, _cfg.RmsNormEps);
            }
        }
    }

    /// <summary>Applies rotary position embeddings in place to every head of <paramref name="x"/>.</summary>
    private static void ApplyRope(float[] x, float[] cos, float[] sin, int seq, int heads, int headDim)
    {
        int half = headDim / 2;
        int stride = heads * headDim;
        for (int s = 0; s < seq; s++)
        {
            int cosBase = s * headDim;
            for (int hh = 0; hh < heads; hh++)
            {
                int baseIdx = s * stride + hh * headDim;
                for (int d = 0; d < half; d++)
                {
                    float x1 = x[baseIdx + d];
                    float x2 = x[baseIdx + d + half];
                    float c = cos[cosBase + d];
                    float sn = sin[cosBase + d];
                    x[baseIdx + d] = x1 * c - x2 * sn;
                    x[baseIdx + d + half] = x2 * c + x1 * sn;
                }
            }
        }
    }

    /// <summary>Causal grouped-query attention. The single KV head is shared by all query heads.</summary>
    private void Attention(float[] q, float[] k, float[] v, float[] attnOut, int seq, int headDim)
    {
        int qStride = _cfg.NumHeads * headDim;
        int kvStride = _cfg.NumKvHeads * headDim; // == headDim (1 KV head)
        float scale = _cfg.AttentionScale;

        // Parallelize over (head, query-position) pairs.
        int total = _cfg.NumHeads * seq;
        Parallel.For(0, total, idx =>
        {
            int head = idx / seq;
            int i = idx % seq;

            var scores = new float[i + 1];
            var qSpan = new ReadOnlySpan<float>(q, i * qStride + head * headDim, headDim);
            for (int j = 0; j <= i; j++)
            {
                var kSpan = new ReadOnlySpan<float>(k, j * kvStride, headDim);
                scores[j] = System.Numerics.Tensors.TensorPrimitives.Dot(qSpan, kSpan) * scale;
            }

            Ops.SoftmaxInPlace(scores, i + 1);

            int outBase = i * qStride + head * headDim;
            var outSpan = new Span<float>(attnOut, outBase, headDim);
            outSpan.Clear();
            for (int j = 0; j <= i; j++)
            {
                float w = scores[j];
                var vSpan = new ReadOnlySpan<float>(v, j * kvStride, headDim);
                for (int d = 0; d < headDim; d++)
                {
                    outSpan[d] += w * vSpan[d];
                }
            }
        });
    }

    private static (float[] cos, float[] sin) BuildRope(int seq, int headDim, float theta)
    {
        int half = headDim / 2;
        var cos = new float[seq * headDim];
        var sin = new float[seq * headDim];
        var invFreq = new float[half];
        for (int d = 0; d < half; d++)
        {
            invFreq[d] = 1f / MathF.Pow(theta, (2f * d) / headDim);
        }
        for (int p = 0; p < seq; p++)
        {
            int b = p * headDim;
            for (int d = 0; d < half; d++)
            {
                float angle = p * invFreq[d];
                float c = MathF.Cos(angle);
                float s = MathF.Sin(angle);
                cos[b + d] = c;
                cos[b + d + half] = c;
                sin[b + d] = s;
                sin[b + d + half] = s;
            }
        }
        return (cos, sin);
    }
}
