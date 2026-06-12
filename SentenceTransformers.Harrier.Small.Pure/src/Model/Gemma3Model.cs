using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using SentenceTransformers;
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

    public static async Task<Gemma3Model> LoadAsync(string safetensorsPath, Gemma3Config cfg, Quantization quantization, ParallelOptions parallelOptions)
    {
        var st = await SafeTensors.LoadAsync(safetensorsPath, parallelOptions.CancellationToken).ConfigureAwait(false);

        // Some exports prefix every tensor with "model."; tolerate both.
        string prefix = st.Contains("embed_tokens.weight") ? "" : "model.";

        var embed = st.ReadRaw16(prefix + "embed_tokens.weight");

        // The attention/MLP projections dominate weight size and compute, so they carry the chosen
        // quantization (built on Parallel.ForAsync, never blocking); the tiny per-channel norm vectors
        // stay in float32.
        Task<IWeightMatrix> Proj(string p, string name, int outDim, int inDim) => IWeightMatrix.CreateAsync(st.ReadFloat(p + name), outDim, inDim, quantization, parallelOptions);

        var layers = new Layer[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"{prefix}layers.{i}.";
            var qProj    = await Proj(p, "self_attn.q_proj.weight", cfg.QProjOut,         cfg.HiddenSize).ConfigureAwait(false);
            var kProj    = await Proj(p, "self_attn.k_proj.weight", cfg.KvProjOut,        cfg.HiddenSize).ConfigureAwait(false);
            var vProj    = await Proj(p, "self_attn.v_proj.weight", cfg.KvProjOut,        cfg.HiddenSize).ConfigureAwait(false);
            var oProj    = await Proj(p, "self_attn.o_proj.weight", cfg.HiddenSize,       cfg.QProjOut).ConfigureAwait(false);
            var gateProj = await Proj(p, "mlp.gate_proj.weight",    cfg.IntermediateSize, cfg.HiddenSize).ConfigureAwait(false);
            var upProj   = await Proj(p, "mlp.up_proj.weight",      cfg.IntermediateSize, cfg.HiddenSize).ConfigureAwait(false);
            var downProj = await Proj(p, "mlp.down_proj.weight",    cfg.HiddenSize,       cfg.IntermediateSize).ConfigureAwait(false);

            layers[i] = new Layer
            {
                InputLayerNorm           = st.ReadFloat(p + "input_layernorm.weight"),
                PostAttentionLayerNorm   = st.ReadFloat(p + "post_attention_layernorm.weight"),
                PreFeedforwardLayerNorm  = st.ReadFloat(p + "pre_feedforward_layernorm.weight"),
                PostFeedforwardLayerNorm = st.ReadFloat(p + "post_feedforward_layernorm.weight"),
                QProj                    = qProj,
                KProj                    = kProj,
                VProj                    = vProj,
                OProj                    = oProj,
                QNorm                    = st.ReadFloat(p + "self_attn.q_norm.weight"),
                KNorm                    = st.ReadFloat(p + "self_attn.k_norm.weight"),
                GateProj                 = gateProj,
                UpProj                   = upProj,
                DownProj                 = downProj,
            };
        }

        var finalNorm = st.ReadFloat(prefix + "norm.weight");

        return new Gemma3Model(cfg, embed, layers, finalNorm);
    }

    internal static ArrayPool<float> _pooledArray = ArrayPool<float>.Create(512000, 24);

    [MethodImpl(MethodImplOptions.AggressiveOptimization)]
    /// <summary>Runs the full forward pass for one tokenized sequence and returns the (un-normalized)
    /// last-token hidden state - a <c>HiddenSize</c>-length embedding. The per-layer matmuls and
    /// attention parallelize via <see cref="ParallelExecution.ForAsync"/>, so the heavy CPU work is
    /// awaited rather than blocking the caller. <paramref name="ct"/> is observed at every token
    /// embedding step and at the start of each transformer layer (in addition to inside the awaited
    /// matmuls), so cancellation is honoured promptly even when running single-threaded.
    /// Takes <c>int[]</c> (not a span) because async methods cannot have ref-struct parameters.</summary>
    public async Task<float[]> ForwardAsync(int[] tokenIds, ParallelOptions parallelOptions)
    {
        int seq = tokenIds.Length;
        int h = _cfg.HiddenSize;
        int headDim = _cfg.HeadDim;

        // All per-forward scratch scales with the sequence length and is multi-megabyte at long
        // contexts (e.g. gate/up are seq*2048 floats), so it is rented from the shared array pool
        // rather than allocated (and LOH-collected) on every call. Pooled arrays may be larger than
        // requested, so every consumer is driven by explicit (seq * dim) lengths, never Array.Length.
        int hiddenLen = seq * h;
        var hidden    = _pooledArray.Rent(hiddenLen); // 156800
        var cos       = _pooledArray.Rent(seq * headDim);
        var sin       = _pooledArray.Rent(seq * headDim);
        var normed    = _pooledArray.Rent(hiddenLen);
        var q         = _pooledArray.Rent(seq * _cfg.QProjOut); //250880
        var k         = _pooledArray.Rent(seq * _cfg.KvProjOut);
        var v         = _pooledArray.Rent(seq * _cfg.KvProjOut);
        var attnOut   = _pooledArray.Rent(seq * _cfg.QProjOut);
        var attnProj  = _pooledArray.Rent(hiddenLen);
        var gate      = _pooledArray.Rent(seq * _cfg.IntermediateSize); // 501760
        var up        = _pooledArray.Rent(seq * _cfg.IntermediateSize);
        var down      = _pooledArray.Rent(hiddenLen);
        try
        {
            // --- token embedding (+ sqrt(hidden) scale) ---
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
            BuildRope(seq, headDim, _cfg.RopeTheta, cos, sin);

            foreach (var layer in _layers)
            {
                parallelOptions.CancellationToken.ThrowIfCancellationRequested();

                // ---- self attention ----
                Ops.RmsNorm(hidden, layer.InputLayerNorm, normed, seq, h, _cfg.RmsNormEps);
                await layer.QProj.MultiplyAsync(normed, q, seq, parallelOptions).ConfigureAwait(false);
                await layer.KProj.MultiplyAsync(normed, k, seq, parallelOptions).ConfigureAwait(false);
                await layer.VProj.MultiplyAsync(normed, v, seq, parallelOptions).ConfigureAwait(false);

                ApplyHeadNorm(q, layer.QNorm, seq, _cfg.NumHeads, headDim);
                ApplyHeadNorm(k, layer.KNorm, seq, _cfg.NumKvHeads, headDim);

                ApplyRope(q, cos, sin, seq, _cfg.NumHeads, headDim);
                ApplyRope(k, cos, sin, seq, _cfg.NumKvHeads, headDim);

                await AttentionAsync(q, k, v, attnOut, seq, headDim, parallelOptions).ConfigureAwait(false);

                await layer.OProj.MultiplyAsync(attnOut, attnProj, seq, parallelOptions).ConfigureAwait(false);
                // post-attention norm then residual add
                Ops.RmsNorm(attnProj, layer.PostAttentionLayerNorm, attnProj, seq, h, _cfg.RmsNormEps);
                TensorPrimitives.Add(hidden.AsSpan(0, hiddenLen), attnProj.AsSpan(0, hiddenLen), hidden.AsSpan(0, hiddenLen));

                // ---- feed forward ----
                Ops.RmsNorm(hidden, layer.PreFeedforwardLayerNorm, normed, seq, h, _cfg.RmsNormEps);
                await layer.GateProj.MultiplyAsync(normed, gate, seq, parallelOptions).ConfigureAwait(false);
                await layer.UpProj.MultiplyAsync(normed, up, seq, parallelOptions).ConfigureAwait(false);
                int ffLen = seq * _cfg.IntermediateSize;
                Ops.GeGluInPlace(gate.AsSpan(0, ffLen), up.AsSpan(0, ffLen));
                await layer.DownProj.MultiplyAsync(gate, down, seq, parallelOptions).ConfigureAwait(false);
                Ops.RmsNorm(down, layer.PostFeedforwardLayerNorm, down, seq, h, _cfg.RmsNormEps);
                TensorPrimitives.Add(hidden.AsSpan(0, hiddenLen), down.AsSpan(0, hiddenLen), hidden.AsSpan(0, hiddenLen));
            }

            // ---- final norm + last-token pooling ----
            // The result is the caller's embedding, so it is a real (small, 640-float) array, not pooled.
            var lastNormed = new float[h];
            Ops.RmsNorm(hidden.AsSpan((seq - 1) * h, h), _finalNorm, lastNormed, 1, h, _cfg.RmsNormEps);
            return lastNormed;
        }
        finally
        {
            _pooledArray.Return(hidden);
            _pooledArray.Return(cos);
            _pooledArray.Return(sin);
            _pooledArray.Return(normed);
            _pooledArray.Return(q);
            _pooledArray.Return(k);
            _pooledArray.Return(v);
            _pooledArray.Return(attnOut);
            _pooledArray.Return(attnProj);
            _pooledArray.Return(gate);
            _pooledArray.Return(up);
            _pooledArray.Return(down);
        }
    }

    
    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
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

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
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

    /// <summary>Causal grouped-query attention. The single KV head is shared by all query heads.
    /// Parallelizes over query positions via <see cref="ParallelExecution.ForAsync"/>: per-index
    /// <c>Parallel.ForAsync</c> dispatch gives dynamic load balancing for the triangular causal work
    /// when <see cref="ParallelExecution.Enabled"/>, and runs single-threaded otherwise.</summary>
    private Task AttentionAsync(float[] q, float[] k, float[] v, float[] attnOut, int seq, int headDim, ParallelOptions parallelOptions)
    {
        int qStride  = _cfg.NumHeads * headDim;
        int kvStride = _cfg.NumKvHeads * headDim; // == headDim (1 KV head)
        int heads    = _cfg.NumHeads;
        float scale  = _cfg.AttentionScale;

        return ParallelExecution.ForAsync(0, seq, parallelOptions, (i, _) =>
        {
            AttentionRow(q, k, v, attnOut, i, headDim, qStride, kvStride, heads, scale);
            return ValueTask.CompletedTask;
        });
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
    /// <summary>Computes attention output for one query position across all heads. Synchronous so the
    /// pooled score buffer and spans stay out of an async body. Score buffers come from a pool (no
    /// per-call allocations); the value accumulation is a vectorized scalar*vector add.</summary>
    private static void AttentionRow(float[] q, float[] k, float[] v, float[] attnOut, int i, int headDim, int qStride, int kvStride, int heads, float scale)
    {
        float[] scores = ArrayPool<float>.Shared.Rent(i + 1);

        for (int head = 0; head < heads; head++)
        {
            var qSpan = new ReadOnlySpan<float>(q, i * qStride + head * headDim, headDim);

            for (int j = 0; j <= i; j++)
            {
                scores[j] = TensorPrimitives.Dot(qSpan, new ReadOnlySpan<float>(k, j * kvStride, headDim)) * scale;
            }

            Ops.SoftmaxInPlace(scores, i + 1);

            var outSpan = new Span<float>(attnOut, i * qStride + head * headDim, headDim);
            outSpan.Clear();
            for (int j = 0; j <= i; j++)
            {
                // outSpan += scores[j] * v[j]
                TensorPrimitives.MultiplyAdd(new ReadOnlySpan<float>(v, j * kvStride, headDim), scores[j], outSpan, outSpan);
            }
        }

        ArrayPool<float>.Shared.Return(scores);
    }

    [MethodImpl(MethodImplOptions.AggressiveOptimization | MethodImplOptions.AggressiveInlining)]
    /// <summary>Fills the caller-provided <paramref name="cos"/>/<paramref name="sin"/> rotary tables
    /// (each at least <c>seq * headDim</c> long). <paramref name="invFreq"/> is tiny (headDim/2 = 128
    /// floats) so it is stack-allocated.</summary>
    private static void BuildRope(int seq, int headDim, float theta, float[] cos, float[] sin)
    {
        int half = headDim / 2;
        Span<float> invFreq = stackalloc float[half];
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
    }
}
