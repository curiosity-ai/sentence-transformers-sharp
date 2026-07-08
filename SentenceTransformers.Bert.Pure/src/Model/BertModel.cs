using SentenceTransformers.Bert.Pure.Numerics;

namespace SentenceTransformers.Bert.Pure.Model;

/// <summary>
/// Pure-managed implementation of a HuggingFace <c>BertModel</c> encoder — the same architecture behind
/// all-MiniLM-L6-v2 and snowflake-arctic-embed-xs. The forward pass is expressed entirely as
/// <see cref="Graph"/> ops so the exact gradients for injected LoRA parameters are obtained by replaying
/// the tape (<see cref="Graph.Backward"/>); the same code therefore serves both inference (build graph,
/// read the pooled vector) and training (build graph, seed the loss gradient, run backward).
///
/// <para>All base weights are frozen fp32 arrays loaded from the original safetensors checkpoint; only the
/// (tiny) <see cref="LoraRuntime"/> tensors are trainable.</para>
/// </summary>
internal sealed class BertModel
{
    private readonly BertConfig _cfg;

    // embeddings
    private readonly float[] _wordEmb;   // [vocab, hidden]
    private readonly float[] _posEmb;    // [maxPos, hidden]
    private readonly float[] _typeEmb;   // [2, hidden]
    private readonly float[] _embLnG, _embLnB;

    private readonly Layer[] _layers;

    public BertConfig Config => _cfg;

    private sealed class Layer
    {
        public float[] Wq, Bq, Wk, Bk, Wv, Bv;    // attention self
        public float[] Wo, Bo, AttnLnG, AttnLnB;  // attention output
        public float[] Wi, Bi;                    // intermediate
        public float[] Wout, Bout, OutLnG, OutLnB;// output
    }

    private BertModel(BertConfig cfg, float[] wordEmb, float[] posEmb, float[] typeEmb, float[] embLnG, float[] embLnB, Layer[] layers)
    {
        _cfg = cfg; _wordEmb = wordEmb; _posEmb = posEmb; _typeEmb = typeEmb; _embLnG = embLnG; _embLnB = embLnB; _layers = layers;
    }

    public static async Task<BertModel> LoadAsync(string safetensorsPath, BertConfig cfg, CancellationToken ct = default)
    {
        var st = await SafeTensors.LoadAsync(safetensorsPath, ct).ConfigureAwait(false);
        return Load(st, cfg);
    }

    public static BertModel Load(string safetensorsPath, BertConfig cfg) => Load(SafeTensors.Load(safetensorsPath), cfg);

    /// <summary>Loads the frozen weights straight from an embedded fp32 ONNX graph (no download, no ONNX Runtime).</summary>
    public static BertModel LoadFromOnnx(byte[] onnx, BertConfig cfg) => Load(new OnnxBertWeights(onnx, cfg), cfg);

    private static BertModel Load(IBertWeightProvider w, BertConfig cfg)
    {
        string prefix = w.Contains("embeddings.word_embeddings.weight") ? "" : "bert.";
        float[] R(string n) => w.Read(prefix + n);

        var layers = new Layer[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            string p = $"encoder.layer.{i}.";
            layers[i] = new Layer
            {
                Wq = R(p + "attention.self.query.weight"),  Bq = R(p + "attention.self.query.bias"),
                Wk = R(p + "attention.self.key.weight"),    Bk = R(p + "attention.self.key.bias"),
                Wv = R(p + "attention.self.value.weight"),  Bv = R(p + "attention.self.value.bias"),
                Wo = R(p + "attention.output.dense.weight"),Bo = R(p + "attention.output.dense.bias"),
                AttnLnG = R(p + "attention.output.LayerNorm.weight"), AttnLnB = R(p + "attention.output.LayerNorm.bias"),
                Wi = R(p + "intermediate.dense.weight"),    Bi = R(p + "intermediate.dense.bias"),
                Wout = R(p + "output.dense.weight"),        Bout = R(p + "output.dense.bias"),
                OutLnG = R(p + "output.LayerNorm.weight"),  OutLnB = R(p + "output.LayerNorm.bias"),
            };
        }

        return new BertModel(
            cfg,
            R("embeddings.word_embeddings.weight"),
            R("embeddings.position_embeddings.weight"),
            R("embeddings.token_type_embeddings.weight"),
            R("embeddings.LayerNorm.weight"),
            R("embeddings.LayerNorm.bias"),
            layers);
    }

    /// <summary>Builds a model with small random frozen weights — for gradient-checking the backward pass
    /// without downloading a real checkpoint.</summary>
    internal static BertModel CreateRandom(BertConfig cfg, int seed)
    {
        var rng = new Random(seed);
        float[] Rand(int n, float s) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1) * s; return a; }
        float[] Ones(int n, float jitter) { var a = new float[n]; for (int i = 0; i < n; i++) a[i] = 1f + (float)(rng.NextDouble() * 2 - 1) * jitter; return a; }

        int h = cfg.HiddenSize, inter = cfg.IntermediateSize;
        var layers = new Layer[cfg.NumLayers];
        for (int i = 0; i < cfg.NumLayers; i++)
        {
            layers[i] = new Layer
            {
                Wq = Rand(h * h, 0.1f), Bq = Rand(h, 0.05f), Wk = Rand(h * h, 0.1f), Bk = Rand(h, 0.05f), Wv = Rand(h * h, 0.1f), Bv = Rand(h, 0.05f),
                Wo = Rand(h * h, 0.1f), Bo = Rand(h, 0.05f), AttnLnG = Ones(h, 0.1f), AttnLnB = Rand(h, 0.05f),
                Wi = Rand(inter * h, 0.1f), Bi = Rand(inter, 0.05f),
                Wout = Rand(h * inter, 0.1f), Bout = Rand(h, 0.05f), OutLnG = Ones(h, 0.1f), OutLnB = Rand(h, 0.05f),
            };
        }
        return new BertModel(cfg, Rand(cfg.VocabSize * h, 0.1f), Rand(cfg.MaxPositionEmbeddings * h, 0.1f), Rand(2 * h, 0.1f), Ones(h, 0.1f), Rand(h, 0.05f), layers);
    }

    /// <summary>
    /// Runs the full forward pass for one tokenized sequence, returning the pooled embedding <b>before</b>
    /// L2 normalization (the caller normalizes — training normalizes in-graph, inference normalizes the
    /// data, optionally after whitening). <paramref name="rt"/> = null runs the frozen base encoder.
    /// </summary>
    public Tensor Forward(Graph g, int[] tokenIds, LoraRuntime rt)
    {
        int seq = tokenIds.Length;
        int h   = _cfg.HiddenSize;
        int heads = _cfg.NumHeads;
        int hd  = _cfg.HeadDim;
        float attnScale = 1f / MathF.Sqrt(hd);

        // --- embeddings: word + position + token_type(0), then LayerNorm ---
        var emb = new float[seq * h];
        for (int i = 0; i < seq; i++)
        {
            int tok = tokenIds[i];
            int b = i * h;
            var wsp = _wordEmb.AsSpan(tok * h, h);
            var psp = _posEmb.AsSpan(i * h, h);
            for (int j = 0; j < h; j++) emb[b + j] = wsp[j] + psp[j] + _typeEmb[j]; // type 0 row
        }
        var x = Graph.Leaf(emb, seq, h);
        x = g.LayerNorm(x, _embLnG, _embLnB, _cfg.LayerNormEps);

        for (int l = 0; l < _cfg.NumLayers; l++)
        {
            var L = _layers[l];

            // ---- self attention ----
            var q = Lin(g, x, L.Wq, L.Bq, h, h, rt, l, 0);
            var k = Lin(g, x, L.Wk, L.Bk, h, h, rt, l, 1);
            var v = Lin(g, x, L.Wv, L.Bv, h, h, rt, l, 2);

            var ctxHeads = new Tensor[heads];
            for (int head = 0; head < heads; head++)
            {
                var qh = g.SliceCols(q, head * hd, hd);
                var kh = g.SliceCols(k, head * hd, hd);
                var vh = g.SliceCols(v, head * hd, hd);
                var scores = g.Scale(g.MatMul(qh, kh, transposeB: true), attnScale); // [seq,seq]
                var probs  = g.Softmax(scores);
                ctxHeads[head] = g.MatMul(probs, vh, transposeB: false);             // [seq,hd]
            }
            var ctx = g.ConcatCols(ctxHeads);                                        // [seq,h]

            var attnDense = Lin(g, ctx, L.Wo, L.Bo, h, h, rt, l, 3);
            var attnRes   = g.Add(x, attnDense);
            var attnNorm  = g.LayerNorm(attnRes, L.AttnLnG, L.AttnLnB, _cfg.LayerNormEps);

            // ---- feed forward ----
            var inter    = Lin(g, attnNorm, L.Wi, L.Bi, _cfg.IntermediateSize, h, rt, l, 4);
            var interAct = g.Gelu(inter);
            var outDense = Lin(g, interAct, L.Wout, L.Bout, h, _cfg.IntermediateSize, rt, l, 5);
            var outRes   = g.Add(attnNorm, outDense);
            x = g.LayerNorm(outRes, L.OutLnG, L.OutLnB, _cfg.LayerNormEps);
        }

        // ---- pooling ----
        var pooled = _cfg.Pooling == PoolingMode.Cls ? g.Row(x, 0) : g.MeanRows(x);
        if (rt?.OutBias != null) pooled = g.AddRowVector(pooled, rt.OutBias);
        return pooled;
    }

    private static Tensor Lin(Graph g, Tensor x, float[] w, float[] bias, int outDim, int inDim, LoraRuntime rt, int layer, int t)
    {
        Tensor a = null, b = null;
        float scale = 1f;
        if (rt != null && (rt.Targets & LoraAdapter.Flag(t)) != 0)
        {
            a = rt.Aof(layer, t);
            b = rt.Bof(layer, t);
            scale = rt.Scale;
        }
        return g.LoraLinear(x, w, bias, outDim, inDim, a, b, scale);
    }
}
