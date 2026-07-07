namespace SentenceTransformers.Training;

/// <summary>
/// Trains a <see cref="LoraAdapter"/> on top of a frozen <see cref="ISentenceEncoder"/> from a set of
/// related <see cref="SentencePair"/>s, using a contrastive InfoNCE objective with in-batch negatives.
///
/// <para><b>How it works.</b> The base encoder is treated as a frozen black box: every unique sentence
/// is embedded exactly once up front and cached, so the training loop never touches the (expensive)
/// model again — it only does cheap <c>O(rank · dim)</c> math on the cached vectors. Because the
/// adapter is a small analytic function of the frozen embedding, its gradients are computed exactly
/// (no autodiff framework needed) and optimized with AdamW. This makes the exact same trainer work for
/// every model in the library — MiniLM, Arctic XS, Qwen3, Harrier — since they all expose the same
/// <see cref="ISentenceEncoder.EncodeAsync"/> contract.</para>
///
/// <para><b>Objective.</b> For a batch of <c>N</c> positive pairs the adapted, normalized anchor and
/// positive vectors form a similarity matrix; the loss is the symmetric cross-entropy that pushes each
/// anchor towards its own positive and away from the other <c>N-1</c> positives in the batch
/// (the negatives). The data is split into training and validation sets; after each epoch the trainer
/// reports validation loss, top-1 retrieval accuracy and — when the pairs carry similarity scores —
/// the STS Spearman correlation, and keeps the best adapter seen.</para>
/// </summary>
public static class LoraTrainer
{
    /// <summary>
    /// Fine-tunes a LoRA adapter for <paramref name="baseEncoder"/> on <paramref name="dataset"/>.
    /// </summary>
    /// <param name="baseEncoder">The frozen encoder to adapt. It is only used to produce embeddings.</param>
    /// <param name="dataset">Related pairs to learn from.</param>
    /// <param name="options">Training hyper-parameters (see <see cref="LoraTrainingOptions"/>).</param>
    /// <param name="cancellationToken">Cancels embedding and training.</param>
    /// <returns>A report holding the best adapter, per-epoch metrics and baseline-vs-tuned numbers.</returns>
    public static async Task<LoraTrainingReport> TrainAsync(
        ISentenceEncoder    baseEncoder,
        SentencePairDataset dataset,
        LoraTrainingOptions options           = null,
        CancellationToken   cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(baseEncoder);
        ArgumentNullException.ThrowIfNull(dataset);
        options ??= new LoraTrainingOptions();

        if (dataset.Count < 2)
        {
            throw new ArgumentException("Need at least two pairs to train.", nameof(dataset));
        }

        var (train, validation) = dataset.Split(options.ValidationFraction, options.Seed);

        // 1. Embed every unique sentence once with the frozen encoder, then work purely on the cache.
        var embeddings = await EmbedUniqueAsync(baseEncoder, train, validation, cancellationToken);
        int dim        = embeddings.Dimension;

        // 2. Materialize positive-pair index lists (filtered by score threshold when scores exist).
        var trainPos = SelectPositivePairs(train,      embeddings, options.PositiveScoreThreshold);
        var valPos   = SelectPositivePairs(validation, embeddings, options.PositiveScoreThreshold);

        if (trainPos.Count == 0)
        {
            throw new InvalidOperationException(
                "No positive training pairs remained after applying the score threshold. " +
                "Lower LoraTrainingOptions.PositiveScoreThreshold or provide pairs without scores.");
        }

        // 3. Init adapter + AdamW optimizer state.
        var adapter = LoraAdapter.CreateInitialized(dim, options.Rank, options.Alpha, options.Seed);
        var optA    = new Adam(adapter.A.Length, options);
        var optB    = new Adam(adapter.B.Length, options);

        var gradA = new float[adapter.A.Length];
        var gradB = new float[adapter.B.Length];

        // 4. Baseline metrics for the un-adapted encoder (identity adapter).
        var (baselineAcc, _)      = Evaluate(null, embeddings, valPos, options.Temperature, dim, options.Rank);
        float baselineSpearman    = EvaluateSpearman(null, embeddings, validation, dim, options.Rank);

        var epochMetrics = new List<EpochMetrics>(options.Epochs);
        var rng          = new Random(options.Seed);
        var order        = Enumerable.Range(0, trainPos.Count).ToArray();

        LoraAdapter best         = CloneAdapter(adapter);
        float       bestPrimary  = float.NegativeInfinity;
        bool        haveScores   = validation.Count > 0 && validation.Pairs.All(p => p.Score.HasValue);

        for (int epoch = 1; epoch <= options.Epochs; epoch++)
        {
            cancellationToken.ThrowIfCancellationRequested();

            Shuffle(order, rng);

            double epochLoss   = 0;
            int    batchCount  = 0;

            for (int start = 0; start < order.Length; start += options.BatchSize)
            {
                int end = Math.Min(start + options.BatchSize, order.Length);
                int n   = end - start;
                if (n < 2) continue; // a batch of 1 has no negatives; skip the trailing singleton

                Array.Clear(gradA);
                Array.Clear(gradB);

                float loss = ContrastiveBatch(adapter, embeddings, trainPos, order, start, n, options.Temperature, gradA, gradB);
                epochLoss += loss;
                batchCount++;

                optA.Step(adapter.A, gradA);
                optB.Step(adapter.B, gradB);
            }

            float trainLoss = batchCount > 0 ? (float)(epochLoss / batchCount) : float.NaN;

            var (valAcc, valLoss) = Evaluate(adapter, embeddings, valPos, options.Temperature, dim, options.Rank);
            float valSpearman     = haveScores ? EvaluateSpearman(adapter, embeddings, validation, dim, options.Rank) : float.NaN;

            // Primary metric: Spearman when scores exist, otherwise retrieval accuracy.
            float primary = haveScores ? valSpearman : valAcc;
            bool  isBest  = primary > bestPrimary;
            if (isBest)
            {
                bestPrimary = primary;
                best        = CloneAdapter(adapter);
            }

            var m = new EpochMetrics(epoch, trainLoss, valLoss, valAcc, valSpearman, isBest);
            epochMetrics.Add(m);
            options.OnEpoch?.Invoke(m);
        }

        var (bestAcc, _)   = Evaluate(best, embeddings, valPos, options.Temperature, dim, options.Rank);
        float bestSpearman = haveScores ? EvaluateSpearman(best, embeddings, validation, dim, options.Rank) : float.NaN;

        return new LoraTrainingReport(best, epochMetrics, baselineAcc, baselineSpearman, bestAcc, bestSpearman);
    }

    // ----- embedding cache ------------------------------------------------------------------------

    private sealed class EmbeddingCache
    {
        public required Dictionary<string, int> Index;
        public required float[][]                Vectors;
        public required int                      Dimension;

        public float[] Get(string text) => Vectors[Index[text ?? string.Empty]];
    }

    private static async Task<EmbeddingCache> EmbedUniqueAsync(
        ISentenceEncoder    encoder,
        SentencePairDataset train,
        SentencePairDataset validation,
        CancellationToken   cancellationToken)
    {
        var index  = new Dictionary<string, int>(StringComparer.Ordinal);
        var unique = new List<string>();

        void Register(string s)
        {
            s ??= string.Empty;
            if (!index.ContainsKey(s))
            {
                index[s] = unique.Count;
                unique.Add(s);
            }
        }

        foreach (var p in train.Pairs)      { Register(p.Anchor); Register(p.Positive); }
        foreach (var p in validation.Pairs) { Register(p.Anchor); Register(p.Positive); }

        var vectors = new float[unique.Count][];

        const int batch = 64;
        for (int start = 0; start < unique.Count; start += batch)
        {
            cancellationToken.ThrowIfCancellationRequested();
            int end   = Math.Min(start + batch, unique.Count);
            var slice = unique.GetRange(start, end - start).ToArray();
            var vecs  = await encoder.EncodeAsync(slice, cancellationToken);
            for (int i = 0; i < vecs.Length; i++)
            {
                vectors[start + i] = vecs[i];
            }
        }

        int dim = vectors.Length > 0 && vectors[0] is { Length: > 0 } ? vectors[0].Length : 0;
        if (dim == 0)
        {
            throw new InvalidOperationException("Base encoder produced empty embeddings.");
        }

        return new EmbeddingCache { Index = index, Vectors = vectors, Dimension = dim };
    }

    // ----- positive-pair selection ----------------------------------------------------------------

    private readonly record struct PosPair(float[] Anchor, float[] Positive);

    private static List<PosPair> SelectPositivePairs(SentencePairDataset data, EmbeddingCache cache, float threshold)
    {
        var result = new List<PosPair>(data.Count);
        foreach (var p in data.Pairs)
        {
            if (p.Score.HasValue && p.Score.Value < threshold)
            {
                continue; // low-similarity pairs are not treated as positives for contrastive training
            }
            result.Add(new PosPair(cache.Get(p.Anchor), cache.Get(p.Positive)));
        }
        return result;
    }

    // ----- contrastive forward + backward ---------------------------------------------------------

    // Test-only entry point: run one contrastive batch over explicit anchor/positive vectors and
    // return the loss while accumulating exact gradients. Used by the numerical gradient check.
    internal static float DebugContrastiveBatch(LoraAdapter adapter, float[][] anchors, float[][] positives, float temperature, float[] gradA, float[] gradB)
    {
        int n     = anchors.Length;
        var pairs = new List<PosPair>(n);
        for (int i = 0; i < n; i++) pairs.Add(new PosPair(anchors[i], positives[i]));
        var order = Enumerable.Range(0, n).ToArray();
        return ContrastiveBatch(adapter, null, pairs, order, 0, n, temperature, gradA, gradB);
    }

    // Runs one batch of the symmetric InfoNCE loss and accumulates exact gradients into gradA/gradB.
    private static float ContrastiveBatch(
        LoraAdapter    adapter,
        EmbeddingCache embeddings,
        List<PosPair>  pairs,
        int[]          order,
        int            start,
        int            n,
        float          temperature,
        float[]        gradA,
        float[]        gradB)
    {
        int dim  = adapter.Dimension;
        int rank = adapter.Rank;

        // Forward: adapt every anchor and positive, caching intermediates for backprop.
        var qE = new float[n][]; var qU = new float[n][]; var qH = new float[n][]; var qN = new float[n];
        var pE = new float[n][]; var pU = new float[n][]; var pH = new float[n][]; var pN = new float[n];

        for (int b = 0; b < n; b++)
        {
            var pair = pairs[order[start + b]];
            qE[b] = pair.Anchor;   (qU[b], qH[b], qN[b]) = Forward(adapter, pair.Anchor,   dim, rank);
            pE[b] = pair.Positive; (pU[b], pH[b], pN[b]) = Forward(adapter, pair.Positive, dim, rank);
        }

        // Similarity matrix S[i,j] = <qU_i, pU_j> / temperature.
        float invTemp = 1f / temperature;
        var   s       = new float[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                s[i * n + j] = EmbeddingMath.Dot(qU[i], pU[j]) * invTemp;
            }
        }

        // Row softmax (anchor -> positive) and column softmax (positive -> anchor).
        var rowP = Softmax2D(s, n, byRow: true);
        var colP = Softmax2D(s, n, byRow: false);

        // Loss = 0.5 * (mean_i -log rowP[i,i] + mean_j -log colP[j,j]).
        double loss = 0;
        for (int i = 0; i < n; i++)
        {
            loss += -Math.Log(Math.Max(rowP[i * n + i], 1e-20f));
            loss += -Math.Log(Math.Max(colP[i * n + i], 1e-20f));
        }
        loss = 0.5 * loss / n;

        // dLoss/dS. Both softmax directions contribute (p - onehot), scaled by 0.5/n.
        float scale = 0.5f / n;
        var   dS    = new float[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float rowTerm = rowP[i * n + j] - (i == j ? 1f : 0f);
                float colTerm = colP[i * n + j] - (i == j ? 1f : 0f);
                dS[i * n + j] = scale * (rowTerm + colTerm);
            }
        }

        // dLoss/dqU_i = sum_j dS[i,j] * pU_j / temp ; dLoss/dpU_j = sum_i dS[i,j] * qU_i / temp.
        var gQ = new float[n][];
        var gP = new float[n][];
        for (int i = 0; i < n; i++) { gQ[i] = new float[dim]; gP[i] = new float[dim]; }

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float c = dS[i * n + j] * invTemp;
                if (c == 0f) continue;
                var pu = pU[j]; var gq = gQ[i];
                var qu = qU[i]; var gp = gP[j];
                for (int k = 0; k < dim; k++)
                {
                    gq[k] += c * pu[k];
                    gp[k] += c * qu[k];
                }
            }
        }

        // Backprop each adapted vector through normalization + the low-rank residual.
        for (int b = 0; b < n; b++)
        {
            Backward(adapter, qE[b], qU[b], qH[b], qN[b], gQ[b], gradA, gradB);
            Backward(adapter, pE[b], pU[b], pH[b], pN[b], gP[b], gradA, gradB);
        }

        return (float)loss;
    }

    private static (float[] u, float[] h, float norm) Forward(LoraAdapter adapter, float[] e, int dim, int rank)
    {
        var u = new float[dim];
        var h = new float[rank];
        adapter.Forward(e, h, u, out float norm);
        return (u, h, norm);
    }

    // Backward through u = z/||z|| with z = e + scaling * B h, accumulating grads for A and B.
    private static void Backward(LoraAdapter adapter, float[] e, float[] u, float[] h, float norm, float[] gU, float[] gradA, float[] gradB)
    {
        int   dim     = adapter.Dimension;
        int   rank    = adapter.Rank;
        float scaling = adapter.Scaling;
        var   A       = adapter.A;
        var   B       = adapter.B;

        // Normalization backward: dL/dz = (gU - (gU·u) u) / norm.
        float dot = 0f;
        for (int k = 0; k < dim; k++) dot += gU[k] * u[k];

        var gz = new float[dim];
        float invNorm = 1f / norm;
        for (int k = 0; k < dim; k++)
        {
            gz[k] = (gU[k] - dot * u[k]) * invNorm;
        }

        // gradB[k,j] += scaling * gz[k] * h[j] ; and gh[j] = scaling * sum_k B[k,j] * gz[k].
        var gh = new float[rank];
        for (int k = 0; k < dim; k++)
        {
            float sgz     = scaling * gz[k];
            int   rowBase = k * rank;
            for (int j = 0; j < rank; j++)
            {
                gradB[rowBase + j] += sgz * h[j];
                gh[j]              += scaling * B[rowBase + j] * gz[k];
            }
        }

        // gradA[j,i] += gh[j] * e[i]  (from h = A e).
        for (int j = 0; j < rank; j++)
        {
            float ghj     = gh[j];
            if (ghj == 0f) continue;
            int   rowBase = j * dim;
            for (int i = 0; i < dim; i++)
            {
                gradA[rowBase + i] += ghj * e[i];
            }
        }
    }

    // ----- evaluation ------------------------------------------------------------------------------

    // Returns (top-1 in-batch retrieval accuracy, InfoNCE loss) over the positive validation pairs.
    // A null adapter evaluates the un-adapted (identity) base encoder.
    private static (float accuracy, float loss) Evaluate(
        LoraAdapter    adapter,
        EmbeddingCache embeddings,
        List<PosPair>  valPos,
        float          temperature,
        int            dim,
        int            rank)
    {
        int n = valPos.Count;
        if (n < 2) return (float.NaN, float.NaN);

        var qU = new float[n][];
        var pU = new float[n][];
        for (int i = 0; i < n; i++)
        {
            qU[i] = Adapt(adapter, valPos[i].Anchor,   dim, rank);
            pU[i] = Adapt(adapter, valPos[i].Positive, dim, rank);
        }

        float invTemp = 1f / temperature;
        var   s       = new float[n * n];
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                s[i * n + j] = EmbeddingMath.Dot(qU[i], pU[j]) * invTemp;
            }
        }

        int correct = 0;
        for (int i = 0; i < n; i++)
        {
            int   argmax = 0;
            float bestS  = float.NegativeInfinity;
            for (int j = 0; j < n; j++)
            {
                float v = s[i * n + j];
                if (v > bestS) { bestS = v; argmax = j; }
            }
            if (argmax == i) correct++;
        }

        var   rowP = Softmax2D(s, n, byRow: true);
        double loss = 0;
        for (int i = 0; i < n; i++) loss += -Math.Log(Math.Max(rowP[i * n + i], 1e-20f));
        loss /= n;

        return ((float)correct / n, (float)loss);
    }

    // STS-style Spearman correlation between adapted cosine similarity and gold score over all pairs.
    private static float EvaluateSpearman(LoraAdapter adapter, EmbeddingCache embeddings, SentencePairDataset data, int dim, int rank)
    {
        int n = data.Count;
        if (n < 2) return float.NaN;

        var sims   = new float[n];
        var scores = new float[n];
        for (int i = 0; i < n; i++)
        {
            var p = data.Pairs[i];
            var a = Adapt(adapter, embeddings.Get(p.Anchor),   dim, rank);
            var b = Adapt(adapter, embeddings.Get(p.Positive), dim, rank);
            sims[i]   = EmbeddingMath.Dot(a, b); // cosine, since both are L2-normalized
            scores[i] = p.Score ?? 0f;
        }

        return EmbeddingMath.Spearman(sims, scores);
    }

    // Applies the adapter (or just normalizes, for the identity/base case).
    private static float[] Adapt(LoraAdapter adapter, float[] e, int dim, int rank)
    {
        if (adapter is not null)
        {
            return adapter.Apply(e);
        }

        // Identity: normalize a copy so baseline cosine is well-defined even if the base wasn't unit-length.
        var u    = (float[])e.Clone();
        float ss = 0f;
        for (int k = 0; k < u.Length; k++) ss += u[k] * u[k];
        float inv = 1f / MathF.Max(MathF.Sqrt(ss), 1e-12f);
        for (int k = 0; k < u.Length; k++) u[k] *= inv;
        return u;
    }

    // ----- small numeric helpers ------------------------------------------------------------------

    private static float[] Softmax2D(float[] s, int n, bool byRow)
    {
        var result = new float[n * n];
        for (int a = 0; a < n; a++)
        {
            // For byRow, 'a' is the row index; for byColumn, 'a' is the column index.
            float max = float.NegativeInfinity;
            for (int b = 0; b < n; b++)
            {
                float v = byRow ? s[a * n + b] : s[b * n + a];
                if (v > max) max = v;
            }

            float sum = 0f;
            for (int b = 0; b < n; b++)
            {
                float v   = byRow ? s[a * n + b] : s[b * n + a];
                float e   = MathF.Exp(v - max);
                sum      += e;
                if (byRow) result[a * n + b] = e; else result[b * n + a] = e;
            }

            float inv = 1f / sum;
            for (int b = 0; b < n; b++)
            {
                if (byRow) result[a * n + b] *= inv; else result[b * n + a] *= inv;
            }
        }
        return result;
    }

    private static void Shuffle(int[] a, Random rng)
    {
        for (int i = a.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (a[i], a[j]) = (a[j], a[i]);
        }
    }

    private static LoraAdapter CloneAdapter(LoraAdapter src)
    {
        using var ms = new MemoryStream();
        src.Save(ms);
        ms.Position = 0;
        return LoraAdapter.Load(ms);
    }

    // ----- AdamW optimizer -------------------------------------------------------------------------

    private sealed class Adam
    {
        private readonly float[] _m;
        private readonly float[] _v;
        private readonly float   _lr, _beta1, _beta2, _eps, _weightDecay;
        private int              _t;

        public Adam(int size, LoraTrainingOptions o)
        {
            _m           = new float[size];
            _v           = new float[size];
            _lr          = o.LearningRate;
            _beta1       = o.Beta1;
            _beta2       = o.Beta2;
            _eps         = o.Epsilon;
            _weightDecay = o.WeightDecay;
        }

        public void Step(float[] param, float[] grad)
        {
            _t++;
            float bc1 = 1f - MathF.Pow(_beta1, _t);
            float bc2 = 1f - MathF.Pow(_beta2, _t);

            for (int i = 0; i < param.Length; i++)
            {
                float g = grad[i];
                _m[i]   = _beta1 * _m[i] + (1f - _beta1) * g;
                _v[i]   = _beta2 * _v[i] + (1f - _beta2) * g * g;

                float mHat = _m[i] / bc1;
                float vHat = _v[i] / bc2;

                // Decoupled weight decay (AdamW).
                if (_weightDecay != 0f)
                {
                    param[i] -= _lr * _weightDecay * param[i];
                }
                param[i] -= _lr * mHat / (MathF.Sqrt(vHat) + _eps);
            }
        }
    }
}
