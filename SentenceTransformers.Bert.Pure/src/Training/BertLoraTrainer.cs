using System.Numerics.Tensors;
using Microsoft.Extensions.Logging;
using SentenceTransformers.Bert.Pure.Model;
using SentenceTransformers.Training;
using Tensor = SentenceTransformers.Training.Autograd.Tensor;
using Graph  = SentenceTransformers.Training.Autograd.Graph;

namespace SentenceTransformers.Bert.Pure.Training;

/// <summary>
/// Trains real weight-space LoRA adapters for a pure-C# BERT encoder. Unlike an output-space adapter, the
/// low-rank factors are injected <b>inside</b> the transformer's linear projections and optimized by
/// backpropagating the contrastive / CoSENT / regression loss through the whole (frozen) network via the
/// autograd graph. Supports in-batch + hard (explicit and mined) negatives with false-negative masking,
/// a learned temperature, warmup + cosine LR, multi-seed model selection, Matryoshka sub-dimension losses,
/// asymmetric query/document prefixes, a learned output-centering bias and optional post-hoc whitening.
/// </summary>
public static class BertLoraTrainer
{
    public static async Task<BertLoraTrainingReport> TrainAsync(
        SentenceEncoder baseEncoder, SentencePairDataset dataset, BertLoraTrainingOptions options = null, CancellationToken ct = default)
    {
        ArgumentNullException.ThrowIfNull(baseEncoder);
        ArgumentNullException.ThrowIfNull(dataset);
        options ??= new BertLoraTrainingOptions();
        if (dataset.Count < 2) throw new ArgumentException("Need at least two pairs to train.", nameof(dataset));

        var model = baseEncoder.ModelInternal;
        var cfg   = model.Config;
        baseEncoder.Tokenizer.SetMaxTokens(Math.Min(options.MaxTokens, baseEncoder.MaxChunkLength));

        var (train, validation) = dataset.Split(options.ValidationFraction, options.Seed);
        bool haveScores = validation.Count > 0 && validation.Pairs.All(p => p.Score.HasValue);

        // Baseline (frozen encoder) validation metrics.
        var baseRetr = await EmbeddingEvaluation.RetrievalAsync(baseEncoder, validation, minScore: 0.6f, ct);
        float baselineSpearman = haveScores ? await EmbeddingEvaluation.SpearmanAsync(baseEncoder, validation, ct) : float.NaN;

        options.Logger?.LogInformation(
            "BERT LoRA training: {Pairs} train pairs, objective {Objective}, rank {Rank}, targets {Targets}, {Seeds} seed(s), up to {Epochs} epochs. Baseline retrieval acc {Acc:0.000}, spearman {Spear:0.000}.",
            train.Count, options.Objective, options.Rank, options.Targets, Math.Max(1, options.NumSeeds), options.Epochs, baseRetr.Accuracy, baselineSpearman);

        LoraAdapter best = null;
        float bestPrimary = float.NegativeInfinity;
        float bestAcc = float.NaN, bestSpear = float.NaN;
        var allMetrics = new List<BertEpochMetrics>();

        for (int s = 0; s < Math.Max(1, options.NumSeeds); s++)
        {
            int seed = options.Seed + s;
            var (adapter, metrics, acc, spear) = await TrainOneSeedAsync(baseEncoder, model, cfg, train, validation, haveScores, options, seed, ct);
            allMetrics.AddRange(metrics);
            float primary = haveScores ? spear : acc;
            if (primary > bestPrimary) { bestPrimary = primary; best = adapter; bestAcc = acc; bestSpear = spear; }
        }

        // Optional post-hoc whitening fitted from the tuned training embeddings.
        if (options.ApplyWhitening && best != null)
        {
            options.Logger?.LogInformation("Fitting post-hoc ZCA whitening transform ...");
            using var tuned = baseEncoder.WithAdapter(best);
            WhiteningFitter.Fit(tuned, train, best, ct);
            // Re-evaluate with whitening applied.
            using var tunedW = baseEncoder.WithAdapter(best);
            bestAcc   = (await EmbeddingEvaluation.RetrievalAsync(tunedW, validation, minScore: 0.6f, ct)).Accuracy;
            bestSpear = haveScores ? await EmbeddingEvaluation.SpearmanAsync(tunedW, validation, ct) : float.NaN;
        }

        options.Logger?.LogInformation(
            "BERT LoRA training done. Tuned retrieval acc {BaseAcc:0.000} -> {Acc:0.000}, spearman {BaseSpear:0.000} -> {Spear:0.000}.",
            baseRetr.Accuracy, bestAcc, baselineSpearman, bestSpear);

        return new BertLoraTrainingReport(best, allMetrics, baseRetr.Accuracy, baselineSpearman, bestAcc, bestSpear);
    }

    private static async Task<(LoraAdapter adapter, List<BertEpochMetrics> metrics, float acc, float spear)> TrainOneSeedAsync(
        SentenceEncoder baseEncoder, BertModel model, BertConfig cfg,
        SentencePairDataset train, SentencePairDataset validation, bool haveScores,
        BertLoraTrainingOptions options, int seed, CancellationToken ct)
    {
        int dim = cfg.HiddenSize;
        bool contrastive = options.Objective == BertTrainingObjective.Contrastive;

        // Build the training examples (anchor/positive [+ explicit negative], score).
        var examples = BuildExamples(train, options, contrastive);
        if (examples.Count == 0)
            throw new InvalidOperationException("No usable training examples after filtering (check PositiveScoreThreshold / scores).");

        var adapter = LoraAdapter.CreateInitialized(cfg, options.Rank, options.Alpha, options.Targets, seed);
        if (options.UseOutputBias) adapter.OutputBias = new float[dim];
        var rt = adapter.ToRuntime();
        var paramTensors = rt.Parameters().ToList();
        var adam = new AdamW(paramTensors, options);

        float temp = options.Temperature;
        float tempM = 0, tempV = 0; // Adam state for the learned temperature

        int[] dims = (options.MatryoshkaDims != null && options.MatryoshkaDims.Length > 0)
            ? options.MatryoshkaDims.Where(d => d > 0 && d <= dim).Append(dim).Distinct().OrderBy(d => d).ToArray()
            : new[] { dim };

        var tokenCache = new Dictionary<string, int[]>(StringComparer.Ordinal);
        int[] Ids(string prefixed) => tokenCache.TryGetValue(prefixed, out var v) ? v : (tokenCache[prefixed] = baseEncoder.TokenIds(prefixed));

        var rng = new Random(seed);
        int totalSteps = Math.Max(1, options.Epochs * ((examples.Count + options.BatchSize - 1) / options.BatchSize));
        int step = 0;

        var metrics = new List<BertEpochMetrics>();
        LoraAdapter bestSeed = null;
        float bestPrimary = float.NegativeInfinity, bestAcc = float.NaN, bestSpear = float.NaN;
        int sinceImprovement = 0;

        List<string>[] minedNeg = null;

        for (int epoch = 1; epoch <= options.Epochs; epoch++)
        {
            ct.ThrowIfCancellationRequested();

            if (contrastive && options.MinedNegativesPerAnchor > 0 && (epoch - 1) % Math.Max(1, options.MineEveryEpochs) == 0)
            {
                options.Logger?.LogDebug("seed {Seed} epoch {Epoch}: mining {K} hard negative(s) per anchor", seed, epoch, options.MinedNegativesPerAnchor);
                minedNeg = MineNegatives(baseEncoder, adapter, rt, examples, options, Ids, ct);
            }

            var order = Enumerable.Range(0, examples.Count).ToArray();
            Shuffle(order, rng);

            double epochLoss = 0; int batches = 0;
            for (int start = 0; start < order.Length; start += options.BatchSize)
            {
                ct.ThrowIfCancellationRequested();
                int end = Math.Min(start + options.BatchSize, order.Length);
                var batch = new ArraySegment<int>(order, start, end - start);

                float lr = ScheduleLr(options, step, totalSteps);
                double loss = TrainBatch(model, rt, examples, batch, minedNeg, dims, dim, options, ref temp, Ids, adam, lr, ref tempM, ref tempV, contrastive);
                epochLoss += loss; batches++;
                step++;
            }

            adapter.CopyFrom(rt);
            using var tuned = baseEncoder.WithAdapter(adapter);
            float valAcc   = (await EmbeddingEvaluation.RetrievalAsync(tuned, validation, minScore: 0.6f, ct)).Accuracy;
            float valSpear = haveScores ? await EmbeddingEvaluation.SpearmanAsync(tuned, validation, ct) : float.NaN;
            float primary  = haveScores ? valSpear : valAcc;
            bool isBest = primary > bestPrimary;
            if (isBest) { bestPrimary = primary; bestSeed = Clone(adapter); bestAcc = valAcc; bestSpear = valSpear; sinceImprovement = 0; }
            else sinceImprovement++;

            var m = new BertEpochMetrics(seed, epoch, (float)(epochLoss / Math.Max(1, batches)), float.NaN, valAcc, valSpear, isBest);
            metrics.Add(m);
            options.OnEpoch?.Invoke(m);
            options.Logger?.LogInformation(
                "seed {Seed} epoch {Epoch}/{Epochs}  loss {Loss:0.0000}  val_acc {Acc:0.000}  val_spearman {Spear:0.000}{Best}",
                seed, epoch, options.Epochs, m.TrainLoss, valAcc, valSpear, isBest ? "  *best" : "");

            if (options.Patience > 0 && sinceImprovement >= options.Patience)
            {
                options.Logger?.LogInformation("seed {Seed}: early stopping at epoch {Epoch} (no improvement for {Patience} epoch(s)).", seed, epoch, options.Patience);
                break;
            }
        }

        return (bestSeed ?? Clone(adapter), metrics, bestAcc, bestSpear);
    }

    // ----- one optimization step ------------------------------------------------------------------

    private static double TrainBatch(
        BertModel model, LoraRuntime rt, List<Example> examples, ArraySegment<int> batch, List<string>[] minedNeg,
        int[] dims, int fullDim, BertLoraTrainingOptions options, ref float temp,
        Func<string, int[]> ids, AdamW adam, float lr, ref float tempM, ref float tempV, bool contrastive)
    {
        // Intern every unique (prefixed) string in the batch and run one forward per unique text.
        var uniq   = new Dictionary<string, int>(StringComparer.Ordinal);
        var graphs = new List<Graph>();
        var zs     = new List<Tensor>();

        int Intern(string text)
        {
            if (uniq.TryGetValue(text, out int id)) return id;
            id = zs.Count; uniq[text] = id;
            var g = new Graph();
            var z = model.Forward(g, ids(text), rt);
            graphs.Add(g); zs.Add(z);
            return id;
        }

        int n = batch.Count;
        var aIdx = new int[n]; var pIdx = new int[n]; var scores = new float[n];
        List<int>[] extraNeg = contrastive ? new List<int>[n] : null;

        for (int b = 0; b < n; b++)
        {
            var ex = examples[batch[b]];
            aIdx[b] = Intern(ex.AnchorPrefixed);
            pIdx[b] = Intern(ex.PositivePrefixed);
            scores[b] = ex.Score;
            if (contrastive)
            {
                var negs = new List<int>();
                if (options.UseExplicitNegatives && ex.NegativePrefixed != null) negs.Add(Intern(ex.NegativePrefixed));
                if (minedNeg != null && minedNeg[batch[b]] != null)
                    foreach (var t in minedNeg[batch[b]]) negs.Add(Intern(t));
                extraNeg[b] = negs;
            }
        }

        // False-negative masks (index equality captures text equality since we deduped by string).
        bool[][] allowP = null, allowQ = null;
        if (contrastive)
        {
            allowP = new bool[n][]; allowQ = new bool[n][];
            for (int i = 0; i < n; i++)
            {
                allowP[i] = new bool[n]; allowQ[i] = new bool[n];
                for (int j = 0; j < n; j++)
                {
                    if (i == j) continue;
                    bool okP = !options.MaskFalseNegatives || (pIdx[j] != pIdx[i] && pIdx[j] != aIdx[i]);
                    bool okQ = !options.MaskFalseNegatives || (aIdx[j] != aIdx[i] && aIdx[j] != pIdx[i]);
                    allowP[i][j] = okP; allowQ[i][j] = okQ;
                }
            }
        }

        int u = zs.Count;
        double totalLoss = 0, totalDTemp = 0;

        foreach (int d in dims)
        {
            // Normalize the leading-d prefix of each pooled vector.
            var vecs  = new float[u][];
            var norms = new float[u];
            for (int t = 0; t < u; t++)
            {
                var zt = zs[t].Data;
                float ss = 0; for (int k = 0; k < d; k++) ss += zt[k] * zt[k];
                float nrm = MathF.Max(MathF.Sqrt(ss), 1e-12f);
                norms[t] = nrm;
                var uu = new float[d];
                float invn = 1f / nrm;
                for (int k = 0; k < d; k++) uu[k] = zt[k] * invn;
                vecs[t] = uu;
            }

            var gU = new float[u][];
            for (int t = 0; t < u; t++) gU[t] = new float[d];

            double loss, dTemp = 0;
            switch (options.Objective)
            {
                case BertTrainingObjective.Contrastive:
                    loss = LoraLosses.Contrastive(vecs, aIdx, pIdx, extraNeg, allowP, allowQ, temp, gU, out dTemp); break;
                case BertTrainingObjective.CoSent:
                    loss = LoraLosses.CoSent(vecs, aIdx, pIdx, scores, temp, gU, out dTemp); break;
                default:
                    loss = LoraLosses.Regression(vecs, aIdx, pIdx, scores, gU); break;
            }
            totalLoss += loss / dims.Length;
            totalDTemp += dTemp;

            // Backprop the prefix normalization into z.Grad[0..d): dz = (gU - (gU·u)u)/norm.
            for (int t = 0; t < u; t++)
            {
                var uu = vecs[t]; var g = gU[t]; var gz = zs[t].Grad;
                float dot = 0; for (int k = 0; k < d; k++) dot += g[k] * uu[k];
                float invn = 1f / norms[t] / dims.Length;
                for (int k = 0; k < d; k++) gz[k] += (g[k] - dot * uu[k]) * invn;
            }
        }

        // Backward each unique text's graph (accumulates into the shared LoRA params), then step.
        adam.ClearGrads();
        for (int t = 0; t < u; t++) graphs[t].Backward();
        adam.Step(lr);

        if (options.LearnableTemperature && (options.Objective != BertTrainingObjective.CosineRegression))
        {
            float g = (float)(totalDTemp / dims.Length);
            tempM = options.Beta1 * tempM + (1 - options.Beta1) * g;
            tempV = options.Beta2 * tempV + (1 - options.Beta2) * g * g;
            temp -= lr * tempM / (MathF.Sqrt(tempV) + options.Epsilon);
            temp = Math.Clamp(temp, 1e-3f, 1f);
        }

        return totalLoss;
    }

    // ----- helpers --------------------------------------------------------------------------------

    private readonly record struct Example(string AnchorPrefixed, string PositivePrefixed, string NegativePrefixed, float Score);

    private static List<Example> BuildExamples(SentencePairDataset train, BertLoraTrainingOptions options, bool contrastive)
    {
        string q = options.QueryPrefix ?? "";
        string doc = options.DocumentPrefix ?? "";
        var list = new List<Example>(train.Count);
        foreach (var p in train.Pairs)
        {
            if (contrastive)
            {
                if (p.Score.HasValue && p.Score.Value < options.PositiveScoreThreshold) continue;
            }
            else
            {
                if (!p.Score.HasValue) continue;
            }
            string neg = (contrastive && options.UseExplicitNegatives && p.Negative != null) ? doc + p.Negative : null;
            list.Add(new Example(q + p.Anchor, doc + p.Positive, neg, p.Score ?? 1f));
        }
        return list;
    }

    // Embed all anchors/positives with the current adapter (no grad) and pick each anchor's nearest
    // foreign positives as hard negatives (skipping near-duplicates likely to be false negatives).
    private static List<string>[] MineNegatives(
        SentenceEncoder baseEncoder, LoraAdapter adapter, LoraRuntime rt, List<Example> examples,
        BertLoraTrainingOptions options, Func<string, int[]> ids, CancellationToken ct)
    {
        var model = baseEncoder.ModelInternal;
        int nEx = examples.Count;

        // Unique positive texts and their embeddings.
        var posText = new List<string>();
        var posOf   = new int[nEx];
        var posIndex = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int i = 0; i < nEx; i++)
        {
            if (!posIndex.TryGetValue(examples[i].PositivePrefixed, out int id))
            { id = posText.Count; posIndex[examples[i].PositivePrefixed] = id; posText.Add(examples[i].PositivePrefixed); }
            posOf[i] = id;
        }
        var posVec = new float[posText.Count][];
        for (int i = 0; i < posText.Count; i++) posVec[i] = EmbedNoGrad(model, rt, ids(posText[i]));

        var result = new List<string>[nEx];
        int k = options.MinedNegativesPerAnchor;
        for (int i = 0; i < nEx; i++)
        {
            ct.ThrowIfCancellationRequested();
            var a = EmbedNoGrad(model, rt, ids(examples[i].AnchorPrefixed));
            // score against all positives except this example's own positive
            var scored = new List<(int idx, float sim)>(posText.Count);
            for (int j = 0; j < posText.Count; j++)
            {
                if (j == posOf[i]) continue;
                float sim = TensorPrimitives.Dot(a, posVec[j]);
                if (sim > options.MinedNegativeMaxCosine) continue; // likely false negative
                scored.Add((j, sim));
            }
            scored.Sort((x, y) => y.sim.CompareTo(x.sim));
            var negs = new List<string>(k);
            for (int t = 0; t < scored.Count && negs.Count < k; t++) negs.Add(posText[scored[t].idx]);
            result[i] = negs;
        }
        return result;
    }

    private static float[] EmbedNoGrad(BertModel model, LoraRuntime rt, int[] ids)
    {
        var g = new Graph();
        var z = model.Forward(g, ids, rt);
        var v = (float[])z.Data.Clone();
        float norm = MathF.Max(MathF.Sqrt(TensorPrimitives.Dot(v, v)), 1e-12f);
        TensorPrimitives.Divide(v, norm, v);
        return v;
    }

    private static float ScheduleLr(BertLoraTrainingOptions o, int step, int totalSteps)
    {
        int warmup = (int)(o.WarmupFraction * totalSteps);
        if (warmup > 0 && step < warmup) return o.LearningRate * (step + 1) / warmup;
        if (!o.CosineDecay) return o.LearningRate;
        float progress = totalSteps > warmup ? (float)(step - warmup) / (totalSteps - warmup) : 1f;
        progress = Math.Clamp(progress, 0f, 1f);
        return o.LearningRate * 0.5f * (1f + MathF.Cos(MathF.PI * progress));
    }

    private static void Shuffle(int[] a, Random rng)
    {
        for (int i = a.Length - 1; i > 0; i--) { int j = rng.Next(i + 1); (a[i], a[j]) = (a[j], a[i]); }
    }

    private static LoraAdapter Clone(LoraAdapter a)
    {
        using var ms = new MemoryStream();
        a.Save(ms); ms.Position = 0;
        return LoraAdapter.Load(ms);
    }

    // ----- AdamW ----------------------------------------------------------------------------------

    private sealed class AdamW
    {
        private readonly List<Tensor> _params;
        private readonly float[][] _m, _v;
        private readonly float _beta1, _beta2, _eps, _wd;
        private int _t;

        public AdamW(List<Tensor> parameters, BertLoraTrainingOptions o)
        {
            _params = parameters;
            _m = new float[parameters.Count][];
            _v = new float[parameters.Count][];
            for (int i = 0; i < parameters.Count; i++) { _m[i] = new float[parameters[i].Data.Length]; _v[i] = new float[parameters[i].Data.Length]; }
            _beta1 = o.Beta1; _beta2 = o.Beta2; _eps = o.Epsilon; _wd = o.WeightDecay;
        }

        public void ClearGrads() { foreach (var p in _params) p.ClearGrad(); }

        public void Step(float lr)
        {
            _t++;
            float bc1 = 1f - MathF.Pow(_beta1, _t);
            float bc2 = 1f - MathF.Pow(_beta2, _t);
            for (int pi = 0; pi < _params.Count; pi++)
            {
                var data = _params[pi].Data; var grad = _params[pi].Grad; var m = _m[pi]; var v = _v[pi];
                for (int i = 0; i < data.Length; i++)
                {
                    float gg = grad[i];
                    m[i] = _beta1 * m[i] + (1 - _beta1) * gg;
                    v[i] = _beta2 * v[i] + (1 - _beta2) * gg * gg;
                    float mHat = m[i] / bc1, vHat = v[i] / bc2;
                    if (_wd != 0f) data[i] -= lr * _wd * data[i];
                    data[i] -= lr * mHat / (MathF.Sqrt(vHat) + _eps);
                }
            }
        }
    }
}
