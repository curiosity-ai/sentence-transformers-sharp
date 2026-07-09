using Microsoft.Extensions.Logging;
using SentenceTransformers.Bert.Pure.Model;

namespace SentenceTransformers.Bert.Pure.Training;

/// <summary>Which loss the BERT LoRA trainer optimizes.</summary>
public enum BertTrainingObjective
{
    /// <summary>Symmetric InfoNCE (MultipleNegativesRanking) with in-batch + hard negatives. Best for retrieval.</summary>
    Contrastive,

    /// <summary>CoSENT pairwise ranking loss over graded pairs — directly targets STS Spearman; beats MSE.</summary>
    CoSent,

    /// <summary>Mean-squared error between adapted cosine and the gold score (cosine-similarity regression).</summary>
    CosineRegression,
}

/// <summary>
/// Hyper-parameters for <see cref="BertLoraTrainer"/>. Defaults are reasonable for adapting a BERT
/// sentence encoder on a few thousand pairs with real weight-space LoRA injected into the attention
/// projections.
/// </summary>
public sealed class BertLoraTrainingOptions
{
    // ----- adapter shape -----
    public BertTrainingObjective Objective { get; set; } = BertTrainingObjective.Contrastive;
    public int         Rank    { get; set; } = 8;
    public float?      Alpha   { get; set; }
    /// <summary>Which linear projections get a LoRA adapter (default: the four attention projections).</summary>
    public LoraTargets Targets { get; set; } = LoraTargets.Attention;

    // ----- optimization -----
    public int   Epochs       { get; set; } = 10;
    public int   BatchSize    { get; set; } = 16;
    public float LearningRate { get; set; } = 5e-4f;
    public float WeightDecay  { get; set; } = 1e-4f;
    public float Beta1        { get; set; } = 0.9f;
    public float Beta2        { get; set; } = 0.999f;
    public float Epsilon      { get; set; } = 1e-8f;

    /// <summary>Fraction of total steps spent linearly warming the LR up from 0.</summary>
    public float WarmupFraction { get; set; } = 0.1f;
    /// <summary>Cosine-decay the LR to zero over training after warmup.</summary>
    public bool  CosineDecay    { get; set; } = true;

    /// <summary>InfoNCE / CoSENT softmax temperature (lower = sharper).</summary>
    public float Temperature { get; set; } = 0.05f;
    /// <summary>Learn the temperature jointly (CLIP-style) instead of keeping it fixed.</summary>
    public bool  LearnableTemperature { get; set; } = false;

    // ----- data -----
    /// <summary>Truncate training sequences to this many tokens (bounds compute/memory; sentences are short).</summary>
    public int   MaxTokens              { get; set; } = 128;
    public float ValidationFraction     { get; set; } = 0.1f;
    /// <summary>Min score in [0,1] for a scored pair to count as a positive for the contrastive objective.</summary>
    public float PositiveScoreThreshold { get; set; } = 0.6f;

    // ----- hard negatives -----
    /// <summary>Use each pair's explicit <see cref="SentenceTransformers.Training.SentencePair.Negative"/> (if present) as a hard negative.</summary>
    public bool UseExplicitNegatives { get; set; } = true;
    /// <summary>Mine this many hard negatives per anchor from the current embeddings each epoch (0 = off).</summary>
    public int  MinedNegativesPerAnchor { get; set; } = 0;
    /// <summary>Re-mine negatives every N epochs (mining runs a full no-grad embedding pass).</summary>
    public int  MineEveryEpochs { get; set; } = 1;
    /// <summary>Mask in-batch negatives whose text duplicates the row's anchor/positive (false negatives).</summary>
    public bool MaskFalseNegatives { get; set; } = true;
    /// <summary>Skip a mined candidate whose cosine to the anchor exceeds this (likely a false negative).</summary>
    public float MinedNegativeMaxCosine { get; set; } = 0.95f;

    // ----- output head extras -----
    /// <summary>Learn a per-dimension output bias added before normalization (mean-centering / de-anisotropy).</summary>
    public bool UseOutputBias { get; set; } = false;
    /// <summary>Fit a post-hoc whitening transform (ZCA) from the tuned training embeddings and fold it in.</summary>
    public bool ApplyWhitening { get; set; } = false;

    /// <summary>Optional Matryoshka sub-dimensions; the objective is also applied to these leading-prefix
    /// (re-normalized) sub-vectors so truncated embeddings stay good. Null = full dimension only.</summary>
    public int[] MatryoshkaDims { get; set; }

    // ----- asymmetric prefixes (instruction-style) -----
    /// <summary>Prefix prepended to anchors (queries) before encoding, e.g. an instruction. Null = none.</summary>
    public string QueryPrefix    { get; set; }
    /// <summary>Prefix prepended to positives/negatives (documents) before encoding. Null = none.</summary>
    public string DocumentPrefix { get; set; }

    // ----- reproducibility / selection -----
    public int Seed     { get; set; } = 42;
    /// <summary>Train this many adapters with consecutive seeds and keep the one with the best validation
    /// metric (results are seed-sensitive; the frozen weights make extra seeds relatively cheap).</summary>
    public int NumSeeds { get; set; } = 1;

    /// <summary>Early stopping: stop a seed's training after this many epochs with no improvement in the
    /// validation metric (Spearman when scored, else retrieval accuracy). 0 disables it (run all epochs).
    /// The best adapter seen is always the one returned, regardless of this setting.</summary>
    public int Patience { get; set; } = 0;

    // ----- progress reporting -----
    /// <summary>Optional structured progress callback invoked once per epoch.</summary>
    public Action<BertEpochMetrics> OnEpoch { get; set; }

    /// <summary>Optional logger; when set, the trainer logs the baseline, per-epoch metrics, hard-negative
    /// mining, early stopping, whitening and the final summary. Null disables logging.</summary>
    public ILogger Logger { get; set; }
}

/// <summary>Metrics captured at the end of a training epoch.</summary>
public readonly record struct BertEpochMetrics(
    int   Seed,
    int   Epoch,
    float TrainLoss,
    float ValidationLoss,
    float ValidationAccuracy,
    float ValidationSpearman,
    bool  IsBest);

/// <summary>Summary returned once training completes.</summary>
public sealed record BertLoraTrainingReport(
    LoraAdapter                 Adapter,
    IReadOnlyList<BertEpochMetrics> Epochs,
    float BaselineAccuracy,
    float BaselineSpearman,
    float BestAccuracy,
    float BestSpearman);
