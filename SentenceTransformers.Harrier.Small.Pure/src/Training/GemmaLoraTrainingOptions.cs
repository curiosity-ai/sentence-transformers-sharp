using Microsoft.Extensions.Logging;

namespace SentenceTransformers.Harrier.Small.Pure.Training;

/// <summary>Which loss the Gemma LoRA trainer optimizes (see <c>SentenceTransformers.Training.LoraLosses</c>).</summary>
public enum GemmaTrainingObjective
{
    Contrastive,
    CoSent,
    CosineRegression,
}

/// <summary>Hyper-parameters for <see cref="Gemma3LoraTrainer"/>. Mirrors the BERT trainer's options; the
/// same Tier-1/2/3 features apply (hard negatives, CoSENT, warmup+cosine LR, learned temperature,
/// multi-seed, output-centering bias, whitening, Matryoshka, prefixes).</summary>
public sealed class GemmaLoraTrainingOptions
{
    public GemmaTrainingObjective Objective { get; set; } = GemmaTrainingObjective.Contrastive;
    public int   Rank    { get; set; } = 8;
    public float? Alpha  { get; set; }
    public GemmaLoraTargets Targets { get; set; } = GemmaLoraTargets.Attention;

    public int   Epochs       { get; set; } = 5;
    public int   BatchSize    { get; set; } = 8;
    public float LearningRate { get; set; } = 5e-4f;
    public float WeightDecay  { get; set; } = 1e-4f;
    public float Beta1        { get; set; } = 0.9f;
    public float Beta2        { get; set; } = 0.999f;
    public float Epsilon      { get; set; } = 1e-8f;

    public float WarmupFraction { get; set; } = 0.1f;
    public bool  CosineDecay    { get; set; } = true;

    public float Temperature          { get; set; } = 0.05f;
    public bool  LearnableTemperature { get; set; } = false;

    public int   MaxTokens              { get; set; } = 64;
    public float ValidationFraction     { get; set; } = 0.1f;
    public float PositiveScoreThreshold { get; set; } = 0.6f;

    public bool  UseExplicitNegatives    { get; set; } = true;
    public int   MinedNegativesPerAnchor { get; set; } = 0;
    public int   MineEveryEpochs         { get; set; } = 1;
    public bool  MaskFalseNegatives      { get; set; } = true;
    public float MinedNegativeMaxCosine  { get; set; } = 0.95f;

    public bool  UseOutputBias  { get; set; } = false;
    public bool  ApplyWhitening { get; set; } = false;
    public int[] MatryoshkaDims { get; set; }

    public string QueryPrefix    { get; set; }
    public string DocumentPrefix { get; set; }

    public int Seed     { get; set; } = 42;
    public int NumSeeds { get; set; } = 1;

    /// <summary>Early stopping: stop a seed after this many epochs with no validation improvement (0 = off).
    /// The best adapter seen is always returned regardless.</summary>
    public int Patience { get; set; } = 0;

    /// <summary>Optional structured progress callback invoked once per epoch.</summary>
    public Action<GemmaEpochMetrics> OnEpoch { get; set; }

    /// <summary>Optional logger; when set, the trainer logs baseline, per-epoch metrics, mining, early
    /// stopping, whitening and the final summary. Null disables logging.</summary>
    public ILogger Logger { get; set; }
}

public readonly record struct GemmaEpochMetrics(int Seed, int Epoch, float TrainLoss, float ValidationAccuracy, float ValidationSpearman, bool IsBest);

public sealed record GemmaLoraTrainingReport(
    GemmaLoraAdapter Adapter,
    IReadOnlyList<GemmaEpochMetrics> Epochs,
    float BaselineAccuracy, float BaselineSpearman, float BestAccuracy, float BestSpearman);
