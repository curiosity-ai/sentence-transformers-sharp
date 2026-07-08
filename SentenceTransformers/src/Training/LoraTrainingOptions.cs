namespace SentenceTransformers.Training;

/// <summary>Which loss the trainer optimizes.</summary>
public enum TrainingObjective
{
    /// <summary>
    /// Symmetric InfoNCE contrastive loss with in-batch negatives. Pulls each anchor towards its
    /// positive and away from the other positives in the batch. Uses only the pairs at or above
    /// <see cref="LoraTrainingOptions.PositiveScoreThreshold"/> (or all pairs when unscored). Best when
    /// the goal is retrieval / nearest-neighbour separation.
    /// </summary>
    Contrastive,

    /// <summary>
    /// Cosine-similarity regression: minimizes the mean squared error between each pair's adapted cosine
    /// similarity and its gold <see cref="SentencePair.Score"/>. Uses <b>all</b> scored pairs (including
    /// dissimilar ones), so it directly shapes the full graded ordering — the objective STS Spearman
    /// rewards. Requires every training pair to carry a score.
    /// </summary>
    CosineRegression,
}

/// <summary>
/// Hyper-parameters for <see cref="LoraTrainer"/>. Defaults are reasonable for fine-tuning a
/// sentence-embedding adapter on a few thousand related pairs.
/// </summary>
public sealed class LoraTrainingOptions
{
    /// <summary>The loss to optimize. See <see cref="TrainingObjective"/>.</summary>
    public TrainingObjective Objective { get; set; } = TrainingObjective.Contrastive;

    /// <summary>Low-rank bottleneck size of the adapter.</summary>
    public int Rank { get; set; } = 16;

    /// <summary>LoRA <c>α</c> scaling numerator. The residual scale is <c>α / rank</c>. Defaults to <see cref="Rank"/> when null.</summary>
    public float? Alpha { get; set; }

    /// <summary>Number of passes over the training data.</summary>
    public int Epochs { get; set; } = 20;

    /// <summary>Number of pairs per contrastive batch. Larger batches give more in-batch negatives.</summary>
    public int BatchSize { get; set; } = 32;

    /// <summary>Adam learning rate.</summary>
    public float LearningRate { get; set; } = 1e-3f;

    /// <summary>Decoupled (AdamW) weight decay applied to the adapter matrices.</summary>
    public float WeightDecay { get; set; } = 1e-4f;

    /// <summary>Softmax temperature for the InfoNCE contrastive loss. Lower = sharper.</summary>
    public float Temperature { get; set; } = 0.05f;

    /// <summary>Fraction of the training pairs held out for validation.</summary>
    public float ValidationFraction { get; set; } = 0.1f;

    /// <summary>
    /// When pairs carry similarity scores in <c>[0,1]</c>, only pairs at or above this threshold are
    /// used as positives for contrastive training. Pairs without scores are always treated as positives.
    /// </summary>
    public float PositiveScoreThreshold { get; set; } = 0.6f;

    /// <summary>Master seed for weight init, the train/val split and batch shuffling.</summary>
    public int Seed { get; set; } = 42;

    /// <summary>Adam first-moment decay.</summary>
    public float Beta1 { get; set; } = 0.9f;

    /// <summary>Adam second-moment decay.</summary>
    public float Beta2 { get; set; } = 0.999f;

    /// <summary>Adam numerical-stability epsilon.</summary>
    public float Epsilon { get; set; } = 1e-8f;

    /// <summary>Optional per-epoch progress callback for logging.</summary>
    public Action<EpochMetrics> OnEpoch { get; set; }
}

/// <summary>Metrics captured at the end of a training epoch.</summary>
/// <param name="Epoch">1-based epoch number.</param>
/// <param name="TrainLoss">Mean contrastive loss over the training batches this epoch.</param>
/// <param name="ValidationLoss">Contrastive loss over the validation pairs (NaN when there is no validation set).</param>
/// <param name="ValidationAccuracy">Top-1 in-batch retrieval accuracy on the validation pairs.</param>
/// <param name="ValidationSpearman">STS-style Spearman correlation on the validation pairs (NaN when pairs have no scores).</param>
/// <param name="IsBest">True when this epoch produced the best validation metric seen so far.</param>
public readonly record struct EpochMetrics(
    int   Epoch,
    float TrainLoss,
    float ValidationLoss,
    float ValidationAccuracy,
    float ValidationSpearman,
    bool  IsBest);

/// <summary>Summary returned by <see cref="LoraTrainer"/> once training completes.</summary>
/// <param name="Adapter">The best adapter found during training (by validation metric).</param>
/// <param name="Epochs">Per-epoch metrics, in order.</param>
/// <param name="BaselineAccuracy">Validation retrieval accuracy of the un-adapted base encoder.</param>
/// <param name="BaselineSpearman">Validation Spearman of the un-adapted base encoder (NaN without scores).</param>
/// <param name="BestAccuracy">Validation retrieval accuracy of the returned adapter.</param>
/// <param name="BestSpearman">Validation Spearman of the returned adapter (NaN without scores).</param>
public sealed record LoraTrainingReport(
    LoraAdapter        Adapter,
    IReadOnlyList<EpochMetrics> Epochs,
    float              BaselineAccuracy,
    float              BaselineSpearman,
    float              BestAccuracy,
    float              BestSpearman);
