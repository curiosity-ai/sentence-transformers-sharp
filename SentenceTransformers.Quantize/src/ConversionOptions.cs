using SentenceTransformers.Stq;

namespace SentenceTransformers.Quantize;

/// <summary>Everything the converter needs to turn a safetensors checkpoint into an <c>.stq</c> file.</summary>
public sealed record ConversionOptions
{
    public required string InputPath { get; init; }
    public required string OutputPath { get; init; }

    /// <summary>
    /// Band for the transformer projections. Defaults to 4-bit rather than ternary: on this model
    /// ternary projections do not survive post-training quantization (see <c>QUANTIZATION.md</c> §4),
    /// while 4-bit ones clear the shipped <c>Int4</c> path's quality at a third of its memory.
    /// </summary>
    public StqBand Band { get; init; } = StqBand.Q4_0;

    /// <summary>
    /// Band for the token embedding table, which is 63% of Harrier Small's parameters. Defaults to
    /// 4-bit, which quantizes the table for the first time (the load-time <c>Int8</c>/<c>Int4</c>
    /// modes leave it in bfloat16) while keeping quality above the shipped <c>Int4</c> path. Set it
    /// to <c>TQ1_0</c> for the smallest file - the table is the one tensor in the model that
    /// ternarizes usefully - or to a float band to leave it alone for an ablation.
    /// </summary>
    public StqBand EmbeddingBand { get; init; } = StqBand.Q4_0;

    /// <summary>Group size for the ternary bands.</summary>
    public int GroupSize { get; init; } = StqFormat.DefaultGroupSize;

    /// <summary>Group size for the 4-bit band. 128 by default: it is both smaller on disk (4.125
    /// bits/weight against 32's 4.5) and substantially faster to run, because the kernel reduces and
    /// rescales once per group. Finer groups buy no measurable quality here.</summary>
    public int Int4GroupSize { get; init; } = StqFormat.DefaultGroupSizeFor(StqBand.Q4_0);

    public TernaryMethod Method { get; init; } = TernaryMethod.Optimal;

    /// <summary>Which tensors are stored in the Hadamard-rotated basis.</summary>
    public enum RotationScope
    {
        /// <summary>Every packed tensor. The most accurate, and the most work at inference: each
        /// projection group has to rotate its activation before the matmul.</summary>
        All,

        /// <summary>The token embedding table only. A lookup is un-rotated once per token, which is
        /// nothing next to the layers after it, and no projection needs a rotated activation - so
        /// this keeps rotation where it pays most (63% of the parameters) and drops the per-layer
        /// activation rotation entirely.</summary>
        EmbeddingOnly,

        /// <summary>Nothing. Ablation only - ternary quantization without rotation is markedly
        /// worse, and it is also the required setting for a checkpoint already trained ternary in
        /// the original basis (see QUANTIZATION.md §5).</summary>
        None,
    }

    /// <summary>Store weights in the Hadamard-rotated basis. See <see cref="RotationScope"/>.</summary>
    public RotationScope Rotation { get; init; } = RotationScope.All;

    /// <summary>True when any tensor is rotated.</summary>
    public bool Rotate => Rotation != RotationScope.None;

    /// <summary>Whether a given tensor is stored rotated.</summary>
    public bool RotateTensor(string tensorName) => Rotation switch
    {
        RotationScope.All => true,
        RotationScope.EmbeddingOnly => IsEmbedding(tensorName),
        _ => false,
    };

    /// <summary>Upper bound on the Walsh-Hadamard block; the actual block per dimension is the
    /// largest power of two below it that divides that dimension.</summary>
    public int MaxRotationBlock { get; init; } = 1024;

    /// <summary>Seed for the rotations' sign diagonals. Recorded in the file's metadata; the signs
    /// themselves are written out, so a reader never needs the seed.</summary>
    public ulong Seed { get; init; } = 0x5EED_5EED_5EED_5EEDUL;

    /// <summary>Tensor names (exact match) to leave in float32 regardless of rank.</summary>
    public HashSet<string> KeepFloat { get; init; } = new(StringComparer.Ordinal);

    /// <summary>Skip the per-tensor reconstruction error measurement. Roughly halves conversion time.</summary>
    public bool SkipErrorReport { get; init; }

    public int MaxDegreeOfParallelism { get; init; } = Environment.ProcessorCount;

    /// <summary>
    /// Group size for the token embedding table, independent of the projections. Defaults to 128 for
    /// either band: the table is where a group size costs real megabytes (a 4-bit table is 86.6 MB at
    /// 128 against 94.4 MB at 32) and it is also where quality is least sensitive to it - measured,
    /// the two differ by 0.00006 in end-to-end cosine.
    /// </summary>
    public int? EmbeddingGroupSize { get; init; } = StqFormat.DefaultGroupSize;

    private static bool IsEmbedding(string tensorName)
        => tensorName.EndsWith("embed_tokens.weight", StringComparison.Ordinal);

    /// <summary>How one tensor is stored.</summary>
    public readonly record struct TensorPolicy(StqBand Band, int GroupSize);

    /// <summary>Resolves the band and group size for a tensor. The embedding table is the only
    /// tensor treated specially, because it is 63% of the parameters and the only one whose band
    /// meaningfully moves the file size on its own.</summary>
    public TensorPolicy PolicyFor(string tensorName)
    {
        bool isEmbedding = IsEmbedding(tensorName);
        var band = isEmbedding ? EmbeddingBand : Band;
        int defaultGroup = band == StqBand.Q4_0 ? Int4GroupSize : GroupSize;
        return new TensorPolicy(band, isEmbedding ? EmbeddingGroupSize ?? defaultGroup : defaultGroup);
    }
}
