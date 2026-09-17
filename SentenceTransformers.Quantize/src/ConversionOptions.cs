using SentenceTransformers.Ternary;

namespace SentenceTransformers.Quantize;

/// <summary>Everything the converter needs to turn a safetensors checkpoint into an <c>.stq</c> file.</summary>
public sealed class ConversionOptions
{
    public required string InputPath { get; init; }
    public required string OutputPath { get; init; }

    /// <summary>Band for the transformer projections.</summary>
    public TernaryBand Band { get; init; } = TernaryBand.TQ1_0;

    /// <summary>Band for the token embedding table. Defaults to <see cref="Band"/>. The embedding is
    /// 63% of Harrier Small's parameters, so it is the first thing worth an ablation if quality
    /// slips; a float band here leaves it unquantized.</summary>
    public TernaryBand? EmbeddingBand { get; init; }

    public int GroupSize { get; init; } = TernaryFormat.DefaultGroupSize;

    public TernaryMethod Method { get; init; } = TernaryMethod.Optimal;

    /// <summary>Store weights in the Hadamard-rotated basis. Off only for ablations - ternary
    /// quantization without it is markedly worse.</summary>
    public bool Rotate { get; init; } = true;

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

    public TernaryBand BandFor(string tensorName)
        => tensorName.EndsWith("embed_tokens.weight", StringComparison.Ordinal) ? (EmbeddingBand ?? Band) : Band;
}
