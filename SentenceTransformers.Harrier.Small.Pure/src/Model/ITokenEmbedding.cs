using System.Numerics.Tensors;
using SentenceTransformers.Ternary;
using SentenceTransformers.Harrier.Small.Pure.Numerics;

namespace SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// The token embedding table, read one row at a time. Harrier Small's table is
/// <c>262144 x 640</c> - 167.8M of the checkpoint's 268M parameters, so how it is stored dominates
/// the model's footprint far more than the transformer layers do. The forward pass only ever needs
/// the rows for the tokens in the current sequence, so the interface is a row lookup rather than a
/// materialized matrix.
/// </summary>
internal interface ITokenEmbedding
{
    int HiddenSize { get; }

    /// <summary>Writes token <paramref name="tokenId"/>'s embedding, multiplied by
    /// <paramref name="scale"/>, into <paramref name="dst"/> (exactly <see cref="HiddenSize"/> long).</summary>
    void Lookup(int tokenId, Span<float> dst, float scale);

    /// <summary>Bytes the table occupies in memory.</summary>
    long ResidentBytes { get; }
}

/// <summary>The checkpoint's native storage: bfloat16 rows widened one token at a time (335 MB).</summary>
internal sealed class BFloat16Embedding : ITokenEmbedding
{
    private readonly ushort[] _table;
    public int HiddenSize { get; }

    public BFloat16Embedding(ushort[] table, int hiddenSize)
    {
        _table = table;
        HiddenSize = hiddenSize;
    }

    public void Lookup(int tokenId, Span<float> dst, float scale)
    {
        int h = HiddenSize;
        int src = tokenId * h;
        for (int i = 0; i < h; i++)
        {
            dst[i] = FloatConversions.BFloat16ToSingle(_table[src + i]) * scale;
        }
    }

    public long ResidentBytes => _table.LongLength * 2;
}

/// <summary>
/// The ternary table: five 128-weight groups per row, each a packed code block plus an FP16 scale.
/// At <c>TQ1_0</c> that is 36.7 MB against bfloat16's 335 MB, and it is the single biggest win in
/// the whole conversion.
///
/// <para>The rows are stored rotated, like every other ternary tensor - the rotation is exactly what
/// makes a 640-wide embedding row survive 1.75 bits/weight. A lookup has no activation to push the
/// rotation onto, so the row is un-rotated after unpacking with <see cref="TernaryRotation.ApplyInverse"/>:
/// one Walsh-Hadamard transform over 640 floats per token, which is nothing next to the 18 layers
/// that follow.</para>
/// </summary>
internal sealed class TernaryEmbedding : ITokenEmbedding
{
    private readonly TernaryBand _band;
    private readonly byte[] _codes;
    private readonly float[] _scales;
    private readonly TernaryRotation _rotation;
    private readonly int _groupSize;
    private readonly int _groups;
    private readonly int _groupBytes;
    private readonly int _rowBytes;
    private readonly int _vocabSize;

    public int HiddenSize { get; }

    public TernaryEmbedding(TernaryModelFile file, TernaryTensorInfo info)
    {
        if (!TernaryFormat.IsTernary(info.Band))
        {
            throw new ArgumentException($"Tensor '{info.Name}' is stored as {TernaryFormat.BandName(info.Band)}, not a ternary band.", nameof(info));
        }

        _band = info.Band;
        _codes = file.Codes(info).ToArray();
        _scales = file.Scales(info);
        _rotation = file.RotationFor(info);
        _groupSize = info.GroupSize;
        _groups = info.GroupsPerRow;
        _groupBytes = TernaryFormat.CodeBytesPerGroup(info.Band, info.GroupSize);
        _rowBytes = _groups * _groupBytes;
        _vocabSize = info.Shape[0];
        HiddenSize = info.Shape[1];
    }

    public void Lookup(int tokenId, Span<float> dst, float scale)
    {
        if ((uint)tokenId >= (uint)_vocabSize)
        {
            throw new ArgumentOutOfRangeException(nameof(tokenId), tokenId, $"Token id is outside the {_vocabSize}-entry vocabulary.");
        }

        int rowBase = tokenId * _rowBytes;
        int scaleBase = tokenId * _groups;
        for (int g = 0; g < _groups; g++)
        {
            TernaryPacking.UnpackGroupScaled(_band, _codes.AsSpan(rowBase + g * _groupBytes, _groupBytes),
                                             dst.Slice(g * _groupSize, _groupSize), _groupSize, _scales[scaleBase + g]);
        }
        _rotation?.ApplyInverse(dst);
        TensorPrimitives.Multiply(dst, scale, dst);
    }

    public long ResidentBytes => _codes.LongLength + _scales.LongLength * 4;
}

/// <summary>A float32 table, for a converter that chose to leave the embeddings unquantized.</summary>
internal sealed class FloatEmbedding : ITokenEmbedding
{
    private readonly float[] _table;
    public int HiddenSize { get; }

    public FloatEmbedding(float[] table, int hiddenSize)
    {
        _table = table;
        HiddenSize = hiddenSize;
    }

    public void Lookup(int tokenId, Span<float> dst, float scale)
        => TensorPrimitives.Multiply(_table.AsSpan(tokenId * HiddenSize, HiddenSize), scale, dst);

    public long ResidentBytes => _table.LongLength * 4;
}
