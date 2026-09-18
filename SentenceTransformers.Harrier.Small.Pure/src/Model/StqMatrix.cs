using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.Intrinsics;
using SentenceTransformers;
using SentenceTransformers.Stq;

namespace SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// A linear projection whose weights are packed integer codes with one FP16 scale per group, read
/// straight out of an <c>.stq</c> file. One class covers every packed band, because they differ only
/// in how a group of codes unpacks to signed bytes:
/// <list type="bullet">
/// <item><c>TQ1_0</c> / <c>TQ2_0</c> - ternary <c>{-1, 0, +1}</c> at 1.75 / 2.125 bits per weight,
/// the same representation Bonsai's <c>PTQ1_0</c> / <c>PQ2_0</c> packings use.</item>
/// <item><c>Q4_0</c> - symmetric 4-bit at 4.125-4.5 bits per weight.</item>
/// </list>
/// Once unpacked the kernel is identical: the codes are int8 operands either way.
///
/// <para><b>The rotation.</b> The file stores <c>W R^T</c>, not <c>W</c>. This kernel therefore
/// rotates its activations by <c>R</c> before the dot products, restoring <c>W x</c> exactly (see
/// <see cref="HadamardRotation"/>). The transform is O(inDim log inDim) per position against an
/// O(inDim * outDim) matmul, so it costs well under 1% here - q/k/v each re-rotate the same input
/// rather than sharing one transform, which keeps this class self-contained at that price.</para>
///
/// <para><b>The kernels.</b> Unchanged in shape from <see cref="Int4Matrix"/>: on a VNNI host the
/// codes expand to signed bytes and feed <c>vpdpbusd</c> against dynamically int8-quantized
/// activations; elsewhere each row is dequantized to float and dotted with
/// <see cref="TensorPrimitives.Dot"/>. <see cref="Int4Matrix"/> remains the path for weights
/// quantized at load time from safetensors; this one reads them already packed off disk, which is
/// what lets the embedding table be quantized too.</para>
/// </summary>
internal sealed class StqMatrix : IWeightMatrix
{
    private readonly StqBand _band;
    private readonly byte[] _codes;          // [outDim * rowBytes]
    private readonly float[] _scales;        // [outDim * groups]
    private readonly int[] _groupSum;        // [outDim * groups], sum of the group's trits
    private readonly HadamardRotation _rotation; // null when the tensor is stored unrotated
    private readonly int _groupSize;
    private readonly int _groups;
    private readonly int _groupBytes;
    private readonly int _rowBytes;

    public int InDim { get; }
    public int OutDim { get; }

    private StqMatrix(StqBand band, byte[] codes, float[] scales, int[] groupSum, HadamardRotation rotation,
                          int groupSize, int groups, int groupBytes, int outDim, int inDim)
    {
        _band = band;
        _codes = codes;
        _scales = scales;
        _groupSum = groupSum;
        _rotation = rotation;
        _groupSize = groupSize;
        _groups = groups;
        _groupBytes = groupBytes;
        _rowBytes = groups * groupBytes;
        InDim = inDim;
        OutDim = outDim;
    }

    /// <summary>
    /// Wraps a tensor from an already-loaded <c>.stq</c> file. The packed codes are referenced as-is;
    /// the only derived state is the per-group code sum, which the VNNI path needs to undo the
    /// activation zero-point offset and which is far cheaper to compute once here than per matmul.
    /// </summary>
    public static async Task<StqMatrix> CreateAsync(StqFile file, StqTensorInfo info, ParallelOptions parallelOptions)
    {
        if (!StqFormat.IsPacked(info.Band))
        {
            throw new ArgumentException($"Tensor '{info.Name}' is stored as {StqFormat.BandName(info.Band)}, not a packed band.", nameof(info));
        }
        if (info.InDim % info.GroupSize != 0)
        {
            throw new ArgumentException($"Tensor '{info.Name}': input dim {info.InDim} is not a multiple of its group size {info.GroupSize}.", nameof(info));
        }
        if (info.Shape.Length != 2)
        {
            throw new ArgumentException($"Tensor '{info.Name}' has rank {info.Shape.Length}; a projection must be 2-D.", nameof(info));
        }

        int outDim = info.Shape[0], inDim = info.Shape[1];
        int groupSize = info.GroupSize;
        int groups = info.GroupsPerRow;
        int groupBytes = StqFormat.CodeBytesPerGroup(info.Band, groupSize);
        int rowBytes = groups * groupBytes;

        var codes = file.Codes(info).ToArray();
        var scales = file.Scales(info);
        var groupSum = new int[(long)outDim * groups];
        var band = info.Band;

        await ParallelExecution.ForAsync(0, outDim, parallelOptions, (o, _) =>
        {
            Span<sbyte> group = stackalloc sbyte[groupSize];
            for (int g = 0; g < groups; g++)
            {
                StqPacking.UnpackGroup(band, codes.AsSpan(o * rowBytes + g * groupBytes, groupBytes), group, groupSize);
                int sum = 0;
                for (int i = 0; i < groupSize; i++)
                {
                    sum += group[i];
                }
                groupSum[o * groups + g] = sum;
            }
            return ValueTask.CompletedTask;
        }).ConfigureAwait(false);

        return new StqMatrix(band, codes, scales, groupSum, file.RotationFor(info), groupSize, groups, groupBytes, outDim, inDim);
    }

    public async ValueTask MultiplyAsync(float[] x, float[] y, int seq, ParallelOptions parallelOptions)
    {
        float[] rotated = null;
        try
        {
            if (_rotation is not null)
            {
                // Rotate into scratch rather than in place: `x` is the caller's activation buffer and is
                // reused by sibling projections (q, k and v all read the same normed hidden state).
                rotated = ArrayPool<float>.Shared.Rent(seq * InDim);
                var src = x;
                var dst = rotated;
                int inDim = InDim;
                var rot = _rotation;
                await ParallelExecution.ForAsync(0, seq, parallelOptions, (s, _) =>
                {
                    var row = dst.AsSpan(s * inDim, inDim);
                    src.AsSpan(s * inDim, inDim).CopyTo(row);
                    rot.Apply(row);
                    return ValueTask.CompletedTask;
                }).ConfigureAwait(false);
                x = rotated;
            }

            if (Vnni.IsSupported)
            {
                await MultiplyVnniAsync(x, y, seq, parallelOptions).ConfigureAwait(false);
            }
            else
            {
                await MultiplyFloatAsync(x, y, seq, parallelOptions).ConfigureAwait(false);
            }
        }
        finally
        {
            if (rotated is not null)
            {
                ArrayPool<float>.Shared.Return(rotated);
            }
        }
    }

    private async ValueTask MultiplyVnniAsync(float[] x, float[] y, int seq, ParallelOptions parallelOptions)
    {
        int inDim = InDim, outDim = OutDim;
        var (ua, aScale) = await VnniActivations.QuantizeAsync(x, seq, inDim, parallelOptions).ConfigureAwait(false);
        try
        {
            await ParallelExecution.ForAsync(0, outDim, parallelOptions, (o, _) =>
            {
                VnniColumn(ua, aScale, y, o, seq, inDim, outDim);
                return ValueTask.CompletedTask;
            }).ConfigureAwait(false);
        }
        finally
        {
            VnniActivations.Return(ua, aScale);
        }
    }

    /// <summary>Unpacks one output row's codes once, then dots it against every position. The int32
    /// accumulator cannot overflow: a uint8 activation is at most 255 and a code at most 8 in
    /// magnitude, so even a 128-wide 4-bit group contributes at most 16 * 255 * 8 = 32,640 per
    /// lane.</summary>
    private void VnniColumn(byte[] ua, float[] aScale, float[] y, int o, int seq, int inDim, int outDim)
    {
        Span<sbyte> wbuf = stackalloc sbyte[inDim];
        int rowBase = o * _rowBytes;
        for (int g = 0; g < _groups; g++)
        {
            StqPacking.UnpackGroup(_band, _codes.AsSpan(rowBase + g * _groupBytes, _groupBytes),
                                       wbuf.Slice(g * _groupSize, _groupSize), _groupSize);
        }

        int scaleBase = o * _groups;
        for (int s = 0; s < seq; s++)
        {
            int sBase = s * inDim;
            float facc = 0f;
            for (int g = 0; g < _groups; g++)
            {
                int gStart = g * _groupSize;
                var acc = Vector256<int>.Zero;
                for (int c = 0; c < _groupSize; c += Vector256<byte>.Count)
                {
                    acc = Vnni.DotAccumulate(acc,
                        Vector256.LoadUnsafe(ref ua[sBase + gStart + c]),
                        Vector256.LoadUnsafe(ref wbuf[gStart + c]));
                }
                int dot = Vector256.Sum(acc) - Vnni.ZeroPoint * _groupSum[scaleBase + g];
                facc += _scales[scaleBase + g] * dot;
            }
            y[s * outDim + o] = aScale[s] * facc;
        }
    }

    private ValueTask MultiplyFloatAsync(float[] x, float[] y, int seq, ParallelOptions parallelOptions)
    {
        int inDim = InDim, outDim = OutDim;
        return new ValueTask(ParallelExecution.ForAsync(0, outDim, parallelOptions, (o, _) =>
        {
            FloatColumn(x, y, seq, inDim, outDim, o);
            return ValueTask.CompletedTask;
        }));
    }

    private void FloatColumn(float[] x, float[] y, int seq, int inDim, int outDim, int o)
    {
        Span<float> buf = stackalloc float[inDim];
        int rowBase = o * _rowBytes;
        for (int g = 0; g < _groups; g++)
        {
            StqPacking.UnpackGroupScaled(_band, _codes.AsSpan(rowBase + g * _groupBytes, _groupBytes),
                                             buf.Slice(g * _groupSize, _groupSize), _groupSize, _scales[o * _groups + g]);
        }
        for (int s = 0; s < seq; s++)
        {
            y[s * outDim + o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(x, s * inDim, inDim), buf);
        }
    }

    /// <summary>Bytes this matrix occupies in memory (codes + scales + the derived group sums).</summary>
    public long ResidentBytes => _codes.LongLength + _scales.LongLength * 4 + _groupSum.LongLength * 4;
}

/// <summary>Builds the right <see cref="IWeightMatrix"/> for a tensor in an <c>.stq</c> file,
/// whichever band the converter chose for it - a packed kernel for the ternary and 4-bit bands, the
/// plain float path for a tensor the converter left unquantized.</summary>
internal static class StqWeights
{
    public static async Task<IWeightMatrix> CreateAsync(StqFile file, StqTensorInfo info, int outDim, int inDim, ParallelOptions parallelOptions)
    {
        if (info.Shape is not [var r, var c] || r != outDim || c != inDim)
        {
            throw new InvalidDataException(
                $"Tensor '{info.Name}' has shape [{string.Join(", ", info.Shape)}]; this model expects [{outDim}, {inDim}].");
        }

        if (StqFormat.IsPacked(info.Band))
        {
            return await StqMatrix.CreateAsync(file, info, parallelOptions).ConfigureAwait(false);
        }

        // A converter may leave a tensor in a float band (say, to isolate one during an ablation).
        // Those run through the existing float path unchanged.
        return new FloatMatrix(file.ReadFloat(info.Name), outDim, inDim);
    }
}
