using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
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
        if (_rotation is null)
        {
            await MultiplyRotatedAsync(x, y, seq, parallelOptions).ConfigureAwait(false);
            return;
        }

        float[] rotated = await RentRotatedAsync(_rotation, x, seq, InDim, parallelOptions).ConfigureAwait(false);
        try
        {
            await MultiplyRotatedAsync(rotated, y, seq, parallelOptions).ConfigureAwait(false);
        }
        finally
        {
            ArrayPool<float>.Shared.Return(rotated);
        }
    }

    /// <summary>The basis this matrix's weights are stored in, or null when they are unrotated.</summary>
    internal HadamardRotation Rotation => _rotation;

    /// <summary>
    /// The rotation shared by a group of projections that read the same activation, or null if they do
    /// not all share one. Sibling projections normally do - the converter creates one rotation per
    /// input width and every tensor of that width references it - which is what lets the caller rotate
    /// once for the group instead of once per projection.
    /// </summary>
    internal static HadamardRotation SharedRotation(IWeightMatrix a, IWeightMatrix b, IWeightMatrix c = null)
    {
        if (a is not StqMatrix sa || sa._rotation is null || b is not StqMatrix sb || !ReferenceEquals(sa._rotation, sb._rotation))
        {
            return null;
        }
        if (c is not null && (c is not StqMatrix sc || !ReferenceEquals(sa._rotation, sc._rotation)))
        {
            return null;
        }
        return sa._rotation;
    }

    /// <summary>
    /// Rotates <paramref name="x"/> into a pooled buffer the caller must return. Public to the model so
    /// a group of projections sharing a rotation can pay for it once.
    /// </summary>
    internal static async ValueTask<float[]> RentRotatedAsync(HadamardRotation rotation, float[] x, int seq, int inDim, ParallelOptions parallelOptions)
    {
        // Rotate into scratch rather than in place: `x` is the caller's activation buffer and is still
        // needed afterwards (the residual stream, and any sibling projection reading the same input).
        float[] dst = ArrayPool<float>.Shared.Rent(seq * inDim);
        var src = x;

        if (!RunInline(parallelOptions) && (long)seq * inDim >= RotationParallelThreshold && parallelOptions.MaxDegreeOfParallelism > 1)
        {
            // Timed inside the body, not around the await: wrapping a parallel loop on the calling
            // thread measures its wall time, which is a quarter of its CPU cost on four threads and is
            // not comparable with the stages that are timed inline. That mistake made the rotation
            // look like 6% of the profile when it is nearer 16%.
            await ParallelExecution.ForAsync(0, seq, parallelOptions, (sIdx, _) =>
            {
                long ts = ForwardProfile.StageStart();
                var row = dst.AsSpan(sIdx * inDim, inDim);
                src.AsSpan(sIdx * inDim, inDim).CopyTo(row);
                rotation.Apply(row);
                ForwardProfile.StageStop(ForwardProfile.Stage.Rotate, ts);
                return ValueTask.CompletedTask;
            }).ConfigureAwait(false);
        }
        else
        {
            long ts = ForwardProfile.StageStart();
            for (int sIdx = 0; sIdx < seq; sIdx++)
            {
                var row = dst.AsSpan(sIdx * inDim, inDim);
                src.AsSpan(sIdx * inDim, inDim).CopyTo(row);
                rotation.Apply(row);
            }
            ForwardProfile.StageStop(ForwardProfile.Stage.Rotate, ts);
        }
        return dst;
    }

    /// <summary>Multiplies activations that are <b>already</b> in this matrix's basis - either because
    /// the weights are unrotated, or because the caller rotated once for a group of projections.</summary>
    internal async ValueTask MultiplyRotatedAsync(float[] x, float[] y, int seq, ParallelOptions parallelOptions)
    {
        if (Vnni.IsSupported)
        {
            await MultiplyVnniAsync(x, y, seq, parallelOptions).ConfigureAwait(false);
        }
        else
        {
            await MultiplyFloatAsync(x, y, seq, parallelOptions).ConfigureAwait(false);
        }
    }

    private async ValueTask MultiplyVnniAsync(float[] x, float[] y, int seq, ParallelOptions parallelOptions)
    {
        int inDim = InDim, outDim = OutDim;
        // Timed around the await because the quantizer is shared with the Int8 path and is not ours
        // to instrument internally. It therefore reads as wall time of a parallel loop rather than CPU
        // time, so this stage is under-reported relative to the other three - read it as a floor.
        long quantTs = ForwardProfile.StageStart();
        var (ua, aScale) = await VnniActivations.QuantizeAsync(x, seq, inDim, parallelOptions).ConfigureAwait(false);
        ForwardProfile.StageStop(ForwardProfile.Stage.QuantizeActivations, quantTs);
        try
        {
            int tiles = (outDim + TileOut - 1) / TileOut;
            if (RunInline(parallelOptions))
            {
                for (int t = 0; t < tiles; t++)
                {
                    RunTile(ua, aScale, y, t, seq, inDim, outDim);
                }
            }
            else
            {
                await ParallelExecution.ForAsync(0, tiles, parallelOptions, (t, _) =>
                {
                    RunTile(ua, aScale, y, t, seq, inDim, outDim);
                    return ValueTask.CompletedTask;
                }).ConfigureAwait(false);
            }
        }
        finally
        {
            VnniActivations.Return(ua, aScale);
        }
    }

    /// <summary>Output channels computed together per tile.</summary>
    private const int TileOut = 4;

    /// <summary>
    /// When true (the default) and the caller asked for a single thread, the kernel runs its loops
    /// directly instead of through <see cref="ParallelExecution.ForAsync"/>.
    ///
    /// <para>At <c>MaxDegreeOfParallelism = 1</c> that helper already runs sequentially, so this
    /// changes no work - but it still allocates a closure, a delegate and an async state machine per
    /// call, and the packed path makes several of those calls per projection per layer. Settable so
    /// the benchmark can A/B it in one process.</para>
    /// </summary>
    internal static bool SingleThreadFastPath = true;

    private static bool RunInline(ParallelOptions parallelOptions)
        => SingleThreadFastPath && parallelOptions.MaxDegreeOfParallelism <= 1;

    /// <summary>
    /// Activation elements (seq * inDim) below which the rotation runs on the calling thread instead
    /// of through a parallel loop. Zero - always fan out - because that is what measured fastest.
    ///
    /// <para>Serializing it looks attractive: the rotation is a few percent of the profile and each
    /// parallel loop allocates scheduling state (~9 MB per encode pass here). Measured A/B inside one
    /// process it is a 1.53x <i>loss</i> - 8191 ms/iter inline against 5356 parallel - because the
    /// forward pass runs one sequence at a time, so an inline rotation is a serial section in an
    /// otherwise parallel pipeline and Amdahl does the rest. Left settable so the trade can be
    /// re-measured rather than re-argued.</para>
    /// </summary>
    internal static long RotationParallelThreshold;

    /// <summary>
    /// The hot kernel: four output channels and two positions at a time, giving eight independent
    /// int32 accumulator chains.
    ///
    /// <para>This is what the naive one-channel-at-a-time loop gets wrong, and it costs far more than
    /// the packing does. <c>acc = DotAccumulate(acc, ...)</c> is a serial dependency: each step waits
    /// on the previous one's ~5-cycle latency, so a single chain runs at a fraction of the unit's
    /// throughput no matter how cheap the unpack is. Eight chains cover that latency. Measured on a
    /// 640-wide projection this is worth several times more than vectorizing the nibble unpack (~1%)
    /// or widening the scale groups (~7%), both of which were tried first.</para>
    ///
    /// <para>The per-group scales are what stop this being a plain int8 GEMM: each group's dot has to
    /// be reduced and scaled separately, so the accumulators are reset per group rather than run the
    /// length of the row.</para>
    /// </summary>
    private void Tile(byte[] ua, float[] aScale, float[] y, int o0, int seq, int inDim, int outDim)
    {
        Span<sbyte> w = stackalloc sbyte[TileOut * inDim];
        long unpackTs = ForwardProfile.StageStart();
        for (int k = 0; k < TileOut; k++)
        {
            UnpackRow(o0 + k, w.Slice(k * inDim, inDim));
        }
        ForwardProfile.StageStop(ForwardProfile.Stage.UnpackWeights, unpackTs);

        long dotTs = ForwardProfile.StageStart();
        int groups = _groups, gs = _groupSize;
        int width = Vector256<byte>.Count;
        ref byte uaRef = ref MemoryMarshal.GetArrayDataReference(ua);
        ref sbyte wRef = ref MemoryMarshal.GetReference(w);

        for (int s = 0; s < seq; s += 2)
        {
            bool two = s + 1 < seq;
            int sb0 = s * inDim;
            int sb1 = two ? sb0 + inDim : sb0;

            float f00 = 0, f01 = 0, f02 = 0, f03 = 0;
            float f10 = 0, f11 = 0, f12 = 0, f13 = 0;

            for (int g = 0; g < groups; g++)
            {
                int gStart = g * gs;
                var a00 = Vector256<int>.Zero; var a01 = Vector256<int>.Zero;
                var a02 = Vector256<int>.Zero; var a03 = Vector256<int>.Zero;
                var a10 = Vector256<int>.Zero; var a11 = Vector256<int>.Zero;
                var a12 = Vector256<int>.Zero; var a13 = Vector256<int>.Zero;

                for (int c = 0; c < gs; c += width)
                {
                    int off = gStart + c;
                    var w0 = Vector256.LoadUnsafe(ref wRef, (nuint)off);
                    var w1 = Vector256.LoadUnsafe(ref wRef, (nuint)(inDim + off));
                    var w2 = Vector256.LoadUnsafe(ref wRef, (nuint)(2 * inDim + off));
                    var w3 = Vector256.LoadUnsafe(ref wRef, (nuint)(3 * inDim + off));

                    var av0 = Vector256.LoadUnsafe(ref uaRef, (nuint)(sb0 + off));
                    a00 = Vnni.DotAccumulate(a00, av0, w0);
                    a01 = Vnni.DotAccumulate(a01, av0, w1);
                    a02 = Vnni.DotAccumulate(a02, av0, w2);
                    a03 = Vnni.DotAccumulate(a03, av0, w3);

                    if (two)
                    {
                        var av1 = Vector256.LoadUnsafe(ref uaRef, (nuint)(sb1 + off));
                        a10 = Vnni.DotAccumulate(a10, av1, w0);
                        a11 = Vnni.DotAccumulate(a11, av1, w1);
                        a12 = Vnni.DotAccumulate(a12, av1, w2);
                        a13 = Vnni.DotAccumulate(a13, av1, w3);
                    }
                }

                int gi = g;
                float sc0 = _scales[(o0) * groups + gi],     sc1 = _scales[(o0 + 1) * groups + gi];
                float sc2 = _scales[(o0 + 2) * groups + gi], sc3 = _scales[(o0 + 3) * groups + gi];
                int z0 = Vnni.ZeroPoint * _groupSum[(o0) * groups + gi];
                int z1 = Vnni.ZeroPoint * _groupSum[(o0 + 1) * groups + gi];
                int z2 = Vnni.ZeroPoint * _groupSum[(o0 + 2) * groups + gi];
                int z3 = Vnni.ZeroPoint * _groupSum[(o0 + 3) * groups + gi];

                f00 += sc0 * (Vector256.Sum(a00) - z0);
                f01 += sc1 * (Vector256.Sum(a01) - z1);
                f02 += sc2 * (Vector256.Sum(a02) - z2);
                f03 += sc3 * (Vector256.Sum(a03) - z3);

                if (two)
                {
                    f10 += sc0 * (Vector256.Sum(a10) - z0);
                    f11 += sc1 * (Vector256.Sum(a11) - z1);
                    f12 += sc2 * (Vector256.Sum(a12) - z2);
                    f13 += sc3 * (Vector256.Sum(a13) - z3);
                }
            }

            float as0 = aScale[s];
            int yb0 = s * outDim + o0;
            y[yb0] = as0 * f00; y[yb0 + 1] = as0 * f01; y[yb0 + 2] = as0 * f02; y[yb0 + 3] = as0 * f03;

            if (two)
            {
                float as1 = aScale[s + 1];
                int yb1 = (s + 1) * outDim + o0;
                y[yb1] = as1 * f10; y[yb1 + 1] = as1 * f11; y[yb1 + 2] = as1 * f12; y[yb1 + 3] = as1 * f13;
            }
        }
        ForwardProfile.StageStop(ForwardProfile.Stage.Dot, dotTs);
    }

    /// <summary>Unpacks one output row's packed codes into signed bytes.</summary>
    private void UnpackRow(int o, Span<sbyte> dst)
    {
        int rowBase = o * _rowBytes;
        for (int g = 0; g < _groups; g++)
        {
            StqPacking.UnpackGroup(_band, _codes.AsSpan(rowBase + g * _groupBytes, _groupBytes),
                                   dst.Slice(g * _groupSize, _groupSize), _groupSize);
        }
    }

    /// <summary>One output tile, or the ragged remainder when the tile would run past the last
    /// output channel.</summary>
    private void RunTile(byte[] ua, float[] aScale, float[] y, int t, int seq, int inDim, int outDim)
    {
        int o0 = t * TileOut;
        if (o0 + TileOut <= outDim)
        {
            Tile(ua, aScale, y, o0, seq, inDim, outDim);
        }
        else
        {
            for (int o = o0; o < outDim; o++)
            {
                VnniColumn(ua, aScale, y, o, seq, inDim, outDim);
            }
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
        if (RunInline(parallelOptions))
        {
            for (int o = 0; o < outDim; o++)
            {
                FloatColumn(x, y, seq, inDim, outDim, o);
            }
            return ValueTask.CompletedTask;
        }

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
