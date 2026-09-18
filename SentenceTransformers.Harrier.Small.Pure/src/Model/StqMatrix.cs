using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
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
    // Both are stored group-major - [g * outDim + o], not [o * groups + g] - so a tile's four output
    // channels are contiguous for one group and load as a single Vector128. In row-major order they
    // are strided by `groups`, which would need a gather per group and undo the point of the tile.
    private readonly float[] _scales;        // [groups * outDim]
    // The activation zero-point correction, folded per output channel: sum over groups of
    // scale[g,o] * ZeroPoint * (sum of group g's codes). Every term of it is position-independent, so
    // subtracting it once at the end of a row is exactly equivalent to subtracting each group's share
    // inside the group loop - and takes a load, a multiply and four subtracts per group out of the
    // hottest loop in the kernel, where per-group scales already force a reduction every 128 weights.
    private readonly float[] _zeroBias;      // [outDim]
    private readonly HadamardRotation _rotation; // null when the tensor is stored unrotated
    private readonly int _groupSize;
    private readonly int _groups;
    private readonly int _groupBytes;
    private readonly int _rowBytes;

    public int InDim { get; }
    public int OutDim { get; }

    private StqMatrix(StqBand band, byte[] codes, float[] scales, float[] zeroBias, HadamardRotation rotation,
                          int groupSize, int groups, int groupBytes, int outDim, int inDim)
    {
        _band = band;
        _codes = codes;
        _scales = scales;
        _zeroBias = zeroBias;
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
    /// the only derived state is the per-channel zero-point bias, which the VNNI path needs to undo
    /// the activation offset and which is far cheaper to compute once here than per matmul.
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
        var rowMajorScales = file.Scales(info);
        var scales = new float[(long)groups * outDim];
        var zeroBias = new float[outDim];
        var band = info.Band;

        await ParallelExecution.ForAsync(0, outDim, parallelOptions, (o, _) =>
        {
            Span<sbyte> group = stackalloc sbyte[groupSize];
            float bias = 0f;
            for (int g = 0; g < groups; g++)
            {
                StqPacking.UnpackGroup(band, codes.AsSpan(o * rowBytes + g * groupBytes, groupBytes), group, groupSize);
                int sum = 0;
                for (int i = 0; i < groupSize; i++)
                {
                    sum += group[i];
                }
                float scale = rowMajorScales[(long)o * groups + g];
                scales[(long)g * outDim + o] = scale;
                bias += scale * (Vnni.ZeroPoint * sum);
            }
            zeroBias[o] = bias;
            return ValueTask.CompletedTask;
        }).ConfigureAwait(false);

        return new StqMatrix(band, codes, scales, zeroBias, file.RotationFor(info), groupSize, groups, groupBytes, outDim, inDim);
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
        // Timed around the await because the quantizer is shared with the Int8 path and is not ours
        // to instrument internally. It therefore reads as wall time of a parallel loop rather than CPU
        // time, so this stage is under-reported relative to the other three - read it as a floor.
        long quantTs = ForwardProfile.StageStart();
        var (ua, aScale) = await VnniActivations.QuantizeAsync(x, seq, InDim, parallelOptions).ConfigureAwait(false);
        ForwardProfile.StageStop(ForwardProfile.Stage.QuantizeActivations, quantTs);
        try
        {
            await MultiplyQuantizedAsync(ua, aScale, y, seq, parallelOptions).ConfigureAwait(false);
        }
        finally
        {
            VnniActivations.Return(ua, aScale);
        }
    }

    /// <summary>True when this matrix can consume already-quantized activations - i.e. when the host
    /// has a VNNI-style int8 dot at all.</summary>
    internal bool UsesVnni => Vnni.IsSupported;

    /// <summary>
    /// The matmul from the point where the activations are already quantized, so a group of sibling
    /// projections reading one activation can share that work instead of repeating it per projection.
    /// </summary>
    internal ValueTask MultiplyQuantizedAsync(byte[] ua, float[] aScale, float[] y, int seq, ParallelOptions parallelOptions)
    {
        int inDim = InDim, outDim = OutDim;
        int tiles = (outDim + TileOut - 1) / TileOut;
        if (RunInline(parallelOptions))
        {
            for (int t = 0; t < tiles; t++)
            {
                RunTile(ua, aScale, y, t, seq, inDim, outDim);
            }
            return ValueTask.CompletedTask;
        }

        return new ValueTask(ParallelExecution.ForAsync(0, tiles, parallelOptions, (t, _) =>
        {
            RunTile(ua, aScale, y, t, seq, inDim, outDim);
            return ValueTask.CompletedTask;
        }));
    }

    /// <summary>Output channels computed together per tile.</summary>
    private const int TileOut = 4;

    /// <summary>Positions computed together per tile, so each weight load is reused four times.</summary>
    private const int TilePos = 4;

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
    // SkipLocalsInit: `w` is a few kilobytes and UnpackRow writes every byte of it before anything
    // reads it, so the implicit zeroing is a memset of the entire weight matrix per matmul - about as
    // much store traffic as the unpacking itself.
    [SkipLocalsInit]
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
        ref byte uaRef = ref MemoryMarshal.GetArrayDataReference(ua);
        ref sbyte wRef = ref MemoryMarshal.GetReference(w);

        // Four positions at a time: every weight vector loaded in the inner loop is used against four
        // activations, which is the reuse the Int8 kernel gets and the two-position version did not.
        int s = 0;
        for (; s + TilePos <= seq; s += TilePos)
        {
            Dot4(ref uaRef, ref wRef, aScale, y, o0, s, inDim, outDim);
        }
        for (; s < seq; s++)
        {
            Dot1(ref uaRef, ref wRef, aScale, y, o0, s, inDim, outDim);
        }
        ForwardProfile.StageStop(ForwardProfile.Stage.Dot, dotTs);
    }

    /// <summary>Four output channels by four positions: sixteen independent accumulator chains, and
    /// each weight load feeds four of them.</summary>
    private void Dot4(ref byte uaRef, ref sbyte wRef, float[] aScale, float[] y, int o0, int s, int inDim, int outDim)
    {
        int groups = _groups, gs = _groupSize;
        int width = Vector256<byte>.Count;
        int sb0 = s * inDim, sb1 = sb0 + inDim, sb2 = sb1 + inDim, sb3 = sb2 + inDim;

        var f0 = Vector128<float>.Zero; var f1 = Vector128<float>.Zero;
        var f2 = Vector128<float>.Zero; var f3 = Vector128<float>.Zero;

        for (int g = 0; g < groups; g++)
        {
            int gStart = g * gs;
            var a00 = Vector256<int>.Zero; var a01 = Vector256<int>.Zero; var a02 = Vector256<int>.Zero; var a03 = Vector256<int>.Zero;
            var a10 = Vector256<int>.Zero; var a11 = Vector256<int>.Zero; var a12 = Vector256<int>.Zero; var a13 = Vector256<int>.Zero;
            var a20 = Vector256<int>.Zero; var a21 = Vector256<int>.Zero; var a22 = Vector256<int>.Zero; var a23 = Vector256<int>.Zero;
            var a30 = Vector256<int>.Zero; var a31 = Vector256<int>.Zero; var a32 = Vector256<int>.Zero; var a33 = Vector256<int>.Zero;

            for (int c = 0; c < gs; c += width)
            {
                int off = gStart + c;
                var w0 = Vector256.LoadUnsafe(ref wRef, (nuint)off);
                var w1 = Vector256.LoadUnsafe(ref wRef, (nuint)(inDim + off));
                var w2 = Vector256.LoadUnsafe(ref wRef, (nuint)(2 * inDim + off));
                var w3 = Vector256.LoadUnsafe(ref wRef, (nuint)(3 * inDim + off));

                var av0 = Vector256.LoadUnsafe(ref uaRef, (nuint)(sb0 + off));
                a00 = Vnni.DotAccumulate(a00, av0, w0); a01 = Vnni.DotAccumulate(a01, av0, w1);
                a02 = Vnni.DotAccumulate(a02, av0, w2); a03 = Vnni.DotAccumulate(a03, av0, w3);

                var av1 = Vector256.LoadUnsafe(ref uaRef, (nuint)(sb1 + off));
                a10 = Vnni.DotAccumulate(a10, av1, w0); a11 = Vnni.DotAccumulate(a11, av1, w1);
                a12 = Vnni.DotAccumulate(a12, av1, w2); a13 = Vnni.DotAccumulate(a13, av1, w3);

                var av2 = Vector256.LoadUnsafe(ref uaRef, (nuint)(sb2 + off));
                a20 = Vnni.DotAccumulate(a20, av2, w0); a21 = Vnni.DotAccumulate(a21, av2, w1);
                a22 = Vnni.DotAccumulate(a22, av2, w2); a23 = Vnni.DotAccumulate(a23, av2, w3);

                var av3 = Vector256.LoadUnsafe(ref uaRef, (nuint)(sb3 + off));
                a30 = Vnni.DotAccumulate(a30, av3, w0); a31 = Vnni.DotAccumulate(a31, av3, w1);
                a32 = Vnni.DotAccumulate(a32, av3, w2); a33 = Vnni.DotAccumulate(a33, av3, w3);
            }

            var sc = Vector128.LoadUnsafe(ref MemoryMarshal.GetArrayDataReference(_scales), (nuint)(g * outDim + o0));

            f0 = Scale(sc, Sum4(a00, a01, a02, a03), f0);
            f1 = Scale(sc, Sum4(a10, a11, a12, a13), f1);
            f2 = Scale(sc, Sum4(a20, a21, a22, a23), f2);
            f3 = Scale(sc, Sum4(a30, a31, a32, a33), f3);
        }

        var bias = Vector128.LoadUnsafe(ref MemoryMarshal.GetArrayDataReference(_zeroBias), (nuint)o0);
        ref float yRef = ref MemoryMarshal.GetArrayDataReference(y);
        ((f0 - bias) * aScale[s]).StoreUnsafe(ref yRef, (nuint)(s * outDim + o0));
        ((f1 - bias) * aScale[s + 1]).StoreUnsafe(ref yRef, (nuint)((s + 1) * outDim + o0));
        ((f2 - bias) * aScale[s + 2]).StoreUnsafe(ref yRef, (nuint)((s + 2) * outDim + o0));
        ((f3 - bias) * aScale[s + 3]).StoreUnsafe(ref yRef, (nuint)((s + 3) * outDim + o0));
    }

    /// <summary>The tail: four output channels for a single position.</summary>
    private void Dot1(ref byte uaRef, ref sbyte wRef, float[] aScale, float[] y, int o0, int s, int inDim, int outDim)
    {
        int groups = _groups, gs = _groupSize;
        int width = Vector256<byte>.Count;
        int sb0 = s * inDim;
        var f0 = Vector128<float>.Zero;

        for (int g = 0; g < groups; g++)
        {
            int gStart = g * gs;
            var a00 = Vector256<int>.Zero; var a01 = Vector256<int>.Zero;
            var a02 = Vector256<int>.Zero; var a03 = Vector256<int>.Zero;

            for (int c = 0; c < gs; c += width)
            {
                int off = gStart + c;
                var av0 = Vector256.LoadUnsafe(ref uaRef, (nuint)(sb0 + off));
                a00 = Vnni.DotAccumulate(a00, av0, Vector256.LoadUnsafe(ref wRef, (nuint)off));
                a01 = Vnni.DotAccumulate(a01, av0, Vector256.LoadUnsafe(ref wRef, (nuint)(inDim + off)));
                a02 = Vnni.DotAccumulate(a02, av0, Vector256.LoadUnsafe(ref wRef, (nuint)(2 * inDim + off)));
                a03 = Vnni.DotAccumulate(a03, av0, Vector256.LoadUnsafe(ref wRef, (nuint)(3 * inDim + off)));
            }

            var sc = Vector128.LoadUnsafe(ref MemoryMarshal.GetArrayDataReference(_scales), (nuint)(g * outDim + o0));
            f0 = Scale(sc, Sum4(a00, a01, a02, a03), f0);
        }

        var bias = Vector128.LoadUnsafe(ref MemoryMarshal.GetArrayDataReference(_zeroBias), (nuint)o0);
        ((f0 - bias) * aScale[s]).StoreUnsafe(ref MemoryMarshal.GetArrayDataReference(y), (nuint)(s * outDim + o0));
    }

    /// <summary>Unpacks one output row's packed codes into signed bytes.</summary>
    private void UnpackRow(int o, Span<sbyte> dst)
    {
        int rowBase = o * _rowBytes;
        if (_band == StqBand.Q4_0)
        {
            StqPacking.UnpackQ4Row(_codes.AsSpan(rowBase, _rowBytes), dst, _groups, _groupSize);
            return;
        }
        for (int g = 0; g < _groups; g++)
        {
            StqPacking.UnpackGroup(_band, _codes.AsSpan(rowBase + g * _groupBytes, _groupBytes),
                                   dst.Slice(g * _groupSize, _groupSize), _groupSize);
        }
    }

    /// <summary>
    /// Sums four int32 accumulators into one <c>Vector128&lt;int&gt;</c> of four totals.
    ///
    /// <para>Four separate <c>Vector256.Sum</c> calls cost four independent shuffle-and-add chains.
    /// Two <c>vphaddd</c>s fold the four vectors pairwise and a third produces all four totals in one
    /// register, so the per-group reduction is paid once for the tile instead of once per output
    /// channel. This matters here in a way it does not for the Int8 kernel: per-group scales force a
    /// reduction every 128 weights rather than one at the end of the row.</para>
    /// </summary>
    /// <summary>Converts a group's four dot products to float and accumulates them scaled, as one FMA
    /// where the host has it. The per-group scales make this the tail of every 128 weights rather than
    /// of every row, so its instruction count is worth caring about.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<float> Scale(Vector128<float> scale, Vector128<int> dot, Vector128<float> acc)
    {
        var f = Vector128.ConvertToSingle(dot);
        return Fma.IsSupported ? Fma.MultiplyAdd(scale, f, acc) : acc + scale * f;
    }

    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<int> Sum4(Vector256<int> a, Vector256<int> b, Vector256<int> c, Vector256<int> d)
    {
        if (Avx2.IsSupported)
        {
            var ab = Avx2.HorizontalAdd(a, b);      // [sum(a[0..1]), sum(a[2..3]), sum(b[0..1]), sum(b[2..3]) | upper halves]
            var cd = Avx2.HorizontalAdd(c, d);
            var abcd = Avx2.HorizontalAdd(ab, cd);  // lower 128 = per-vector sums of the low halves, upper = of the high halves
            return abcd.GetLower() + abcd.GetUpper();
        }
        return Vector128.Create(Vector256.Sum(a), Vector256.Sum(b), Vector256.Sum(c), Vector256.Sum(d));
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
    [SkipLocalsInit]
    private void VnniColumn(byte[] ua, float[] aScale, float[] y, int o, int seq, int inDim, int outDim)
    {
        Span<sbyte> wbuf = stackalloc sbyte[inDim];
        int rowBase = o * _rowBytes;
        for (int g = 0; g < _groups; g++)
        {
            StqPacking.UnpackGroup(_band, _codes.AsSpan(rowBase + g * _groupBytes, _groupBytes),
                                       wbuf.Slice(g * _groupSize, _groupSize), _groupSize);
        }

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
                facc += _scales[g * OutDim + o] * Vector256.Sum(acc);
            }
            y[s * outDim + o] = aScale[s] * (facc - _zeroBias[o]);
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

    [SkipLocalsInit]
    private void FloatColumn(float[] x, float[] y, int seq, int inDim, int outDim, int o)
    {
        Span<float> buf = stackalloc float[inDim];
        int rowBase = o * _rowBytes;
        for (int g = 0; g < _groups; g++)
        {
            StqPacking.UnpackGroupScaled(_band, _codes.AsSpan(rowBase + g * _groupBytes, _groupBytes),
                                             buf.Slice(g * _groupSize, _groupSize), _groupSize, _scales[g * OutDim + o]);
        }
        for (int s = 0; s < seq; s++)
        {
            y[s * outDim + o] = TensorPrimitives.Dot(new ReadOnlySpan<float>(x, s * inDim, inDim), buf);
        }
    }

    /// <summary>Bytes this matrix occupies in memory (codes + scales + the derived zero-point bias).</summary>
    public long ResidentBytes => _codes.LongLength + _scales.LongLength * 4 + _zeroBias.LongLength * 4;
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
