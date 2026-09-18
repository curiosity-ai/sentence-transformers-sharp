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

    /// <summary>
    /// The packed codes, in one of two layouts picked at load time:
    /// <list type="bullet">
    /// <item>on a VNNI host, the tile-blocked order <see cref="BuildBlocked"/> builds - indexed
    /// <c>[(tile * groups + g) * _tileGroupBytes]</c>;</item>
    /// <item>otherwise the file's own row-major order, <c>[outDim * _rowBytes]</c>, which is what the
    /// float fallback wants since it decodes a row at a time.</item>
    /// </list>
    /// </summary>
    private readonly byte[] _codes;

    // Scales are stored group-major - [g * _scaleStride + o], not [o * groups + g] - so a block's
    // eight output channels are contiguous for one group and load as a single Vector256. In row-major
    // order they are strided by `groups`, which would need a gather per group.
    private readonly float[] _scales;        // [groups * _scaleStride]

    // The activation zero-point correction, folded per output channel: sum over groups of
    // scale[g,o] * ZeroPoint * (sum of group g's codes). Every term of it is position-independent, so
    // subtracting it once at the end of a row is exactly equivalent to subtracting each group's share
    // inside the group loop - and takes a load, a multiply and a subtract per group out of the hottest
    // loop in the kernel, where per-group scales already force a reduction every 128 weights.
    private readonly float[] _zeroBias;      // [_scaleStride]

    private readonly HadamardRotation _rotation; // null when the tensor is stored unrotated
    private readonly int _groupSize;
    private readonly int _groups;
    private readonly int _groupBytes;
    private readonly int _rowBytes;

    // Output channels rounded up to a whole number of tiles. The padding channels hold zero codes and
    // zero scales, so they contribute nothing and only their stores are skipped - which keeps the
    // kernel free of a ragged-tail variant.
    private readonly int _scaleStride;
    private readonly int _tiles;             // _scaleStride / TileOut
    private readonly int _tileGroupBytes;    // bytes one tile's worth of one scale group packs into

    public int InDim { get; }
    public int OutDim { get; }

    private StqMatrix(StqBand band, byte[] codes, float[] scales, float[] zeroBias, HadamardRotation rotation,
                          int groupSize, int groups, int groupBytes, int outDim, int inDim,
                          int tiles, int tileGroupBytes)
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
        _tiles = tiles;
        _scaleStride = tiles * TileOut;
        _tileGroupBytes = tileGroupBytes;
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
        if (groupSize % 4 != 0)
        {
            throw new ArgumentException($"Tensor '{info.Name}': group size {groupSize} must be a multiple of 4 - the int8 dot consumes four inputs per lane.", nameof(info));
        }

        int groups = info.GroupsPerRow;
        int groupBytes = StqFormat.CodeBytesPerGroup(info.Band, groupSize);
        int rowBytes = groups * groupBytes;
        int tiles = (outDim + TileOut - 1) / TileOut;
        int scaleStride = tiles * TileOut;
        int tileGroupBytes = StqFormat.CodeBytesPerGroup(info.Band, TileOut * groupSize);

        var rowCodes = file.Codes(info).ToArray();
        var rowMajorScales = file.Scales(info);
        var scales = new float[(long)groups * scaleStride];
        var zeroBias = new float[scaleStride];
        var band = info.Band;

        await ParallelExecution.ForAsync(0, outDim, parallelOptions, (o, _) =>
        {
            Span<sbyte> group = stackalloc sbyte[groupSize];
            float bias = 0f;
            for (int g = 0; g < groups; g++)
            {
                StqPacking.UnpackGroup(band, rowCodes.AsSpan(o * rowBytes + g * groupBytes, groupBytes), group, groupSize);
                int sum = 0;
                for (int i = 0; i < groupSize; i++)
                {
                    sum += group[i];
                }
                float scale = rowMajorScales[(long)o * groups + g];
                scales[(long)g * scaleStride + o] = scale;
                bias += scale * (Vnni.ZeroPoint * sum);
            }
            zeroBias[o] = bias;
            return ValueTask.CompletedTask;
        }).ConfigureAwait(false);

        // The float fallback decodes a row at a time and wants the file's own order; the VNNI kernel
        // wants the blocked one and never looks at the row-major array again, so only one is kept.
        var codes = Vnni.IsSupported
            ? await BuildBlocked(band, rowCodes, outDim, groupSize, groups, groupBytes, rowBytes,
                                 tiles, tileGroupBytes, parallelOptions).ConfigureAwait(false)
            : rowCodes;

        return new StqMatrix(band, codes, scales, zeroBias, file.RotationFor(info), groupSize, groups,
                             groupBytes, outDim, inDim, tiles, tileGroupBytes);
    }

    /// <summary>
    /// Rewrites the file's row-major codes into the order the VNNI kernel wants, once at load.
    ///
    /// <para><c>vpdpbusd</c> sums four byte products into each of its eight int32 lanes. Feed it a row
    /// of weights and those lanes hold four partial sums of <i>one</i> output channel, so every scale
    /// group ends in a horizontal reduction - three <c>vphaddd</c>s per four channels, every 128
    /// weights, which was about 30% of the kernel. Feed it instead a vector whose lane <c>L</c> holds
    /// four consecutive weights of output channel <c>L</c>, against an activation vector that is those
    /// same four activations broadcast into all eight lanes, and each lane accumulates a different
    /// output channel. The group then ends in a convert and a multiply-add and no reduction at all.
    /// The instruction doing the multiplying is the same one either way; only the operand layout
    /// changes, which is why this is worth a pass over the weights at load.</para>
    ///
    /// <para>The order, for one tile of <see cref="TileOut"/> channels and one scale group, is
    /// <c>q * (TileOut * 4) + b * 32 + L * 4 + m</c>: input quad <c>q</c>, then block <c>b</c> of
    /// eight channels, then channel <c>L</c> within it, then input <c>m</c> within the quad. Quad-major
    /// so that the four weight vectors one step needs are 128 contiguous bytes, and whole-tile groups
    /// so a tile's codes for a group unpack in a single call.</para>
    /// </summary>
    private static async Task<byte[]> BuildBlocked(StqBand band, byte[] rowCodes, int outDim, int groupSize,
                                                   int groups, int groupBytes, int rowBytes,
                                                   int tiles, int tileGroupBytes, ParallelOptions parallelOptions)
    {
        // Pinned: the kernel takes a raw pointer into this to prefetch the next group, and it is a
        // long-lived, large, never-resized array - exactly what the pinned object heap is for.
        var blocked = GC.AllocateArray<byte>(checked((int)((long)tiles * groups * tileGroupBytes)), pinned: true);
        int tileCodeCount = TileOut * groupSize;
        int quads = groupSize / 4;

        await ParallelExecution.ForAsync(0, tiles, parallelOptions, (t, _) =>
        {
            Span<sbyte> row = stackalloc sbyte[groupSize];
            Span<sbyte> tile = tileCodeCount <= 8192 ? stackalloc sbyte[tileCodeCount] : new sbyte[tileCodeCount];

            for (int g = 0; g < groups; g++)
            {
                // Channels past outDim stay zero: a zero code times anything is zero, and their scales
                // are zero too, so the padding contributes nothing to a real output.
                tile.Clear();

                for (int c = 0; c < TileOut; c++)
                {
                    int o = t * TileOut + c;
                    if (o >= outDim)
                    {
                        break;
                    }

                    StqPacking.UnpackGroup(band, rowCodes.AsSpan(o * rowBytes + g * groupBytes, groupBytes), row, groupSize);

                    int lane = (c >> 3) * 32 + (c & 7) * 4;
                    for (int q = 0; q < quads; q++)
                    {
                        int dst = q * (TileOut * 4) + lane;
                        int src = q * 4;
                        tile[dst]     = row[src];
                        tile[dst + 1] = row[src + 1];
                        tile[dst + 2] = row[src + 2];
                        tile[dst + 3] = row[src + 3];
                    }
                }

                StqPacking.PackGroup(band, tile, blocked.AsSpan((int)(((long)t * groups + g) * tileGroupBytes), tileGroupBytes));
            }
            return ValueTask.CompletedTask;
        }).ConfigureAwait(false);

        return blocked;
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
        // The out-of-place Apply writes the destination on its sign pass, so this costs no copy.
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
                rotation.Apply(src.AsSpan(sIdx * inDim, inDim), dst.AsSpan(sIdx * inDim, inDim));
                ForwardProfile.StageStop(ForwardProfile.Stage.Rotate, ts);
                return ValueTask.CompletedTask;
            }).ConfigureAwait(false);
        }
        else
        {
            long ts = ForwardProfile.StageStart();
            for (int sIdx = 0; sIdx < seq; sIdx++)
            {
                rotation.Apply(src.AsSpan(sIdx * inDim, inDim), dst.AsSpan(sIdx * inDim, inDim));
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
        int inDim = InDim, outDim = OutDim, tiles = _tiles;
        if (RunInline(parallelOptions))
        {
            for (int t = 0; t < tiles; t++)
            {
                Tile(ua, aScale, y, t, seq, inDim, outDim);
            }
            return ValueTask.CompletedTask;
        }

        return new ValueTask(ParallelExecution.ForAsync(0, tiles, parallelOptions, (t, _) =>
        {
            Tile(ua, aScale, y, t, seq, inDim, outDim);
            return ValueTask.CompletedTask;
        }));
    }

    /// <summary>Output channels a vpdpbusd accumulator covers - one per int32 lane.</summary>
    private const int Blk = 8;

    /// <summary>Accumulator vectors per position, so a tile is <c>Blk * BlocksPerTile</c> channels.
    /// Four is what the register file allows: sixteen int32 accumulators, four weight vectors and four
    /// broadcast activations is twenty-four vectors, and an AVX-512 host gives the JIT thirty-two.</summary>
    private const int BlocksPerTile = 4;

    /// <summary>Output channels computed together per tile.</summary>
    private const int TileOut = Blk * BlocksPerTile;

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

    /// <summary>Per-thread float accumulators for the tile being computed, <c>seq * TileOut</c> long.
    /// They cannot stay in registers - sixteen int32 accumulators already fill most of the file - so
    /// they live in a small array that is L1-resident for any realistic sequence length.</summary>
    [ThreadStatic] private static float[] _tileAcc;

    private static float[] TileAcc(int needed)
    {
        var acc = _tileAcc;
        if (acc is null || acc.Length < needed)
        {
            acc = new float[needed];
            _tileAcc = acc;
        }
        Array.Clear(acc, 0, needed);
        return acc;
    }

    /// <summary>
    /// The hot kernel: <see cref="TileOut"/> output channels by <see cref="TilePos"/> positions,
    /// sixteen independent int32 accumulator chains.
    ///
    /// <para>Two things are going on. The chains are what a naive one-channel-at-a-time loop gets
    /// wrong: <c>acc = DotAccumulate(acc, ...)</c> is a serial dependency, so a single chain stalls on
    /// its own ~5-cycle latency no matter how cheap the unpacking is. And the operand layout (see
    /// <see cref="BuildBlocked"/>) is what makes the accumulator lanes <i>be</i> the output channels,
    /// so a scale group ends in a convert and a multiply-add rather than a horizontal reduction.</para>
    ///
    /// <para>The group loop is outermost so a group's weights are unpacked once and then used against
    /// every position while they are still in L1. That is why the float accumulators have to spill to
    /// memory: the alternative, positions outermost, would hold them in registers but would need the
    /// tile's whole row unpacked at once - 64 KB for a 2048-wide projection.</para>
    /// </summary>
    [SkipLocalsInit]
    private void Tile(byte[] ua, float[] aScale, float[] y, int t, int seq, int inDim, int outDim)
    {
        int gs = _groupSize, groups = _groups;
        int tileCodes = TileOut * gs;
        Span<sbyte> w = tileCodes <= 8192 ? stackalloc sbyte[tileCodes] : new sbyte[tileCodes];

        var acc = TileAcc(seq * TileOut);
        int o0 = t * TileOut;
        long codeBase = (long)t * groups * _tileGroupBytes;

        ref byte uaRef = ref MemoryMarshal.GetArrayDataReference(ua);
        ref sbyte wRef = ref MemoryMarshal.GetReference(w);
        ref float accRef = ref MemoryMarshal.GetArrayDataReference(acc);
        ref float scaleRef = ref MemoryMarshal.GetArrayDataReference(_scales);

        for (int g = 0; g < groups; g++)
        {
            long unpackTs = ForwardProfile.StageStart();
            StqPacking.UnpackGroup(_band, _codes.AsSpan((int)(codeBase + (long)g * _tileGroupBytes), _tileGroupBytes), w, tileCodes);
            ForwardProfile.StageStop(ForwardProfile.Stage.UnpackWeights, unpackTs);

            // The dot below is long and touches no new memory - the weights it reads are the 4 KB
            // just unpacked. That is the window in which to fetch the next group's codes. Measured in
            // isolation the unpack runs at 20.5 GB/s streaming from L3 against 35.1 GB/s when its
            // source is already hot, and this recovers part of the difference (584 -> 428 ms/iter).
            // One group ahead, not two: two prefetches every line twice and measured slower overall.
            Prefetch(codeBase + (long)(g + 1) * _tileGroupBytes, _tileGroupBytes);

            long dotTs = ForwardProfile.StageStart();
            nuint scaleBase = (nuint)((long)g * _scaleStride + o0);

            int s = 0;
            for (; s + TilePos <= seq; s += TilePos)
            {
                Dot4(ref uaRef, ref wRef, ref accRef, ref scaleRef, scaleBase, g, s, gs, inDim);
            }
            for (; s + 2 <= seq; s += 2)
            {
                Dot2(ref uaRef, ref wRef, ref accRef, ref scaleRef, scaleBase, g, s, gs, inDim);
            }
            for (; s < seq; s++)
            {
                Dot1(ref uaRef, ref wRef, ref accRef, ref scaleRef, scaleBase, g, s, gs, inDim);
            }
            ForwardProfile.StageStop(ForwardProfile.Stage.Dot, dotTs);
        }

        Store(acc, aScale, y, o0, seq, outDim);
    }

    /// <summary>Four positions at a time: each of the four weight vectors a step loads feeds four
    /// accumulators, and each broadcast activation feeds four.</summary>
    private void Dot4(ref byte uaRef, ref sbyte wRef, ref float accRef, ref float scaleRef,
                      nuint scaleBase, int g, int s, int gs, int inDim)
    {
        int quads = gs >> 2;
        int k0 = g * gs;
        int u0 = s * inDim + k0, u1 = u0 + inDim, u2 = u1 + inDim, u3 = u2 + inDim;

        var c00 = Vector256<int>.Zero; var c01 = Vector256<int>.Zero; var c02 = Vector256<int>.Zero; var c03 = Vector256<int>.Zero;
        var c10 = Vector256<int>.Zero; var c11 = Vector256<int>.Zero; var c12 = Vector256<int>.Zero; var c13 = Vector256<int>.Zero;
        var c20 = Vector256<int>.Zero; var c21 = Vector256<int>.Zero; var c22 = Vector256<int>.Zero; var c23 = Vector256<int>.Zero;
        var c30 = Vector256<int>.Zero; var c31 = Vector256<int>.Zero; var c32 = Vector256<int>.Zero; var c33 = Vector256<int>.Zero;

        for (int q = 0; q < quads; q++)
        {
            nuint wo = (nuint)(q * (TileOut * 4));
            var w0 = Vector256.LoadUnsafe(ref wRef, wo);
            var w1 = Vector256.LoadUnsafe(ref wRef, wo + 32);
            var w2 = Vector256.LoadUnsafe(ref wRef, wo + 64);
            var w3 = Vector256.LoadUnsafe(ref wRef, wo + 96);

            int k = q * 4;
            var a0 = Bcast(ref uaRef, (nuint)(u0 + k));
            var a1 = Bcast(ref uaRef, (nuint)(u1 + k));
            var a2 = Bcast(ref uaRef, (nuint)(u2 + k));
            var a3 = Bcast(ref uaRef, (nuint)(u3 + k));

            c00 = Vnni.DotAccumulate(c00, a0, w0); c01 = Vnni.DotAccumulate(c01, a0, w1);
            c02 = Vnni.DotAccumulate(c02, a0, w2); c03 = Vnni.DotAccumulate(c03, a0, w3);

            c10 = Vnni.DotAccumulate(c10, a1, w0); c11 = Vnni.DotAccumulate(c11, a1, w1);
            c12 = Vnni.DotAccumulate(c12, a1, w2); c13 = Vnni.DotAccumulate(c13, a1, w3);

            c20 = Vnni.DotAccumulate(c20, a2, w0); c21 = Vnni.DotAccumulate(c21, a2, w1);
            c22 = Vnni.DotAccumulate(c22, a2, w2); c23 = Vnni.DotAccumulate(c23, a2, w3);

            c30 = Vnni.DotAccumulate(c30, a3, w0); c31 = Vnni.DotAccumulate(c31, a3, w1);
            c32 = Vnni.DotAccumulate(c32, a3, w2); c33 = Vnni.DotAccumulate(c33, a3, w3);
        }

        var s0 = Vector256.LoadUnsafe(ref scaleRef, scaleBase);
        var s1 = Vector256.LoadUnsafe(ref scaleRef, scaleBase + Blk);
        var s2 = Vector256.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(2 * Blk));
        var s3 = Vector256.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(3 * Blk));

        nuint p0 = (nuint)(s * TileOut), p1 = p0 + TileOut, p2 = p1 + TileOut, p3 = p2 + TileOut;
        Accumulate(ref accRef, p0, s0, c00); Accumulate(ref accRef, p0 + Blk, s1, c01);
        Accumulate(ref accRef, p0 + (nuint)(2 * Blk), s2, c02); Accumulate(ref accRef, p0 + (nuint)(3 * Blk), s3, c03);

        Accumulate(ref accRef, p1, s0, c10); Accumulate(ref accRef, p1 + Blk, s1, c11);
        Accumulate(ref accRef, p1 + (nuint)(2 * Blk), s2, c12); Accumulate(ref accRef, p1 + (nuint)(3 * Blk), s3, c13);

        Accumulate(ref accRef, p2, s0, c20); Accumulate(ref accRef, p2 + Blk, s1, c21);
        Accumulate(ref accRef, p2 + (nuint)(2 * Blk), s2, c22); Accumulate(ref accRef, p2 + (nuint)(3 * Blk), s3, c23);

        Accumulate(ref accRef, p3, s0, c30); Accumulate(ref accRef, p3 + Blk, s1, c31);
        Accumulate(ref accRef, p3 + (nuint)(2 * Blk), s2, c32); Accumulate(ref accRef, p3 + (nuint)(3 * Blk), s3, c33);
    }

    /// <summary>Two positions, for the part of a sequence that does not fill a four-position tile.
    /// Without it those positions fall to <see cref="Dot1"/>, which reloads the same four weight
    /// vectors for a quarter of the work.</summary>
    private void Dot2(ref byte uaRef, ref sbyte wRef, ref float accRef, ref float scaleRef,
                      nuint scaleBase, int g, int s, int gs, int inDim)
    {
        int quads = gs >> 2;
        int k0 = g * gs;
        int u0 = s * inDim + k0, u1 = u0 + inDim;

        var c00 = Vector256<int>.Zero; var c01 = Vector256<int>.Zero; var c02 = Vector256<int>.Zero; var c03 = Vector256<int>.Zero;
        var c10 = Vector256<int>.Zero; var c11 = Vector256<int>.Zero; var c12 = Vector256<int>.Zero; var c13 = Vector256<int>.Zero;

        for (int q = 0; q < quads; q++)
        {
            nuint wo = (nuint)(q * (TileOut * 4));
            var w0 = Vector256.LoadUnsafe(ref wRef, wo);
            var w1 = Vector256.LoadUnsafe(ref wRef, wo + 32);
            var w2 = Vector256.LoadUnsafe(ref wRef, wo + 64);
            var w3 = Vector256.LoadUnsafe(ref wRef, wo + 96);

            int k = q * 4;
            var a0 = Bcast(ref uaRef, (nuint)(u0 + k));
            var a1 = Bcast(ref uaRef, (nuint)(u1 + k));

            c00 = Vnni.DotAccumulate(c00, a0, w0); c01 = Vnni.DotAccumulate(c01, a0, w1);
            c02 = Vnni.DotAccumulate(c02, a0, w2); c03 = Vnni.DotAccumulate(c03, a0, w3);

            c10 = Vnni.DotAccumulate(c10, a1, w0); c11 = Vnni.DotAccumulate(c11, a1, w1);
            c12 = Vnni.DotAccumulate(c12, a1, w2); c13 = Vnni.DotAccumulate(c13, a1, w3);
        }

        var s0 = Vector256.LoadUnsafe(ref scaleRef, scaleBase);
        var s1 = Vector256.LoadUnsafe(ref scaleRef, scaleBase + Blk);
        var s2 = Vector256.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(2 * Blk));
        var s3 = Vector256.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(3 * Blk));

        nuint p0 = (nuint)(s * TileOut), p1 = p0 + TileOut;
        Accumulate(ref accRef, p0, s0, c00); Accumulate(ref accRef, p0 + Blk, s1, c01);
        Accumulate(ref accRef, p0 + (nuint)(2 * Blk), s2, c02); Accumulate(ref accRef, p0 + (nuint)(3 * Blk), s3, c03);

        Accumulate(ref accRef, p1, s0, c10); Accumulate(ref accRef, p1 + Blk, s1, c11);
        Accumulate(ref accRef, p1 + (nuint)(2 * Blk), s2, c12); Accumulate(ref accRef, p1 + (nuint)(3 * Blk), s3, c13);
    }

    /// <summary>The tail: one position, four accumulators.</summary>
    private void Dot1(ref byte uaRef, ref sbyte wRef, ref float accRef, ref float scaleRef,
                      nuint scaleBase, int g, int s, int gs, int inDim)
    {
        int quads = gs >> 2;
        int u0 = s * inDim + g * gs;

        var c0 = Vector256<int>.Zero; var c1 = Vector256<int>.Zero;
        var c2 = Vector256<int>.Zero; var c3 = Vector256<int>.Zero;

        for (int q = 0; q < quads; q++)
        {
            nuint wo = (nuint)(q * (TileOut * 4));
            var a0 = Bcast(ref uaRef, (nuint)(u0 + q * 4));
            c0 = Vnni.DotAccumulate(c0, a0, Vector256.LoadUnsafe(ref wRef, wo));
            c1 = Vnni.DotAccumulate(c1, a0, Vector256.LoadUnsafe(ref wRef, wo + 32));
            c2 = Vnni.DotAccumulate(c2, a0, Vector256.LoadUnsafe(ref wRef, wo + 64));
            c3 = Vnni.DotAccumulate(c3, a0, Vector256.LoadUnsafe(ref wRef, wo + 96));
        }

        nuint o = (nuint)(s * TileOut);
        Accumulate(ref accRef, o,                    Vector256.LoadUnsafe(ref scaleRef, scaleBase), c0);
        Accumulate(ref accRef, o + Blk,              Vector256.LoadUnsafe(ref scaleRef, scaleBase + Blk), c1);
        Accumulate(ref accRef, o + (nuint)(2 * Blk), Vector256.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(2 * Blk)), c2);
        Accumulate(ref accRef, o + (nuint)(3 * Blk), Vector256.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(3 * Blk)), c3);
    }

    /// <summary>A group's contribution for eight channels of one position: convert and multiply-add.
    /// This is the whole per-group tail - what a row-major layout would need a horizontal reduction
    /// for, done here by the lanes already being the right channels.
    ///
    /// <para>Folding the final bias-and-scale pass in here, so the last group writes <c>y</c>
    /// directly, was tried and reverted: carrying the extra state into the dot cost it more than the
    /// pass it removed (2125 against 1996 ms/iter, measured in one profile).</para></summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Accumulate(ref float accRef, nuint offset, Vector256<float> scale, Vector256<int> dot)
    {
        var f = Vector256.LoadUnsafe(ref accRef, offset);
        var d = Vector256.ConvertToSingle(dot);
        (Fma.IsSupported ? Fma.MultiplyAdd(scale, d, f) : f + scale * d).StoreUnsafe(ref accRef, offset);
    }

    /// <summary>Asks for a run of the code array to be pulled into L1. A prefetch of an address past
    /// the end of the array would be harmless (prefetches never fault) but the range is clamped anyway
    /// so the loop count stays honest, and <c>_codes</c> is pinned so the pointer cannot go stale.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private unsafe void Prefetch(long offset, int bytes)
    {
        if (!Sse.IsSupported)
        {
            return;
        }

        long end = Math.Min(offset + bytes, _codes.LongLength);
        if (offset >= end)
        {
            return;
        }

        ref byte start = ref MemoryMarshal.GetArrayDataReference(_codes);
        for (long o = offset; o < end; o += 64)
        {
            Sse.Prefetch0(Unsafe.AsPointer(ref Unsafe.Add(ref start, (nint)o)));
        }
    }

    /// <summary>Broadcasts the four activations at <paramref name="offset"/> into all eight lanes,
    /// which is the operand shape the blocked weight layout needs.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<byte> Bcast(ref byte p, nuint offset)
        => Vector256.Create(Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref p, offset))).AsByte();

    /// <summary>Applies the per-channel zero-point bias and the per-position activation scale, and
    /// writes the tile out. Channels past <paramref name="outDim"/> are the tile padding and are
    /// dropped here - the one place the kernel has to know about it.</summary>
    private void Store(float[] acc, float[] aScale, float[] y, int o0, int seq, int outDim)
    {
        ref float accRef = ref MemoryMarshal.GetArrayDataReference(acc);
        ref float yRef = ref MemoryMarshal.GetArrayDataReference(y);
        ref float biasRef = ref MemoryMarshal.GetArrayDataReference(_zeroBias);

        for (int s = 0; s < seq; s++)
        {
            var scale = Vector256.Create(aScale[s]);
            for (int b = 0; b < BlocksPerTile; b++)
            {
                int o = o0 + b * Blk;
                if (o >= outDim)
                {
                    break;
                }

                var f = Vector256.LoadUnsafe(ref accRef, (nuint)(s * TileOut + b * Blk));
                var r = (f - Vector256.LoadUnsafe(ref biasRef, (nuint)o)) * scale;

                if (o + Blk <= outDim)
                {
                    r.StoreUnsafe(ref yRef, (nuint)(s * outDim + o));
                }
                else
                {
                    for (int l = 0; o + l < outDim; l++)
                    {
                        Unsafe.Add(ref yRef, s * outDim + o + l) = r[l];
                    }
                }
            }
        }
    }

    /// <summary>Unpacks one output row's packed codes into signed bytes. Row-major layout only, so
    /// this is the float fallback's path.</summary>
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
                                             buf.Slice(g * _groupSize, _groupSize), _groupSize, _scales[g * _scaleStride + o]);
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
