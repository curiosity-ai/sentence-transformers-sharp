using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;
using SentenceTransformers.Stq;

namespace SentenceTransformers.Harrier.Small.Pure.Model;

/// <summary>
/// The 512-bit form of the packed kernel. Identical in shape to the 256-bit one in
/// <c>StqMatrix.cs</c> - blocked weights so an accumulator's lanes are different output channels, a
/// broadcast activation quad, no per-scale-group horizontal reduction - with one accumulator now
/// covering sixteen channels instead of eight, so a tile is 64 channels wide.
///
/// <para><b>When it runs.</b> Only when a real 512-bit int8 dot instruction exists
/// (<see cref="Vnni.Has512Dot"/>), which in practice means one of two things: <c>AvxVnni.V512</c>
/// (<c>vpdpbusd</c> on zmm), which .NET 11 added in dotnet/runtime#128365 and which is what the
/// common <c>avx512_vnni</c> server parts have; or <c>AvxVnniInt8.V512</c> on AVX10.2-class parts,
/// which .NET 10 already exposed. It deliberately does <i>not</i> run on the widen-and-<c>vpmaddwd</c>
/// emulation that <c>Vnni.Use512</c> also covers - that is slower than 256-bit <c>vpdpbusd</c>.</para>
///
/// <para>Measured on an <c>avx512_vnni</c> host, 512-bit <c>vpdpbusd</c> is 2.7x the MAC throughput
/// of the 256-bit form, which is why this is worth a second copy of the kernel rather than a runtime
/// width parameter in the first one.</para>
/// </summary>
internal sealed partial class StqMatrix
{
    /// <summary>Output channels one 512-bit accumulator covers - one per int32 lane.</summary>
    private const int Blk512 = 16;

    /// <summary>Output channels computed together per 512-bit tile. Sixteen accumulators, four weight
    /// vectors and four broadcast activations is twenty-four zmm of the thirty-two available.</summary>
    private const int TileOut512 = Blk512 * BlocksPerTile;

    /// <inheritdoc cref="Tile"/>
    [SkipLocalsInit]
    private void Tile512(byte[] ua, float[] aScale, float[] y, int t, int seq, int inDim, int outDim)
    {
        int gs = _groupSize, groups = _groups;
        int tileCodes = TileOut512 * gs;
        Span<sbyte> w = tileCodes <= 16384 ? stackalloc sbyte[tileCodes] : new sbyte[tileCodes];

        var acc = TileAcc(seq * TileOut512);
        int o0 = t * TileOut512;
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

            Prefetch(codeBase + (long)(g + 1) * _tileGroupBytes, _tileGroupBytes);

            long dotTs = ForwardProfile.StageStart();
            nuint scaleBase = (nuint)((long)g * _scaleStride + o0);

            int s = 0;
            for (; s + TilePos <= seq; s += TilePos)
            {
                Dot4_512(ref uaRef, ref wRef, ref accRef, ref scaleRef, scaleBase, g, s, gs, inDim);
            }
            for (; s < seq; s++)
            {
                Dot1_512(ref uaRef, ref wRef, ref accRef, ref scaleRef, scaleBase, g, s, gs, inDim);
            }
            ForwardProfile.StageStop(ForwardProfile.Stage.Dot, dotTs);
        }

        Store512(acc, aScale, y, o0, seq, outDim);
    }

    /// <inheritdoc cref="Dot4"/>
    private void Dot4_512(ref byte uaRef, ref sbyte wRef, ref float accRef, ref float scaleRef,
                          nuint scaleBase, int g, int s, int gs, int inDim)
    {
        int quads = gs >> 2;
        int k0 = g * gs;
        int u0 = s * inDim + k0, u1 = u0 + inDim, u2 = u1 + inDim, u3 = u2 + inDim;

        var c00 = Vector512<int>.Zero; var c01 = Vector512<int>.Zero; var c02 = Vector512<int>.Zero; var c03 = Vector512<int>.Zero;
        var c10 = Vector512<int>.Zero; var c11 = Vector512<int>.Zero; var c12 = Vector512<int>.Zero; var c13 = Vector512<int>.Zero;
        var c20 = Vector512<int>.Zero; var c21 = Vector512<int>.Zero; var c22 = Vector512<int>.Zero; var c23 = Vector512<int>.Zero;
        var c30 = Vector512<int>.Zero; var c31 = Vector512<int>.Zero; var c32 = Vector512<int>.Zero; var c33 = Vector512<int>.Zero;

        for (int q = 0; q < quads; q++)
        {
            nuint wo = (nuint)(q * (TileOut512 * 4));
            var w0 = Vector512.LoadUnsafe(ref wRef, wo);
            var w1 = Vector512.LoadUnsafe(ref wRef, wo + 64);
            var w2 = Vector512.LoadUnsafe(ref wRef, wo + 128);
            var w3 = Vector512.LoadUnsafe(ref wRef, wo + 192);

            int k = q * 4;
            var a0 = Bcast512(ref uaRef, (nuint)(u0 + k));
            var a1 = Bcast512(ref uaRef, (nuint)(u1 + k));
            var a2 = Bcast512(ref uaRef, (nuint)(u2 + k));
            var a3 = Bcast512(ref uaRef, (nuint)(u3 + k));

            c00 = Vnni.DotAccumulate512(c00, a0, w0); c01 = Vnni.DotAccumulate512(c01, a0, w1);
            c02 = Vnni.DotAccumulate512(c02, a0, w2); c03 = Vnni.DotAccumulate512(c03, a0, w3);

            c10 = Vnni.DotAccumulate512(c10, a1, w0); c11 = Vnni.DotAccumulate512(c11, a1, w1);
            c12 = Vnni.DotAccumulate512(c12, a1, w2); c13 = Vnni.DotAccumulate512(c13, a1, w3);

            c20 = Vnni.DotAccumulate512(c20, a2, w0); c21 = Vnni.DotAccumulate512(c21, a2, w1);
            c22 = Vnni.DotAccumulate512(c22, a2, w2); c23 = Vnni.DotAccumulate512(c23, a2, w3);

            c30 = Vnni.DotAccumulate512(c30, a3, w0); c31 = Vnni.DotAccumulate512(c31, a3, w1);
            c32 = Vnni.DotAccumulate512(c32, a3, w2); c33 = Vnni.DotAccumulate512(c33, a3, w3);
        }

        var s0 = Vector512.LoadUnsafe(ref scaleRef, scaleBase);
        var s1 = Vector512.LoadUnsafe(ref scaleRef, scaleBase + Blk512);
        var s2 = Vector512.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(2 * Blk512));
        var s3 = Vector512.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(3 * Blk512));

        nuint p0 = (nuint)(s * TileOut512), p1 = p0 + TileOut512, p2 = p1 + TileOut512, p3 = p2 + TileOut512;
        Accumulate512(ref accRef, p0, s0, c00); Accumulate512(ref accRef, p0 + Blk512, s1, c01);
        Accumulate512(ref accRef, p0 + (nuint)(2 * Blk512), s2, c02); Accumulate512(ref accRef, p0 + (nuint)(3 * Blk512), s3, c03);

        Accumulate512(ref accRef, p1, s0, c10); Accumulate512(ref accRef, p1 + Blk512, s1, c11);
        Accumulate512(ref accRef, p1 + (nuint)(2 * Blk512), s2, c12); Accumulate512(ref accRef, p1 + (nuint)(3 * Blk512), s3, c13);

        Accumulate512(ref accRef, p2, s0, c20); Accumulate512(ref accRef, p2 + Blk512, s1, c21);
        Accumulate512(ref accRef, p2 + (nuint)(2 * Blk512), s2, c22); Accumulate512(ref accRef, p2 + (nuint)(3 * Blk512), s3, c23);

        Accumulate512(ref accRef, p3, s0, c30); Accumulate512(ref accRef, p3 + Blk512, s1, c31);
        Accumulate512(ref accRef, p3 + (nuint)(2 * Blk512), s2, c32); Accumulate512(ref accRef, p3 + (nuint)(3 * Blk512), s3, c33);
    }

    /// <inheritdoc cref="Dot1"/>
    private void Dot1_512(ref byte uaRef, ref sbyte wRef, ref float accRef, ref float scaleRef,
                          nuint scaleBase, int g, int s, int gs, int inDim)
    {
        int quads = gs >> 2;
        int u0 = s * inDim + g * gs;

        var c0 = Vector512<int>.Zero; var c1 = Vector512<int>.Zero;
        var c2 = Vector512<int>.Zero; var c3 = Vector512<int>.Zero;

        for (int q = 0; q < quads; q++)
        {
            nuint wo = (nuint)(q * (TileOut512 * 4));
            var a0 = Bcast512(ref uaRef, (nuint)(u0 + q * 4));
            c0 = Vnni.DotAccumulate512(c0, a0, Vector512.LoadUnsafe(ref wRef, wo));
            c1 = Vnni.DotAccumulate512(c1, a0, Vector512.LoadUnsafe(ref wRef, wo + 64));
            c2 = Vnni.DotAccumulate512(c2, a0, Vector512.LoadUnsafe(ref wRef, wo + 128));
            c3 = Vnni.DotAccumulate512(c3, a0, Vector512.LoadUnsafe(ref wRef, wo + 192));
        }

        nuint o = (nuint)(s * TileOut512);
        Accumulate512(ref accRef, o, Vector512.LoadUnsafe(ref scaleRef, scaleBase), c0);
        Accumulate512(ref accRef, o + Blk512, Vector512.LoadUnsafe(ref scaleRef, scaleBase + Blk512), c1);
        Accumulate512(ref accRef, o + (nuint)(2 * Blk512), Vector512.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(2 * Blk512)), c2);
        Accumulate512(ref accRef, o + (nuint)(3 * Blk512), Vector512.LoadUnsafe(ref scaleRef, scaleBase + (nuint)(3 * Blk512)), c3);
    }

    /// <inheritdoc cref="Bcast"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector512<byte> Bcast512(ref byte p, nuint offset)
        => Vector512.Create(Unsafe.ReadUnaligned<uint>(ref Unsafe.Add(ref p, offset))).AsByte();

    /// <inheritdoc cref="Accumulate"/>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Accumulate512(ref float accRef, nuint offset, Vector512<float> scale, Vector512<int> dot)
    {
        var f = Vector512.LoadUnsafe(ref accRef, offset);
        var d = Vector512.ConvertToSingle(dot);
        (Avx512F.IsSupported ? Avx512F.FusedMultiplyAdd(scale, d, f) : f + scale * d).StoreUnsafe(ref accRef, offset);
    }

    /// <inheritdoc cref="Store"/>
    private void Store512(float[] acc, float[] aScale, float[] y, int o0, int seq, int outDim)
    {
        ref float accRef = ref MemoryMarshal.GetArrayDataReference(acc);
        ref float yRef = ref MemoryMarshal.GetArrayDataReference(y);
        ref float biasRef = ref MemoryMarshal.GetArrayDataReference(_zeroBias);

        for (int s = 0; s < seq; s++)
        {
            var scale = Vector512.Create(aScale[s]);
            for (int b = 0; b < BlocksPerTile; b++)
            {
                int o = o0 + b * Blk512;
                if (o >= outDim)
                {
                    break;
                }

                var f = Vector512.LoadUnsafe(ref accRef, (nuint)(s * TileOut512 + b * Blk512));
                var r = (f - Vector512.LoadUnsafe(ref biasRef, (nuint)o)) * scale;

                if (o + Blk512 <= outDim)
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
}
