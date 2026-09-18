#nullable enable

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SentenceTransformers.Stq;

/// <summary>
/// Packs and unpacks ternary codes for the two ternary bands. Both directions go through the same
/// per-byte lookup tables, so unpacking never divides or branches.
///
/// <list type="bullet">
/// <item><see cref="StqBand.TQ2_0"/>: four codes per byte, two bits each, stored as
/// <c>trit + 1</c> in <c>{0, 1, 2}</c> (the fourth code point is unused), least-significant pair
/// first. 32 bytes per 128-weight group.</item>
/// <item><see cref="StqBand.Q4_0"/>: two codes per byte, four bits each, stored as
/// <c>code + 8</c> in <c>[0, 15]</c>. <c>groupSize/2</c> bytes per group. The two nibbles of a byte
/// are <i>half a group apart</i>, not adjacent: byte <c>j</c> carries code <c>j</c> in its low
/// nibble and code <c>j + half</c> in its high nibble (the layout llama.cpp's <c>q4_0</c> uses).
/// Adjacent-nibble packing forces the unpacker to interleave two shuffled halves back together;
/// this layout makes a vector of bytes fall out as two contiguous vectors of codes, so unpacking a
/// group is a mask and a shift with no shuffle at all.</item>
/// <item><see cref="StqBand.TQ1_0"/>: five codes per byte in base 3 -
/// <c>b = t0 + 3*t1 + 9*t2 + 27*t3 + 81*t4</c> with each <c>t = trit + 1</c>, so the largest byte
/// value is 242. 128 codes need 25 full bytes plus one byte carrying the last three (zero-padded),
/// i.e. 26 bytes per group. That is what takes the band to 1.75 bits/weight.</item>
/// </list>
/// </summary>
public static class StqPacking
{
    // byte value -> its 4 (TQ2_0) or 5 (TQ1_0) decoded trits. 1 KiB and 1.25 KiB respectively:
    // both stay resident in L1 across a whole matmul.
    private static readonly sbyte[] _tq2Table = BuildTq2Table();
    private static readonly sbyte[] _tq1Table = BuildTq1Table();

    private static sbyte[] BuildTq2Table()
    {
        var t = new sbyte[256 * 4];
        for (int b = 0; b < 256; b++)
        {
            for (int j = 0; j < 4; j++)
            {
                t[b * 4 + j] = (sbyte)(((b >> (j * 2)) & 0x3) - 1);
            }
        }
        return t;
    }

    private static sbyte[] BuildTq1Table()
    {
        var t = new sbyte[256 * 5];
        for (int b = 0; b < 256; b++)
        {
            int v = b;
            for (int j = 0; j < 5; j++)
            {
                t[b * 5 + j] = (sbyte)((v % 3) - 1);
                v /= 3;
            }
        }
        return t;
    }

    /// <summary>Packs one group of codes into <paramref name="dst"/>, which must be at least
    /// <see cref="StqFormat.CodeBytesPerGroup"/> long. Trailing slots in the final byte are
    /// zero-filled, which decodes to the trit 0 - harmless, since the unpacker is driven by the
    /// group size, not by the byte count.</summary>
    public static void PackGroup(StqBand band, ReadOnlySpan<sbyte> codes, Span<byte> dst)
    {
        switch (band)
        {
            case StqBand.TQ2_0:
            {
                int bytes = (codes.Length + 3) / 4;
                dst[..bytes].Clear();
                for (int i = 0; i < codes.Length; i++)
                {
                    int c = codes[i] + 1;               // {-1,0,1} -> {0,1,2}
                    dst[i >> 2] |= (byte)(c << ((i & 3) * 2));
                }
                break;
            }
            case StqBand.TQ1_0:
            {
                int bytes = (codes.Length + 4) / 5;
                dst[..bytes].Clear();
                for (int i = 0; i < codes.Length; i += 5)
                {
                    int acc = 0;
                    int pow = 1;
                    int end = Math.Min(i + 5, codes.Length);
                    for (int j = i; j < end; j++)
                    {
                        acc += (codes[j] + 1) * pow;
                        pow *= 3;
                    }
                    dst[i / 5] = (byte)acc;
                }
                break;
            }
            case StqBand.Q4_0:
            {
                int half = (codes.Length + 1) / 2;
                dst[..half].Clear();
                for (int i = 0; i < half; i++)
                {
                    dst[i] = (byte)((codes[i] + 8) & 0x0F);       // {-8..7} -> {0..15}
                }
                for (int i = half; i < codes.Length; i++)
                {
                    dst[i - half] |= (byte)(((codes[i] + 8) & 0x0F) << 4);
                }
                break;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(band), band, "Not a packed band.");
        }
    }

    /// <summary>Unpacks <paramref name="count"/> codes from <paramref name="src"/> into
    /// <paramref name="dst"/> as signed bytes in <c>{-1, 0, +1}</c> - the form the VNNI int8 dot
    /// consumes directly.</summary>
    public static void UnpackGroup(StqBand band, ReadOnlySpan<byte> src, Span<sbyte> dst, int count)
    {
        switch (band)
        {
            case StqBand.TQ2_0:
            {
                var table = _tq2Table;
                int i = 0;
                int fullBytes = count >> 2;
                for (int b = 0; b < fullBytes; b++)
                {
                    int t = src[b] * 4;
                    dst[i]     = table[t];
                    dst[i + 1] = table[t + 1];
                    dst[i + 2] = table[t + 2];
                    dst[i + 3] = table[t + 3];
                    i += 4;
                }
                if (i < count)
                {
                    int t = src[fullBytes] * 4;
                    for (int j = 0; i < count; i++, j++)
                    {
                        dst[i] = table[t + j];
                    }
                }
                break;
            }
            case StqBand.TQ1_0:
            {
                var table = _tq1Table;
                int i = 0;
                int fullBytes = count / 5;
                for (int b = 0; b < fullBytes; b++)
                {
                    int t = src[b] * 5;
                    dst[i]     = table[t];
                    dst[i + 1] = table[t + 1];
                    dst[i + 2] = table[t + 2];
                    dst[i + 3] = table[t + 3];
                    dst[i + 4] = table[t + 4];
                    i += 5;
                }
                if (i < count)
                {
                    int t = src[fullBytes] * 5;
                    for (int j = 0; i < count; i++, j++)
                    {
                        dst[i] = table[t + j];
                    }
                }
                break;
            }
            case StqBand.Q4_0:
            {
                // Low nibbles fill dst[0 .. half), high nibbles dst[half .. count). Only the first
                // `pairs` bytes carry a high nibble that is in range, which is every byte except the
                // odd-count tail one.
                int half  = (count + 1) >> 1;
                int pairs = count - half;
                int j = 0;
                if (Vector256.IsHardwareAccelerated && pairs >= Vector256<byte>.Count)
                {
                    j = UnpackQ4Split256(src, dst, half, pairs);
                }
                else if (Vector128.IsHardwareAccelerated && pairs >= Vector128<byte>.Count)
                {
                    j = UnpackQ4Split128(src, dst, half, pairs);
                }

                for (; j < pairs; j++)
                {
                    byte b = src[j];
                    dst[j]        = (sbyte)((b & 0x0F) - 8);
                    dst[j + half] = (sbyte)((b >> 4) - 8);
                }
                for (; j < half; j++)
                {
                    dst[j] = (sbyte)((src[j] & 0x0F) - 8);
                }
                break;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(band), band, "Not a packed band.");
        }
    }

    /// <summary>
    /// Unpacks a whole row of <see cref="StqBand.Q4_0"/> groups in one pass.
    ///
    /// <para>A 128-weight group is only two 256-bit steps, so going through
    /// <see cref="UnpackGroup"/> per group paid a band switch, two span slices and a call for every
    /// two vector operations. Walking the row directly lets the groups' work interleave, which is
    /// what the loop needs to reach store bandwidth rather than call overhead. Falls back to the
    /// per-group path for odd group sizes and groups too short to vectorize.</para>
    /// </summary>
    public static void UnpackQ4Row(ReadOnlySpan<byte> src, Span<sbyte> dst, int groups, int groupSize)
    {
        int half = groupSize >> 1;
        if ((groupSize & 1) != 0 || !Vector256.IsHardwareAccelerated || half < Vector256<byte>.Count)
        {
            for (int g = 0; g < groups; g++)
            {
                UnpackGroup(StqBand.Q4_0, src.Slice(g * ((groupSize + 1) >> 1), (groupSize + 1) >> 1),
                            dst.Slice(g * groupSize, groupSize), groupSize);
            }
            return;
        }

        ref byte  s0 = ref MemoryMarshal.GetReference(src);
        ref sbyte d0 = ref MemoryMarshal.GetReference(dst);
        var mask = Vector256.Create((byte)0x0F);
        var bias = Vector256.Create((byte)8);
        int width = Vector256<byte>.Count;

        // The default group of 128 is exactly two 256-bit steps, and a two-iteration loop is mostly
        // branch. Unrolled, the group's two halves are independent chains and the whole row runs as
        // one straight sequence of loads and stores.
        if (half == 2 * width)
        {
            for (int g = 0; g < groups; g++)
            {
                nuint sBase = (nuint)(g * half);
                nuint dBase = (nuint)(g * groupSize);
                var v0 = Vector256.LoadUnsafe(ref s0, sBase);
                var v1 = Vector256.LoadUnsafe(ref s0, sBase + (nuint)width);
                ((v0 & mask) - bias).AsSByte().StoreUnsafe(ref d0, dBase);
                ((v1 & mask) - bias).AsSByte().StoreUnsafe(ref d0, dBase + (nuint)width);
                (Vector256.ShiftRightLogical(v0, 4) - bias).AsSByte().StoreUnsafe(ref d0, dBase + (nuint)half);
                (Vector256.ShiftRightLogical(v1, 4) - bias).AsSByte().StoreUnsafe(ref d0, dBase + (nuint)(half + width));
            }
            return;
        }

        for (int g = 0; g < groups; g++)
        {
            nuint sBase = (nuint)(g * half);
            nuint dBase = (nuint)(g * groupSize);
            int j = 0;
            for (; j + width <= half; j += width)
            {
                var v = Vector256.LoadUnsafe(ref s0, sBase + (nuint)j);
                ((v & mask) - bias).AsSByte().StoreUnsafe(ref d0, dBase + (nuint)j);
                (Vector256.ShiftRightLogical(v, 4) - bias).AsSByte().StoreUnsafe(ref d0, dBase + (nuint)(j + half));
            }
            for (; j < half; j++)
            {
                byte b = Unsafe.Add(ref s0, sBase + (nuint)j);
                Unsafe.Add(ref d0, dBase + (nuint)j)          = (sbyte)((b & 0x0F) - 8);
                Unsafe.Add(ref d0, dBase + (nuint)(j + half)) = (sbyte)((b >> 4) - 8);
            }
        }
    }

    /// <summary>
    /// Unpacks the byte pairs of a split-layout group with 256-bit vectors: one load produces
    /// thirty-two low codes and thirty-two high codes, each already contiguous in the destination.
    /// Subtracting 8 in byte space yields the right two's-complement sbyte directly, because every
    /// nibble is in [0, 15]. Returns how many source bytes were consumed.
    /// </summary>
    private static int UnpackQ4Split256(ReadOnlySpan<byte> src, Span<sbyte> dst, int half, int pairs)
    {
        ref byte  s0 = ref MemoryMarshal.GetReference(src);
        ref sbyte d0 = ref MemoryMarshal.GetReference(dst);
        var mask = Vector256.Create((byte)0x0F);
        var bias = Vector256.Create((byte)8);

        int width = Vector256<byte>.Count;
        int j = 0;
        for (; j + width <= pairs; j += width)
        {
            var v = Vector256.LoadUnsafe(ref s0, (nuint)j);
            ((v & mask) - bias).AsSByte().StoreUnsafe(ref d0, (nuint)j);
            (Vector256.ShiftRightLogical(v, 4) - bias).AsSByte().StoreUnsafe(ref d0, (nuint)(j + half));
        }
        return j;
    }

    /// <summary>128-bit form of <see cref="UnpackQ4Split256"/>, for groups too short for a 256-bit
    /// step or hosts without one.</summary>
    private static int UnpackQ4Split128(ReadOnlySpan<byte> src, Span<sbyte> dst, int half, int pairs)
    {
        ref byte  s0 = ref MemoryMarshal.GetReference(src);
        ref sbyte d0 = ref MemoryMarshal.GetReference(dst);
        var mask = Vector128.Create((byte)0x0F);
        var bias = Vector128.Create((byte)8);

        int width = Vector128<byte>.Count;
        int j = 0;
        for (; j + width <= pairs; j += width)
        {
            var v = Vector128.LoadUnsafe(ref s0, (nuint)j);
            ((v & mask) - bias).AsSByte().StoreUnsafe(ref d0, (nuint)j);
            (Vector128.ShiftRightLogical(v, 4) - bias).AsSByte().StoreUnsafe(ref d0, (nuint)(j + half));
        }
        return j;
    }

    /// <summary>Unpacks a group straight to float, scaled: <c>dst[i] = scale * trit_i</c>. The float
    /// fallback kernel (and the converter's error report) uses this.</summary>
    public static void UnpackGroupScaled(StqBand band, ReadOnlySpan<byte> src, Span<float> dst, int count, float scale)
    {
        Span<sbyte> codes = count <= 256 ? stackalloc sbyte[count] : new sbyte[count];
        UnpackGroup(band, src, codes, count);
        for (int i = 0; i < count; i++)
        {
            dst[i] = codes[i] * scale;
        }
    }
}
