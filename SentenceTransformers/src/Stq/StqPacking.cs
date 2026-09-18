#nullable enable

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
/// <c>code + 8</c> in <c>[0, 15]</c>, low nibble first. <c>groupSize/2</c> bytes per group.</item>
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
    private static readonly sbyte[] _q4Table = BuildQ4Table();

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

    private static sbyte[] BuildQ4Table()
    {
        var t = new sbyte[256 * 2];
        for (int b = 0; b < 256; b++)
        {
            t[b * 2]     = (sbyte)((b & 0x0F) - 8);
            t[b * 2 + 1] = (sbyte)((b >> 4) - 8);
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
                int bytes = (codes.Length + 1) / 2;
                dst[..bytes].Clear();
                for (int i = 0; i < codes.Length; i++)
                {
                    int nibble = (codes[i] + 8) & 0x0F;   // {-8..7} -> {0..15}
                    dst[i >> 1] |= (byte)((i & 1) == 0 ? nibble : nibble << 4);
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
                var table = _q4Table;
                int i = 0;
                int fullBytes = count >> 1;
                for (int b = 0; b < fullBytes; b++)
                {
                    int t = src[b] * 2;
                    dst[i]     = table[t];
                    dst[i + 1] = table[t + 1];
                    i += 2;
                }
                if (i < count)
                {
                    dst[i] = table[src[fullBytes] * 2];
                }
                break;
            }
            default:
                throw new ArgumentOutOfRangeException(nameof(band), band, "Not a packed band.");
        }
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
