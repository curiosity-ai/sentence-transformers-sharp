// PNG decoding ported from stb_image.h v2.30 by Sean Barrett and contributors
// (https://github.com/nothings/stb), dual-licensed public domain (unlicense) / MIT.
// The port reproduces stb_image's output bit-exactly for stbi_load_from_memory(..., desired_channels: 3);
// only the DEFLATE decompression is delegated to System.IO.Compression.

using System.IO.Compression;

namespace SentenceTransformers.EmbeddingGemma2.Vision;

/// <summary>
/// Pure-managed port of stb_image's PNG decoder (<c>stbi__parse_png_file</c> and friends): color types
/// 0/2/3/4/6, bit depths 1/2/4/8/16, palette with tRNS, Adam7 interlacing, all five filters, multiple
/// IDAT chunks, and Apple's CgBI variant handled the way stb does by default (raw deflate stream, no
/// BGR swap, no un-premultiplication). CRCs are not checked. The result is what
/// <c>stbi_load_from_memory(..., 3)</c> returns: 16-bit samples keep their high byte
/// (<c>stbi__convert_16_to_8</c>), low-bit gray is scaled by <c>stbi__depth_scale_table</c>, gray is
/// replicated and alpha dropped (<c>stbi__convert_format</c>).
/// </summary>
internal static class PngDecoder
{
    private static ReadOnlySpan<byte> Signature => [137, 80, 78, 71, 13, 10, 26, 10];

    /// <summary>stbi__depth_scale_table: maps low-bit gray samples onto 0..255.</summary>
    private static ReadOnlySpan<byte> DepthScaleTable => [0, 0xff, 0x55, 0, 0x11, 0, 0, 0, 0x01];

    /// <summary>first_row_filter: filter substitutions for the first row of each image/pass (no prior row).</summary>
    private static ReadOnlySpan<byte> FirstRowFilter => [FilterNone, FilterSub, FilterNone, FilterAvgFirst, FilterSub];

    private const byte FilterNone = 0;
    private const byte FilterSub = 1;
    private const byte FilterUp = 2;
    private const byte FilterAvg = 3;
    private const byte FilterPaeth = 4;
    private const byte FilterAvgFirst = 5;

    /// <summary>STBI_MAX_DIMENSIONS.</summary>
    private const uint MaxDimensions = 1u << 24;

    private const uint ChunkCgBI = ('C' << 24) | ('g' << 16) | ('B' << 8) | 'I';
    private const uint ChunkIHDR = ('I' << 24) | ('H' << 16) | ('D' << 8) | 'R';
    private const uint ChunkPLTE = ('P' << 24) | ('L' << 16) | ('T' << 8) | 'E';
    private const uint ChunkTRNS = ('t' << 24) | ('R' << 16) | ('N' << 8) | 'S';
    private const uint ChunkIDAT = ('I' << 24) | ('D' << 16) | ('A' << 8) | 'T';
    private const uint ChunkIEND = ('I' << 24) | ('E' << 16) | ('N' << 8) | 'D';

    /// <summary>stbi__png_test: the 8-byte PNG signature.</summary>
    public static bool IsPng(ReadOnlySpan<byte> data) => data.Length >= 8 && data[..8].SequenceEqual(Signature);

    /// <summary>
    /// Decodes a PNG to packed RGB (3 bytes per pixel, row-major, top row first).
    /// </summary>
    /// <exception cref="InvalidDataException">The stream is not a PNG stb_image can decode.</exception>
    public static byte[] Decode(ReadOnlySpan<byte> data, out int width, out int height)
    {
        var s = new Reader(data);
        if (!IsPng(data))
        {
            throw new InvalidDataException("Not a PNG");
        }
        s.Pos = 8;

        Span<byte> palette = stackalloc byte[1024];
        palette.Clear();
        int palImgN = 0;
        uint palLen = 0;
        bool first = true;
        bool isIphone = false;
        int depth = 0, color = 0, interlace = 0, imgN = 0;
        uint imgX = 0, imgY = 0;
        byte[] idata = null;
        uint ioff = 0;

        // stbi__parse_png_file (STBI__SCAN_load)
        for (;;)
        {
            uint length = s.Get32Be();
            uint type = s.Get32Be();
            switch (type)
            {
                case ChunkCgBI:
                    isIphone = true;
                    s.Skip(length);
                    break;

                case ChunkIHDR:
                {
                    if (!first)
                    {
                        throw Corrupt("multiple IHDR");
                    }
                    first = false;
                    if (length != 13)
                    {
                        throw Corrupt("bad IHDR len");
                    }
                    imgX = s.Get32Be();
                    imgY = s.Get32Be();
                    if (imgY > MaxDimensions || imgX > MaxDimensions)
                    {
                        throw new InvalidDataException("Very large image (corrupt?)");
                    }
                    depth = s.Get8();
                    if (depth != 1 && depth != 2 && depth != 4 && depth != 8 && depth != 16)
                    {
                        throw new InvalidDataException("PNG not supported: 1/2/4/8/16-bit only");
                    }
                    color = s.Get8();
                    if (color > 6)
                    {
                        throw Corrupt("bad ctype");
                    }
                    if (color == 3 && depth == 16)
                    {
                        throw Corrupt("bad ctype");
                    }
                    if (color == 3)
                    {
                        palImgN = 3;
                    }
                    else if ((color & 1) != 0)
                    {
                        throw Corrupt("bad ctype");
                    }
                    if (s.Get8() != 0)
                    {
                        throw Corrupt("bad comp method");
                    }
                    if (s.Get8() != 0)
                    {
                        throw Corrupt("bad filter method");
                    }
                    interlace = s.Get8();
                    if (interlace > 1)
                    {
                        throw Corrupt("bad interlace method");
                    }
                    if (imgX == 0 || imgY == 0)
                    {
                        throw Corrupt("0-pixel image");
                    }
                    if (palImgN == 0)
                    {
                        imgN = ((color & 2) != 0 ? 3 : 1) + ((color & 4) != 0 ? 1 : 0);
                        if ((1u << 30) / imgX / (uint)imgN < imgY)
                        {
                            throw new InvalidDataException("Image too large to decode");
                        }
                    }
                    else
                    {
                        // paletted: img_n is the number of components to decompress/filter
                        imgN = 1;
                        if ((1u << 30) / imgX / 4 < imgY)
                        {
                            throw Corrupt("too large");
                        }
                    }
                    break;
                }

                case ChunkPLTE:
                {
                    if (first)
                    {
                        throw Corrupt("first not IHDR");
                    }
                    if (length > 256 * 3)
                    {
                        throw Corrupt("invalid PLTE");
                    }
                    palLen = length / 3;
                    if (palLen * 3 != length)
                    {
                        throw Corrupt("invalid PLTE");
                    }
                    for (int i = 0; i < palLen; ++i)
                    {
                        palette[i * 4 + 0] = s.Get8();
                        palette[i * 4 + 1] = s.Get8();
                        palette[i * 4 + 2] = s.Get8();
                        palette[i * 4 + 3] = 255;
                    }
                    break;
                }

                case ChunkTRNS:
                {
                    if (first)
                    {
                        throw Corrupt("first not IHDR");
                    }
                    if (idata != null)
                    {
                        throw Corrupt("tRNS after IDAT");
                    }
                    if (palImgN != 0)
                    {
                        if (palLen == 0)
                        {
                            throw Corrupt("tRNS before PLTE");
                        }
                        if (length > palLen)
                        {
                            throw Corrupt("bad tRNS len");
                        }
                        palImgN = 4;
                        for (int i = 0; i < length; ++i)
                        {
                            palette[i * 4 + 3] = s.Get8();
                        }
                    }
                    else
                    {
                        if ((imgN & 1) == 0)
                        {
                            throw Corrupt("tRNS with alpha");
                        }
                        if (length != (uint)imgN * 2)
                        {
                            throw Corrupt("bad tRNS len");
                        }
                        // Color-key transparency only affects the alpha channel, which the RGB output drops.
                        s.Skip(length);
                    }
                    break;
                }

                case ChunkIDAT:
                {
                    if (first)
                    {
                        throw Corrupt("first not IHDR");
                    }
                    if (palImgN != 0 && palLen == 0)
                    {
                        throw Corrupt("no PLTE");
                    }
                    if (length > (1u << 30))
                    {
                        throw new InvalidDataException("IDAT section larger than 2^30 bytes");
                    }
                    if ((int)(ioff + length) < (int)ioff)
                    {
                        throw Corrupt("IDAT too large");
                    }
                    if (length > (uint)s.Remaining)
                    {
                        throw Corrupt("outofdata");
                    }
                    if (length > 0)
                    {
                        // grow geometrically, like stb's idata_limit doubling
                        uint need = ioff + length;
                        if (idata == null || need > (uint)idata.Length)
                        {
                            long cap = idata == null ? Math.Max(length, 4096u) : idata.Length;
                            while (cap < need)
                            {
                                cap *= 2;
                            }
                            Array.Resize(ref idata, (int)Math.Min(cap, Array.MaxLength));
                        }
                        s.GetN(idata.AsSpan((int)ioff, (int)length));
                        ioff += length;
                    }
                    break;
                }

                case ChunkIEND:
                {
                    if (first)
                    {
                        throw Corrupt("first not IHDR");
                    }
                    if (idata == null)
                    {
                        throw Corrupt("no IDAT");
                    }
                    width = (int)imgX;
                    height = (int)imgY;
                    return CreateImage(idata, (int)ioff, isIphone, (int)imgX, (int)imgY, imgN, depth, color, interlace != 0, palImgN != 0 ? palette : default);
                }

                default:
                    if (first)
                    {
                        throw Corrupt("first not IHDR");
                    }
                    if ((type & (1u << 29)) == 0)
                    {
                        throw new InvalidDataException("PNG not supported: unknown critical PNG chunk type");
                    }
                    s.Skip(length);
                    break;
            }
            // end of PNG chunk, read and skip CRC
            s.Get32Be();
        }
    }

    private static InvalidDataException Corrupt(string why) => new("Corrupt PNG: " + why);

    /// <summary>Sequential reader with stb's memory-context semantics (reads past the end return 0).</summary>
    private ref struct Reader
    {
        private readonly ReadOnlySpan<byte> _data;
        public int Pos;

        public Reader(ReadOnlySpan<byte> data)
        {
            _data = data;
            Pos = 0;
        }

        public readonly int Remaining => _data.Length - Pos;

        public byte Get8() => Pos < _data.Length ? _data[Pos++] : (byte)0;

        public uint Get32Be()
        {
            uint z = (uint)((Get8() << 8) + Get8());
            return (z << 16) + (uint)((Get8() << 8) + Get8());
        }

        /// <summary>stbi__skip: an out-of-range skip lands at the end of the buffer.</summary>
        public void Skip(uint n)
        {
            Pos = n > (uint)(_data.Length - Pos) ? _data.Length : Pos + (int)n;
        }

        /// <summary>stbi__getn (the caller has checked that enough bytes remain).</summary>
        public void GetN(Span<byte> dst)
        {
            _data.Slice(Pos, dst.Length).CopyTo(dst);
            Pos += dst.Length;
        }
    }

    /// <summary>
    /// Inflates the IDAT stream the way stb's zlib front-end accepts it (<c>stbi__parse_zlib_header</c>
    /// unless CgBI; Adler-32 is not verified) and returns at least <paramref name="needed"/> bytes. Throws
    /// when the stream holds fewer ("not enough pixels"), is corrupt, or ends before its final block.
    /// </summary>
    private static byte[] Inflate(byte[] idata, int idataLength, bool isIphone, long needed)
    {
        ReadOnlySpan<byte> idat = idata.AsSpan(0, idataLength);
        int start = 0;
        if (!isIphone)
        {
            // stbi__parse_zlib_header: reading CMF/FLG must not hit the end of the data
            if (idat.Length <= 2)
            {
                throw Corrupt("bad zlib header");
            }
            int cmf = idat[0];
            int flg = idat[1];
            if ((cmf * 256 + flg) % 31 != 0)
            {
                throw Corrupt("bad zlib header");
            }
            if ((flg & 32) != 0)
            {
                throw Corrupt("no preset dict");
            }
            if ((cmf & 15) != 8)
            {
                throw Corrupt("bad compression");
            }
            start = 2;
        }
        if (needed > Array.MaxLength)
        {
            throw new InvalidDataException("Image too large to decode");
        }

        using var input = new BoundedInput(idata, start, idataLength);
        using var inflater = new DeflateStream(input, CompressionMode.Decompress);

        // Grow the output as data actually arrives so a tiny file cannot force a huge allocation.
        int target = (int)needed;
        var buffer = new byte[(int)Math.Min(target, Math.Max(65536L, (long)idataLength * 8))];
        int filled = 0;
        while (filled < target)
        {
            if (filled == buffer.Length)
            {
                Array.Resize(ref buffer, (int)Math.Min(target, (long)buffer.Length * 2));
            }
            int n = inflater.Read(buffer, filled, buffer.Length - filled);
            if (n == 0)
            {
                break;
            }
            filled += n;
        }
        if (filled < target)
        {
            throw Corrupt("not enough pixels");
        }

        // stb decodes the whole zlib stream, so corrupt trailing data must fail here too.
        Span<byte> scratch = stackalloc byte[4096];
        while (inflater.Read(scratch) > 0)
        {
        }
        return buffer;
    }

    /// <summary>
    /// Read-only input for <see cref="DeflateStream"/> that throws when the inflater asks for bytes past
    /// the end of the IDAT data. A properly terminated DEFLATE stream never does (trailing bytes such as the
    /// Adler-32 are simply left unread), so this reproduces stb's "unexpected end" failure for streams that
    /// stop before their final block, which System.IO.Compression would otherwise accept silently.
    /// </summary>
    private sealed class BoundedInput : Stream
    {
        private readonly byte[] _data;
        private readonly int _end;
        private int _pos;

        public BoundedInput(byte[] data, int start, int end)
        {
            _data = data;
            _pos = start;
            _end = end;
        }

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => throw new NotSupportedException();
        public override long Position
        {
            get => throw new NotSupportedException();
            set => throw new NotSupportedException();
        }

        public override int Read(byte[] buffer, int offset, int count) => Read(buffer.AsSpan(offset, count));

        public override int Read(Span<byte> buffer)
        {
            if (buffer.IsEmpty)
            {
                return 0;
            }
            if (_pos >= _end)
            {
                throw Corrupt("unexpected end of zlib stream");
            }
            int n = Math.Min(buffer.Length, _end - _pos);
            _data.AsSpan(_pos, n).CopyTo(buffer);
            _pos += n;
            return n;
        }

        public override void Flush()
        {
        }

        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    /// <summary>
    /// stbi__create_png_image + stbi__create_png_image_raw + palette expansion + stbi__convert_format to
    /// 3 channels, fused: each unfiltered scanline is written straight into the RGB output.
    /// </summary>
    private static byte[] CreateImage(byte[] idata, int idataLength, bool isIphone, int imgX, int imgY, int imgN, int depth, int color, bool interlaced, ReadOnlySpan<byte> palette)
    {
        ReadOnlySpan<int> xorig = [0, 4, 0, 2, 0, 1, 0];
        ReadOnlySpan<int> yorig = [0, 0, 4, 0, 2, 0, 1];
        ReadOnlySpan<int> xspc = [8, 8, 4, 4, 2, 2, 1];
        ReadOnlySpan<int> yspc = [8, 8, 8, 4, 4, 2, 2];

        int passes = interlaced ? 7 : 1;
        Span<int> passW = stackalloc int[7];
        Span<int> passH = stackalloc int[7];
        long needed = 0;
        long maxWidthBytes = 0;
        for (int p = 0; p < passes; ++p)
        {
            int x, y;
            if (interlaced)
            {
                x = (imgX - xorig[p] + xspc[p] - 1) / xspc[p];
                y = (imgY - yorig[p] + yspc[p] - 1) / yspc[p];
            }
            else
            {
                x = imgX;
                y = imgY;
            }
            passW[p] = x;
            passH[p] = y;
            if (x == 0 || y == 0)
            {
                continue;
            }
            // stbi__mad3sizes_valid(img_n, x, depth, 7) / stbi__mad2sizes_valid(img_width_bytes, y, img_width_bytes)
            long bits = (long)imgN * x * depth;
            if (bits + 7 > int.MaxValue)
            {
                throw Corrupt("too large");
            }
            long widthBytes = (bits + 7) >> 3;
            if (widthBytes * y + widthBytes > int.MaxValue)
            {
                throw Corrupt("too large");
            }
            needed += (widthBytes + 1) * y;
            maxWidthBytes = Math.Max(maxWidthBytes, widthBytes);
        }

        // stbi__convert_format's 3-channel buffer (stb fails on it after inflating; fail before doing the work)
        if ((long)imgX * imgY * 3 > Array.MaxLength)
        {
            throw new InvalidDataException("Image too large to decode");
        }
        byte[] raw = Inflate(idata, idataLength, isIphone, needed);
        var rgb = new byte[imgX * imgY * 3];
        var filterBuf = new byte[maxWidthBytes * 2];
        int rawPos = 0;
        int bytes = depth == 16 ? 2 : 1;
        byte scale = color == 0 && depth < 8 ? DepthScaleTable[depth] : (byte)1; // scale only applies to low-bit gray
        bool isPalette = !palette.IsEmpty;

        for (int p = 0; p < passes; ++p)
        {
            int x = passW[p], y = passH[p];
            if (x == 0 || y == 0)
            {
                continue;
            }
            int x0 = interlaced ? xorig[p] : 0, y0 = interlaced ? yorig[p] : 0;
            int dx = interlaced ? xspc[p] : 1, dy = interlaced ? yspc[p] : 1;
            int imgWidthBytes = (imgN * x * depth + 7) >> 3;
            int filterBytes = depth < 8 ? 1 : imgN * bytes;
            int nk = imgWidthBytes;

            for (int j = 0; j < y; ++j)
            {
                Span<byte> cur = filterBuf.AsSpan((j & 1) * imgWidthBytes, imgWidthBytes);
                Span<byte> prior = filterBuf.AsSpan((~j & 1) * imgWidthBytes, imgWidthBytes);
                int filter = raw[rawPos++];
                if (filter > 4)
                {
                    throw Corrupt("invalid filter");
                }
                if (j == 0)
                {
                    filter = FirstRowFilter[filter];
                }
                ReadOnlySpan<byte> line = raw.AsSpan(rawPos, nk);
                rawPos += nk;
                Unfilter(filter, line, cur, prior, filterBytes);

                // expand the decoded scanline into the RGB output
                int outY = j * dy + y0;
                Span<byte> outRow = rgb.AsSpan(outY * imgX * 3, imgX * 3);
                EmitRow(cur, outRow, x, x0, dx, imgN, depth, scale, isPalette, palette);
            }
        }
        return rgb;
    }

    /// <summary>The filter switch of stbi__create_png_image_raw for one scanline.</summary>
    private static void Unfilter(int filter, ReadOnlySpan<byte> raw, Span<byte> cur, ReadOnlySpan<byte> prior, int filterBytes)
    {
        int nk = raw.Length;
        int fb = Math.Min(filterBytes, nk);
        switch (filter)
        {
            case FilterNone:
                raw.CopyTo(cur);
                break;

            case FilterSub:
                raw[..fb].CopyTo(cur);
                for (int k = filterBytes; k < nk; ++k)
                {
                    cur[k] = (byte)(raw[k] + cur[k - filterBytes]);
                }
                break;

            case FilterUp:
                for (int k = 0; k < nk; ++k)
                {
                    cur[k] = (byte)(raw[k] + prior[k]);
                }
                break;

            case FilterAvg:
                for (int k = 0; k < fb; ++k)
                {
                    cur[k] = (byte)(raw[k] + (prior[k] >> 1));
                }
                for (int k = filterBytes; k < nk; ++k)
                {
                    cur[k] = (byte)(raw[k] + ((prior[k] + cur[k - filterBytes]) >> 1));
                }
                break;

            case FilterPaeth:
                for (int k = 0; k < fb; ++k)
                {
                    cur[k] = (byte)(raw[k] + prior[k]); // prior[k] == stbi__paeth(0, prior[k], 0)
                }
                for (int k = filterBytes; k < nk; ++k)
                {
                    cur[k] = (byte)(raw[k] + Paeth(cur[k - filterBytes], prior[k], prior[k - filterBytes]));
                }
                break;

            case FilterAvgFirst:
                raw[..fb].CopyTo(cur);
                for (int k = filterBytes; k < nk; ++k)
                {
                    cur[k] = (byte)(raw[k] + (cur[k - filterBytes] >> 1));
                }
                break;
        }
    }

    /// <summary>stbi__paeth (branch-free formulation, equivalent to the PNG spec predictor).</summary>
    private static int Paeth(int a, int b, int c)
    {
        int thresh = c * 3 - (a + b);
        int lo = a < b ? a : b;
        int hi = a < b ? b : a;
        int t0 = (hi <= thresh) ? lo : c;
        int t1 = (thresh <= lo) ? hi : t0;
        return t1;
    }

    /// <summary>
    /// Converts one unfiltered scanline of <paramref name="x"/> pixels to RGB and scatters it to output
    /// columns <c>x0 + i*dx</c>: bit expansion with stb's gray scaling, 16-to-8 by keeping the high byte,
    /// palette lookup (stbi__expand_png_palette), gray replication and alpha removal (stbi__convert_format).
    /// </summary>
    private static void EmitRow(ReadOnlySpan<byte> cur, Span<byte> outRow, int x, int x0, int dx, int imgN, int depth, byte scale, bool isPalette, ReadOnlySpan<byte> palette)
    {
        if (depth == 8 && dx == 1)
        {
            // fast paths for the common non-interlaced 8-bit layouts
            if (isPalette)
            {
                for (int i = 0, o = 0; i < x; ++i, o += 3)
                {
                    int n = cur[i] * 4;
                    outRow[o] = palette[n];
                    outRow[o + 1] = palette[n + 1];
                    outRow[o + 2] = palette[n + 2];
                }
                return;
            }
            if (imgN == 3)
            {
                cur[..(x * 3)].CopyTo(outRow);
                return;
            }
            if (imgN == 4)
            {
                for (int i = 0, o = 0, q = 0; i < x; ++i, o += 3, q += 4)
                {
                    outRow[o] = cur[q];
                    outRow[o + 1] = cur[q + 1];
                    outRow[o + 2] = cur[q + 2];
                }
                return;
            }
        }

        int channels = imgN >= 3 ? 3 : 1;
        for (int i = 0; i < x; ++i)
        {
            int o = (x0 + i * dx) * 3;
            int s = i * imgN;
            if (isPalette)
            {
                int n = Sample(cur, s, depth, 1) * 4;
                outRow[o] = palette[n];
                outRow[o + 1] = palette[n + 1];
                outRow[o + 2] = palette[n + 2];
            }
            else if (channels == 1)
            {
                byte g = Sample(cur, s, depth, scale);
                outRow[o] = g;
                outRow[o + 1] = g;
                outRow[o + 2] = g;
            }
            else
            {
                outRow[o] = Sample(cur, s, depth, scale);
                outRow[o + 1] = Sample(cur, s + 1, depth, scale);
                outRow[o + 2] = Sample(cur, s + 2, depth, scale);
            }
        }
    }

    /// <summary>Sample <paramref name="s"/> of a scanline as stb's 8-bit result (low depths MSB-first, scaled; 16-bit high byte).</summary>
    private static byte Sample(ReadOnlySpan<byte> cur, int s, int depth, byte scale)
    {
        switch (depth)
        {
            case 8:
                return cur[s];
            case 16:
                return cur[s * 2];
            case 4:
                return (byte)(scale * ((cur[s >> 1] >> (4 - ((s & 1) << 2))) & 15));
            case 2:
                return (byte)(scale * ((cur[s >> 2] >> (6 - ((s & 3) << 1))) & 3));
            default:
                return (byte)(scale * ((cur[s >> 3] >> (7 - (s & 7))) & 1));
        }
    }
}
