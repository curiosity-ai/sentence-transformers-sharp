// JPEG decoding ported from stb_image.h v2.30 by Sean Barrett and contributors
// (https://github.com/nothings/stb), dual-licensed public domain (unlicense) / MIT.
// The port reproduces stb_image's output bit-exactly for stbi_load_from_memory(..., desired_channels: 3)
// on x86-64 builds, where stb selects its SSE2 kernels: the SSE2 integer IDCT (stbi__idct_simd) is
// reproduced exactly (with Sse2 intrinsics when available, and a scalar emulation of its 16-bit
// wrap/saturation semantics otherwise), the SIMD 2x2 chroma upsampler (stbi__resample_row_hv_2_simd)
// is arithmetically identical to the scalar one, and stb's SIMD YCbCr->RGB kernel only accelerates the
// 4-channel case, so 3-channel output always uses the scalar stbi__YCbCr_to_RGB_row formula.

using System.Diagnostics.CodeAnalysis;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.X86;

namespace SentenceTransformers.EmbeddingGemma2.Vision;

/// <summary>
/// Pure-managed port of stb_image's JPEG decoder (baseline + progressive Huffman, 1/3/4 components,
/// any integer h/v sampling factors 1..4, restart markers, Adobe APP14 transforms incl. CMYK/YCCK).
/// Produces exactly the bytes that <c>stbi_load_from_memory(..., 3)</c> returns: packed 8-bit RGB,
/// top row first. Like stb, EXIF orientation is ignored and arithmetic-coded / lossless / 12-bit
/// JPEGs are rejected. Every call uses its own decoder instance, so the class is thread-safe.
/// </summary>
internal static class JpegDecoder
{
    /// <summary>
    /// <c>stbi__jpeg_test</c>: the stream must start with an SOI marker (<c>FF D8</c>, with any number of
    /// <c>FF</c> fill bytes in between).
    /// </summary>
    public static bool IsJpeg(ReadOnlySpan<byte> data)
    {
        if (data.Length == 0 || data[0] != 0xFF)
        {
            return false;
        }
        int i = 1;
        while (i < data.Length && data[i] == 0xFF)
        {
            i++;
        }
        return i < data.Length && data[i] == 0xD8;
    }

    /// <summary>
    /// Decodes a JPEG to packed RGB (3 bytes per pixel, row-major, top row first).
    /// </summary>
    /// <exception cref="InvalidDataException">The stream is not a JPEG stb_image can decode.</exception>
    public static byte[] Decode(ReadOnlySpan<byte> data, out int width, out int height)
    {
        var decoder = new Decoder(data.ToArray());
        return decoder.Load(out width, out height);
    }

    // ------------------------------------------------------------------------------------------
    // Tables
    // ------------------------------------------------------------------------------------------

    /// <summary>FAST_BITS: number of bits resolved by the Huffman acceleration tables.</summary>
    private const int FastBits = 9;

    /// <summary>STBI__MARKER_none: "no marker pending".</summary>
    private const int MarkerNone = 0xFF;

    /// <summary>stbi__bmask: <c>(1 &lt;&lt; n) - 1</c>.</summary>
    private static ReadOnlySpan<uint> BMask => [0, 1, 3, 7, 15, 31, 63, 127, 255, 511, 1023, 2047, 4095, 8191, 16383, 32767, 65535];

    /// <summary>stbi__jbias: <c>(-1 &lt;&lt; n) + 1</c>.</summary>
    private static ReadOnlySpan<int> JBias => [0, -1, -3, -7, -15, -31, -63, -127, -255, -511, -1023, -2047, -4095, -8191, -16383, -32767];

    /// <summary>stbi__jpeg_dezigzag, padded with 15 extra entries so corrupt runs sample past the end safely.</summary>
    private static ReadOnlySpan<byte> DeZigZag =>
    [
        0, 1, 8, 16, 9, 2, 3, 10,
        17, 24, 32, 25, 18, 11, 4, 5,
        12, 19, 26, 33, 40, 48, 41, 34,
        27, 20, 13, 6, 7, 14, 21, 28,
        35, 42, 49, 56, 57, 50, 43, 36,
        29, 22, 15, 23, 30, 37, 44, 51,
        58, 59, 52, 45, 38, 31, 39, 46,
        53, 60, 61, 54, 47, 55, 62, 63,
        63, 63, 63, 63, 63, 63, 63, 63,
        63, 63, 63, 63, 63, 63, 63,
    ];

    /// <summary>stbi__huffman: canonical Huffman table with a FAST_BITS lookup accelerator.</summary>
    private sealed class Huffman
    {
        public readonly byte[] Fast = new byte[1 << FastBits];
        public readonly ushort[] Code = new ushort[256];
        public readonly byte[] Values = new byte[256];
        public readonly byte[] Size = new byte[257];
        public readonly uint[] MaxCode = new uint[18];
        public readonly int[] Delta = new int[17];

        /// <summary>stbi__build_huffman. Returns false for an invalid code-length list.</summary>
        public bool Build(ReadOnlySpan<int> count)
        {
            int k = 0;
            for (int i = 0; i < 16; ++i)
            {
                for (int j = 0; j < count[i]; ++j)
                {
                    Size[k++] = (byte)(i + 1);
                    if (k >= 257)
                    {
                        return false;
                    }
                }
            }
            Size[k] = 0;

            uint code = 0;
            k = 0;
            int jj;
            for (jj = 1; jj <= 16; ++jj)
            {
                Delta[jj] = unchecked(k - (int)code);
                if (Size[k] == jj)
                {
                    while (Size[k] == jj)
                    {
                        Code[k++] = (ushort)code++;
                    }
                    if (code - 1 >= (1u << jj))
                    {
                        return false;
                    }
                }
                MaxCode[jj] = code << (16 - jj);
                code <<= 1;
            }
            MaxCode[jj] = 0xffffffff;

            Fast.AsSpan().Fill(255);
            for (int i = 0; i < k; ++i)
            {
                int s = Size[i];
                if (s <= FastBits)
                {
                    int c = Code[i] << (FastBits - s);
                    int m = 1 << (FastBits - s);
                    for (int j = 0; j < m; ++j)
                    {
                        Fast[c + j] = (byte)i;
                    }
                }
            }
            return true;
        }

        /// <summary>stbi__build_fast_ac: table decoding run, magnitude and value of small AC coefficients in one lookup.</summary>
        public void BuildFastAc(short[] fastAc)
        {
            for (int i = 0; i < (1 << FastBits); ++i)
            {
                byte fast = Fast[i];
                fastAc[i] = 0;
                if (fast < 255)
                {
                    int rs = Values[fast];
                    int run = (rs >> 4) & 15;
                    int magbits = rs & 15;
                    int len = Size[fast];

                    if (magbits != 0 && len + magbits <= FastBits)
                    {
                        int k = ((i << len) & ((1 << FastBits) - 1)) >> (FastBits - magbits);
                        int m = 1 << (magbits - 1);
                        if (k < m)
                        {
                            k += unchecked((int)((~0u << magbits) + 1));
                        }
                        if (k >= -128 && k <= 127)
                        {
                            fastAc[i] = (short)((k * 256) + (run * 16) + (len + magbits));
                        }
                    }
                }
            }
        }
    }

    /// <summary>One frame component (the <c>img_comp[]</c> entry of <c>stbi__jpeg</c>).</summary>
    private sealed class Component
    {
        public int Id, H, V, Tq, Hd, Ha, DcPred;
        public int X, Y, W2, H2;
        public byte[] Data;
        public short[] Coeff;
        public int CoeffW, CoeffH;
    }

    /// <summary>Upsampling kernel selected per component (<c>stbi__resample.resample</c>).</summary>
    private enum ResampleKind
    {
        Row1,
        V2,
        H2,
        HV2,
        Generic,
    }

    /// <summary>Per-component upsampling state (<c>stbi__resample</c>).</summary>
    private struct ResampleState
    {
        public ResampleKind Kind;
        public int Line0, Line1;
        public int Hs, Vs;
        public int WLores;
        public int YStep;
        public int YPos;
    }

    // ------------------------------------------------------------------------------------------
    // Decoder state (stbi__jpeg + stbi__context)
    // ------------------------------------------------------------------------------------------

    /// <summary>Decoder state for one image (the <c>stbi__jpeg</c> struct plus its memory <c>stbi__context</c>).</summary>
    private sealed class Decoder
    {
        private readonly byte[] _buf;
        private int _pos;
        private readonly int _end;

        private readonly Huffman[] _huffDc = [new(), new(), new(), new()];
        private readonly Huffman[] _huffAc = [new(), new(), new(), new()];
        private readonly ushort[][] _dequant = [new ushort[64], new ushort[64], new ushort[64], new ushort[64]];
        private readonly short[][] _fastAc = [new short[1 << FastBits], new short[1 << FastBits], new short[1 << FastBits], new short[1 << FastBits]];
        private readonly Component[] _comp = [new(), new(), new(), new()];
        private readonly short[] _block = new short[64];

        private int _imgX, _imgY, _imgN;
        private int _hMax, _vMax, _mcuX, _mcuY;

        private uint _codeBuffer;
        private int _codeBits;
        private int _marker;
        private bool _nomore;

        private bool _progressive;
        private int _specStart, _specEnd, _succHigh, _succLow, _eobRun;
        private bool _jfif;
        private int _app14ColorTransform;
        private int _rgb;

        private int _scanN;
        private readonly int[] _order = new int[4];
        private int _restartInterval, _todo;

        public Decoder(byte[] data)
        {
            _buf = data;
            _end = data.Length;
        }

        private static InvalidDataException Corrupt(string why) => new("Corrupt JPEG: " + why);

        // ---- stbi__context reads (memory source: reads past the end return 0) ----

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int Get8() => _pos < _end ? _buf[_pos++] : 0;

        private int Get16Be()
        {
            int z = Get8();
            return (z << 8) + Get8();
        }

        private bool AtEof => _pos >= _end;

        /// <summary>stbi__skip for a memory source.</summary>
        private void Skip(int n)
        {
            if (n == 0)
            {
                return;
            }
            if (n < 0 || n > _end - _pos)
            {
                _pos = _end;
                return;
            }
            _pos += n;
        }

        // ---- entropy decoder ----

        /// <summary>stbi__grow_buffer_unsafe: refill the 32-bit bit buffer, stopping at markers.</summary>
        private void GrowBufferUnsafe()
        {
            do
            {
                uint b = _nomore ? 0u : (uint)Get8();
                if (b == 0xff)
                {
                    int c = Get8();
                    while (c == 0xff)
                    {
                        c = Get8();
                    }
                    if (c != 0)
                    {
                        _marker = c;
                        _nomore = true;
                        return;
                    }
                }
                _codeBuffer |= b << (24 - _codeBits);
                _codeBits += 8;
            }
            while (_codeBits <= 24);
        }

        /// <summary>stbi__jpeg_huff_decode: decode one Huffman symbol, or -1 on a bad code.</summary>
        private int HuffDecode(Huffman h)
        {
            if (_codeBits < 16)
            {
                GrowBufferUnsafe();
            }

            int c = (int)(_codeBuffer >> (32 - FastBits)) & ((1 << FastBits) - 1);
            int k = h.Fast[c];
            if (k < 255)
            {
                int s = h.Size[k];
                if (s > _codeBits)
                {
                    return -1;
                }
                _codeBuffer <<= s;
                _codeBits -= s;
                return h.Values[k];
            }

            uint temp = _codeBuffer >> 16;
            uint[] maxcode = h.MaxCode;
            for (k = FastBits + 1; ; ++k)
            {
                if (temp < maxcode[k])
                {
                    break;
                }
            }
            if (k == 17)
            {
                _codeBits -= 16;
                return -1;
            }

            if (k > _codeBits)
            {
                return -1;
            }

            c = (int)((_codeBuffer >> (32 - k)) & BMask[k]) + h.Delta[k];
            if (c < 0 || c >= 256)
            {
                return -1;
            }

            _codeBits -= k;
            _codeBuffer <<= k;
            return h.Values[c];
        }

        /// <summary>stbi__extend_receive: combined JPEG RECEIVE + EXTEND of an n-bit signed value.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private int ExtendReceive(int n)
        {
            if (_codeBits < n)
            {
                GrowBufferUnsafe();
            }
            if (_codeBits < n)
            {
                return 0;
            }

            int sgn = (int)(_codeBuffer >> 31);
            uint k = BitOperations.RotateLeft(_codeBuffer, n);
            uint mask = BMask[n];
            _codeBuffer = k & ~mask;
            k &= mask;
            _codeBits -= n;
            return unchecked((int)k + (JBias[n] & (sgn - 1)));
        }

        /// <summary>stbi__jpeg_get_bits: n unsigned bits.</summary>
        private int GetBits(int n)
        {
            if (_codeBits < n)
            {
                GrowBufferUnsafe();
            }
            if (_codeBits < n)
            {
                return 0;
            }
            uint k = BitOperations.RotateLeft(_codeBuffer, n);
            uint mask = BMask[n];
            _codeBuffer = k & ~mask;
            k &= mask;
            _codeBits -= n;
            return (int)k;
        }

        /// <summary>stbi__jpeg_get_bit: one bit (nonzero when set).</summary>
        private bool GetBit()
        {
            if (_codeBits < 1)
            {
                GrowBufferUnsafe();
            }
            if (_codeBits < 1)
            {
                return false;
            }
            uint k = _codeBuffer;
            _codeBuffer <<= 1;
            --_codeBits;
            return (k & 0x80000000u) != 0;
        }

        /// <summary>stbi__addints_valid.</summary>
        private static bool AddIntsValid(int a, int b)
        {
            if ((a >= 0) != (b >= 0))
            {
                return true;
            }
            if (a < 0 && b < 0)
            {
                return a >= int.MinValue - b;
            }
            return a <= int.MaxValue - b;
        }

        /// <summary>stbi__mul2shorts_valid.</summary>
        private static bool Mul2ShortsValid(int a, int b)
        {
            if (b == 0 || b == -1)
            {
                return true;
            }
            if ((a >= 0) == (b >= 0))
            {
                return a <= short.MaxValue / b;
            }
            if (b < 0)
            {
                return a <= short.MinValue / b;
            }
            return a >= short.MinValue / b;
        }

        /// <summary>stbi__jpeg_decode_block: one baseline block, dequantized into natural order.</summary>
        private void DecodeBlock(Span<short> data, Huffman hdc, Huffman hac, short[] fac, Component comp, ushort[] dequant)
        {
            if (_codeBits < 16)
            {
                GrowBufferUnsafe();
            }
            int t = HuffDecode(hdc);
            if (t < 0 || t > 15)
            {
                throw Corrupt("bad huffman code");
            }

            data.Clear();

            int diff = t != 0 ? ExtendReceive(t) : 0;
            if (!AddIntsValid(comp.DcPred, diff))
            {
                throw Corrupt("bad delta");
            }
            int dc = comp.DcPred + diff;
            comp.DcPred = dc;
            if (!Mul2ShortsValid(dc, dequant[0]))
            {
                throw Corrupt("can't merge dc and ac");
            }
            data[0] = (short)(dc * dequant[0]);

            ReadOnlySpan<byte> dezigzag = DeZigZag;
            int k = 1;
            do
            {
                if (_codeBits < 16)
                {
                    GrowBufferUnsafe();
                }
                int c = (int)(_codeBuffer >> (32 - FastBits)) & ((1 << FastBits) - 1);
                int r = fac[c];
                int s;
                if (r != 0)
                {
                    // fast-AC path
                    k += (r >> 4) & 15;
                    s = r & 15;
                    if (s > _codeBits)
                    {
                        throw Corrupt("combined length longer than code bits available");
                    }
                    _codeBuffer <<= s;
                    _codeBits -= s;
                    int zig = dezigzag[k++];
                    data[zig] = (short)((r >> 8) * dequant[zig]);
                }
                else
                {
                    int rs = HuffDecode(hac);
                    if (rs < 0)
                    {
                        throw Corrupt("bad huffman code");
                    }
                    s = rs & 15;
                    r = rs >> 4;
                    if (s == 0)
                    {
                        if (rs != 0xf0)
                        {
                            break; // end of block
                        }
                        k += 16;
                    }
                    else
                    {
                        k += r;
                        int zig = dezigzag[k++];
                        data[zig] = (short)(ExtendReceive(s) * dequant[zig]);
                    }
                }
            }
            while (k < 64);
        }

        /// <summary>stbi__jpeg_decode_block_prog_dc: progressive DC first/refinement scan for one block.</summary>
        private void DecodeBlockProgDc(Span<short> data, Huffman hdc, Component comp)
        {
            if (_specEnd != 0)
            {
                throw Corrupt("can't merge dc and ac");
            }

            if (_codeBits < 16)
            {
                GrowBufferUnsafe();
            }

            if (_succHigh == 0)
            {
                data.Clear();
                int t = HuffDecode(hdc);
                if (t < 0 || t > 15)
                {
                    throw Corrupt("can't merge dc and ac");
                }
                int diff = t != 0 ? ExtendReceive(t) : 0;

                if (!AddIntsValid(comp.DcPred, diff))
                {
                    throw Corrupt("bad delta");
                }
                int dc = comp.DcPred + diff;
                comp.DcPred = dc;
                if (!Mul2ShortsValid(dc, 1 << _succLow))
                {
                    throw Corrupt("can't merge dc and ac");
                }
                data[0] = (short)(dc * (1 << _succLow));
            }
            else
            {
                if (GetBit())
                {
                    data[0] += (short)(1 << _succLow);
                }
            }
        }

        /// <summary>stbi__jpeg_decode_block_prog_ac: progressive AC first/refinement scan for one block.</summary>
        private void DecodeBlockProgAc(Span<short> data, Huffman hac, short[] fac)
        {
            if (_specStart == 0)
            {
                throw Corrupt("can't merge dc and ac");
            }

            ReadOnlySpan<byte> dezigzag = DeZigZag;
            if (_succHigh == 0)
            {
                int shift = _succLow;

                if (_eobRun != 0)
                {
                    --_eobRun;
                    return;
                }

                int k = _specStart;
                do
                {
                    if (_codeBits < 16)
                    {
                        GrowBufferUnsafe();
                    }
                    int c = (int)(_codeBuffer >> (32 - FastBits)) & ((1 << FastBits) - 1);
                    int r = fac[c];
                    int s;
                    if (r != 0)
                    {
                        k += (r >> 4) & 15;
                        s = r & 15;
                        if (s > _codeBits)
                        {
                            throw Corrupt("combined length longer than code bits available");
                        }
                        _codeBuffer <<= s;
                        _codeBits -= s;
                        int zig = dezigzag[k++];
                        data[zig] = (short)((r >> 8) * (1 << shift));
                    }
                    else
                    {
                        int rs = HuffDecode(hac);
                        if (rs < 0)
                        {
                            throw Corrupt("bad huffman code");
                        }
                        s = rs & 15;
                        r = rs >> 4;
                        if (s == 0)
                        {
                            if (r < 15)
                            {
                                _eobRun = 1 << r;
                                if (r != 0)
                                {
                                    _eobRun += GetBits(r);
                                }
                                --_eobRun;
                                break;
                            }
                            k += 16;
                        }
                        else
                        {
                            k += r;
                            int zig = dezigzag[k++];
                            data[zig] = (short)(ExtendReceive(s) * (1 << shift));
                        }
                    }
                }
                while (k <= _specEnd);
            }
            else
            {
                // refinement scan for these AC coefficients
                short bit = (short)(1 << _succLow);

                if (_eobRun != 0)
                {
                    --_eobRun;
                    for (int k = _specStart; k <= _specEnd; ++k)
                    {
                        ref short p = ref data[dezigzag[k]];
                        if (p != 0)
                        {
                            if (GetBit())
                            {
                                if ((p & bit) == 0)
                                {
                                    if (p > 0)
                                    {
                                        p += bit;
                                    }
                                    else
                                    {
                                        p -= bit;
                                    }
                                }
                            }
                        }
                    }
                }
                else
                {
                    int k = _specStart;
                    do
                    {
                        int rs = HuffDecode(hac);
                        if (rs < 0)
                        {
                            throw Corrupt("bad huffman code");
                        }
                        int s = rs & 15;
                        int r = rs >> 4;
                        if (s == 0)
                        {
                            if (r < 15)
                            {
                                _eobRun = (1 << r) - 1;
                                if (r != 0)
                                {
                                    _eobRun += GetBits(r);
                                }
                                r = 64; // force end of block
                            }
                            // else: r=15 s=0 writes 16 zeros: a run of 15 zeros then s (which is 0)
                        }
                        else
                        {
                            if (s != 1)
                            {
                                throw Corrupt("bad huffman code");
                            }
                            s = GetBit() ? bit : -bit;
                        }

                        // advance by r
                        while (k <= _specEnd)
                        {
                            ref short p = ref data[dezigzag[k++]];
                            if (p != 0)
                            {
                                if (GetBit())
                                {
                                    if ((p & bit) == 0)
                                    {
                                        if (p > 0)
                                        {
                                            p += bit;
                                        }
                                        else
                                        {
                                            p -= bit;
                                        }
                                    }
                                }
                            }
                            else
                            {
                                if (r == 0)
                                {
                                    p = (short)s;
                                    break;
                                }
                                --r;
                            }
                        }
                    }
                    while (k <= _specEnd);
                }
            }
        }

        // ---- markers / headers ----

        /// <summary>stbi__get_marker: the pending marker from the entropy stream, else the next marker in the stream (0xFF if none).</summary>
        private int GetMarker()
        {
            int x;
            if (_marker != MarkerNone)
            {
                x = _marker;
                _marker = MarkerNone;
                return x;
            }
            x = Get8();
            if (x != 0xff)
            {
                return MarkerNone;
            }
            while (x == 0xff)
            {
                x = Get8();
            }
            return x;
        }

        private static bool IsRestart(int x) => x >= 0xd0 && x <= 0xd7;

        /// <summary>stbi__jpeg_reset: reset the entropy decoder and DC predictions after a restart.</summary>
        private void Reset()
        {
            _codeBits = 0;
            _codeBuffer = 0;
            _nomore = false;
            _comp[0].DcPred = _comp[1].DcPred = _comp[2].DcPred = _comp[3].DcPred = 0;
            _marker = MarkerNone;
            _todo = _restartInterval != 0 ? _restartInterval : 0x7fffffff;
            _eobRun = 0;
        }

        /// <summary>Restart-interval countdown shared by all scan loops. Returns false when the scan must end early.</summary>
        private bool CountdownRestart()
        {
            if (--_todo <= 0)
            {
                if (_codeBits < 24)
                {
                    GrowBufferUnsafe();
                }
                // if it's NOT a restart, then just bail, so we get corrupt data rather than no data
                if (!IsRestart(_marker))
                {
                    return false;
                }
                Reset();
            }
            return true;
        }

        /// <summary>stbi__parse_entropy_coded_data: decode one scan.</summary>
        private void ParseEntropyCodedData()
        {
            Reset();
            short[] block = _block;
            if (!_progressive)
            {
                if (_scanN == 1)
                {
                    // non-interleaved data: one block at a time, in trivial scanline order
                    Component comp = _comp[_order[0]];
                    int w = (comp.X + 7) >> 3;
                    int h = (comp.Y + 7) >> 3;
                    Huffman hdc = _huffDc[comp.Hd], hac = _huffAc[comp.Ha];
                    short[] fac = _fastAc[comp.Ha];
                    ushort[] dq = _dequant[comp.Tq];
                    for (int j = 0; j < h; ++j)
                    {
                        for (int i = 0; i < w; ++i)
                        {
                            DecodeBlock(block, hdc, hac, fac, comp, dq);
                            Idct(block, comp.Data, comp.W2 * j * 8 + i * 8, comp.W2);
                            if (!CountdownRestart())
                            {
                                return;
                            }
                        }
                    }
                }
                else
                {
                    // interleaved
                    for (int j = 0; j < _mcuY; ++j)
                    {
                        for (int i = 0; i < _mcuX; ++i)
                        {
                            for (int k = 0; k < _scanN; ++k)
                            {
                                Component comp = _comp[_order[k]];
                                Huffman hdc = _huffDc[comp.Hd], hac = _huffAc[comp.Ha];
                                short[] fac = _fastAc[comp.Ha];
                                ushort[] dq = _dequant[comp.Tq];
                                for (int y = 0; y < comp.V; ++y)
                                {
                                    for (int x = 0; x < comp.H; ++x)
                                    {
                                        int x2 = (i * comp.H + x) * 8;
                                        int y2 = (j * comp.V + y) * 8;
                                        DecodeBlock(block, hdc, hac, fac, comp, dq);
                                        Idct(block, comp.Data, comp.W2 * y2 + x2, comp.W2);
                                    }
                                }
                            }
                            if (!CountdownRestart())
                            {
                                return;
                            }
                        }
                    }
                }
            }
            else
            {
                if (_scanN == 1)
                {
                    Component comp = _comp[_order[0]];
                    int w = (comp.X + 7) >> 3;
                    int h = (comp.Y + 7) >> 3;
                    for (int j = 0; j < h; ++j)
                    {
                        for (int i = 0; i < w; ++i)
                        {
                            Span<short> data = comp.Coeff.AsSpan(64 * (i + j * comp.CoeffW), 64);
                            if (_specStart == 0)
                            {
                                DecodeBlockProgDc(data, _huffDc[comp.Hd], comp);
                            }
                            else
                            {
                                DecodeBlockProgAc(data, _huffAc[comp.Ha], _fastAc[comp.Ha]);
                            }
                            if (!CountdownRestart())
                            {
                                return;
                            }
                        }
                    }
                }
                else
                {
                    // interleaved (DC only)
                    for (int j = 0; j < _mcuY; ++j)
                    {
                        for (int i = 0; i < _mcuX; ++i)
                        {
                            for (int k = 0; k < _scanN; ++k)
                            {
                                Component comp = _comp[_order[k]];
                                for (int y = 0; y < comp.V; ++y)
                                {
                                    for (int x = 0; x < comp.H; ++x)
                                    {
                                        int x2 = i * comp.H + x;
                                        int y2 = j * comp.V + y;
                                        Span<short> data = comp.Coeff.AsSpan(64 * (x2 + y2 * comp.CoeffW), 64);
                                        DecodeBlockProgDc(data, _huffDc[comp.Hd], comp);
                                    }
                                }
                            }
                            if (!CountdownRestart())
                            {
                                return;
                            }
                        }
                    }
                }
            }
        }

        /// <summary>stbi__jpeg_finish: dequantize and IDCT the coefficients of a progressive image.</summary>
        private void Finish()
        {
            if (!_progressive)
            {
                return;
            }
            short[] block = _block;
            for (int n = 0; n < _imgN; ++n)
            {
                Component comp = _comp[n];
                ushort[] dq = _dequant[comp.Tq];
                int w = (comp.X + 7) >> 3;
                int h = (comp.Y + 7) >> 3;
                for (int j = 0; j < h; ++j)
                {
                    for (int i = 0; i < w; ++i)
                    {
                        // stbi__jpeg_dequantize (short wrap-around, as in C)
                        ReadOnlySpan<short> src = comp.Coeff.AsSpan(64 * (i + j * comp.CoeffW), 64);
                        for (int q = 0; q < 64; ++q)
                        {
                            block[q] = (short)(src[q] * dq[q]);
                        }
                        Idct(block, comp.Data, comp.W2 * j * 8 + i * 8, comp.W2);
                    }
                }
            }
        }

        /// <summary>stbi__process_marker: DRI / DQT / DHT / APPn / COM. Returns false on error.</summary>
        private bool ProcessMarker(int m)
        {
            int l;
            switch (m)
            {
                case MarkerNone:
                    return false;

                case 0xDD: // DRI
                    if (Get16Be() != 4)
                    {
                        return false;
                    }
                    _restartInterval = Get16Be();
                    return true;

                case 0xDB: // DQT
                    l = Get16Be() - 2;
                    while (l > 0)
                    {
                        int q = Get8();
                        int p = q >> 4;
                        bool sixteen = p != 0;
                        int t = q & 15;
                        if (p != 0 && p != 1)
                        {
                            return false;
                        }
                        if (t > 3)
                        {
                            return false;
                        }
                        ushort[] dq = _dequant[t];
                        ReadOnlySpan<byte> dezigzag = DeZigZag;
                        for (int i = 0; i < 64; ++i)
                        {
                            dq[dezigzag[i]] = (ushort)(sixteen ? Get16Be() : Get8());
                        }
                        l -= sixteen ? 129 : 65;
                    }
                    return l == 0;

                case 0xC4: // DHT
                    l = Get16Be() - 2;
                    Span<int> sizes = stackalloc int[16];
                    while (l > 0)
                    {
                        int n = 0;
                        int q = Get8();
                        int tc = q >> 4;
                        int th = q & 15;
                        if (tc > 1 || th > 3)
                        {
                            return false;
                        }
                        for (int i = 0; i < 16; ++i)
                        {
                            sizes[i] = Get8();
                            n += sizes[i];
                        }
                        if (n > 256)
                        {
                            return false;
                        }
                        l -= 17;
                        Huffman h = tc == 0 ? _huffDc[th] : _huffAc[th];
                        if (!h.Build(sizes))
                        {
                            return false;
                        }
                        for (int i = 0; i < n; ++i)
                        {
                            h.Values[i] = (byte)Get8();
                        }
                        if (tc != 0)
                        {
                            h.BuildFastAc(_fastAc[th]);
                        }
                        l -= n;
                    }
                    return l == 0;
            }

            // comment block or APP blocks
            if ((m >= 0xE0 && m <= 0xEF) || m == 0xFE)
            {
                l = Get16Be();
                if (l < 2)
                {
                    return false;
                }
                l -= 2;

                if (m == 0xE0 && l >= 5)
                {
                    // JFIF APP0 segment
                    ReadOnlySpan<byte> tag = "JFIF\0"u8;
                    bool ok = true;
                    for (int i = 0; i < 5; ++i)
                    {
                        if (Get8() != tag[i])
                        {
                            ok = false;
                        }
                    }
                    l -= 5;
                    if (ok)
                    {
                        _jfif = true;
                    }
                }
                else if (m == 0xEE && l >= 12)
                {
                    // Adobe APP14 segment
                    ReadOnlySpan<byte> tag = "Adobe\0"u8;
                    bool ok = true;
                    for (int i = 0; i < 6; ++i)
                    {
                        if (Get8() != tag[i])
                        {
                            ok = false;
                        }
                    }
                    l -= 6;
                    if (ok)
                    {
                        Get8();     // version
                        Get16Be();  // flags0
                        Get16Be();  // flags1
                        _app14ColorTransform = Get8();
                        l -= 6;
                    }
                }

                Skip(l);
                return true;
            }

            return false;
        }

        /// <summary>stbi__process_scan_header (after SOS). Returns false on error.</summary>
        private bool ProcessScanHeader()
        {
            int ls = Get16Be();
            _scanN = Get8();
            if (_scanN < 1 || _scanN > 4 || _scanN > _imgN)
            {
                return false;
            }
            if (ls != 6 + 2 * _scanN)
            {
                return false;
            }
            for (int i = 0; i < _scanN; ++i)
            {
                int id = Get8();
                int q = Get8();
                int which;
                for (which = 0; which < _imgN; ++which)
                {
                    if (_comp[which].Id == id)
                    {
                        break;
                    }
                }
                if (which == _imgN)
                {
                    return false;
                }
                _comp[which].Hd = q >> 4;
                if (_comp[which].Hd > 3)
                {
                    return false;
                }
                _comp[which].Ha = q & 15;
                if (_comp[which].Ha > 3)
                {
                    return false;
                }
                _order[i] = which;
            }

            _specStart = Get8();
            _specEnd = Get8(); // should be 63, but might be 0
            int aa = Get8();
            _succHigh = aa >> 4;
            _succLow = aa & 15;
            if (_progressive)
            {
                if (_specStart > 63 || _specEnd > 63 || _specStart > _specEnd || _succHigh > 13 || _succLow > 13)
                {
                    return false;
                }
            }
            else
            {
                if (_specStart != 0)
                {
                    return false;
                }
                if (_succHigh != 0 || _succLow != 0)
                {
                    return false;
                }
                _specEnd = 63;
            }
            return true;
        }

        /// <summary>stbi__mad3sizes_valid: a*b*c + add fits in a non-negative int.</summary>
        private static bool Mad3SizesValid(long a, long b, long c, long add)
            => a >= 0 && b >= 0 && c >= 0 && a * b <= int.MaxValue && a * b * c <= int.MaxValue && a * b * c + add <= int.MaxValue;

        /// <summary>stbi__process_frame_header (SOF0/1/2) with buffer allocation (STBI__SCAN_load).</summary>
        private void ProcessFrameHeader()
        {
            int lf = Get16Be();
            if (lf < 11)
            {
                throw Corrupt("bad SOF len");
            }
            int p = Get8();
            if (p != 8)
            {
                throw new InvalidDataException("JPEG format not supported: 8-bit only");
            }
            _imgY = Get16Be();
            if (_imgY == 0)
            {
                throw new InvalidDataException("JPEG format not supported: delayed height");
            }
            _imgX = Get16Be();
            if (_imgX == 0)
            {
                throw Corrupt("0 width");
            }
            int c = Get8();
            if (c != 3 && c != 1 && c != 4)
            {
                throw Corrupt("bad component count");
            }
            _imgN = c;

            if (lf != 8 + 3 * _imgN)
            {
                throw Corrupt("bad SOF len");
            }

            _rgb = 0;
            ReadOnlySpan<byte> rgb = "RGB"u8;
            for (int i = 0; i < _imgN; ++i)
            {
                Component comp = _comp[i];
                comp.Id = Get8();
                if (_imgN == 3 && comp.Id == rgb[i])
                {
                    ++_rgb;
                }
                int q = Get8();
                comp.H = q >> 4;
                if (comp.H == 0 || comp.H > 4)
                {
                    throw Corrupt("bad H");
                }
                comp.V = q & 15;
                if (comp.V == 0 || comp.V > 4)
                {
                    throw Corrupt("bad V");
                }
                comp.Tq = Get8();
                if (comp.Tq > 3)
                {
                    throw Corrupt("bad TQ");
                }
            }

            if (!Mad3SizesValid(_imgX, _imgY, _imgN, 0))
            {
                throw new InvalidDataException("JPEG too large to decode");
            }
            // load_jpeg_image's RGB output (stbi__malloc_mad3(3, x, y, 1)) would fail after decoding; fail early instead
            if (!Mad3SizesValid(3, _imgX, _imgY, 1) || (long)_imgX * _imgY * 3 > Array.MaxLength)
            {
                throw new InvalidDataException("JPEG too large to decode");
            }

            int hMax = 1, vMax = 1;
            for (int i = 0; i < _imgN; ++i)
            {
                hMax = Math.Max(hMax, _comp[i].H);
                vMax = Math.Max(vMax, _comp[i].V);
            }

            // plane subsampling factors must be integer ratios
            for (int i = 0; i < _imgN; ++i)
            {
                if (hMax % _comp[i].H != 0)
                {
                    throw Corrupt("bad H");
                }
                if (vMax % _comp[i].V != 0)
                {
                    throw Corrupt("bad V");
                }
            }

            _hMax = hMax;
            _vMax = vMax;
            int mcuW = hMax * 8;
            int mcuH = vMax * 8;
            _mcuX = (_imgX + mcuW - 1) / mcuW;
            _mcuY = (_imgY + mcuH - 1) / mcuH;

            for (int i = 0; i < _imgN; ++i)
            {
                Component comp = _comp[i];
                comp.X = (_imgX * comp.H + hMax - 1) / hMax;
                comp.Y = (_imgY * comp.V + vMax - 1) / vMax;
                comp.W2 = _mcuX * comp.H * 8;
                comp.H2 = _mcuY * comp.V * 8;
                if (!Mad3SizesValid(comp.W2, comp.H2, 1, 15))
                {
                    throw new InvalidDataException("JPEG too large to decode");
                }
                comp.Data = new byte[comp.W2 * comp.H2];
                comp.Coeff = null;
                if (_progressive)
                {
                    comp.CoeffW = comp.W2 / 8;
                    comp.CoeffH = comp.H2 / 8;
                    if (!Mad3SizesValid(comp.W2, comp.H2, 2, 15))
                    {
                        throw new InvalidDataException("JPEG too large to decode");
                    }
                    comp.Coeff = new short[comp.W2 * comp.H2];
                }
            }
        }

        private static bool IsSof(int x) => x == 0xc0 || x == 0xc1 || x == 0xc2;

        /// <summary>stbi__decode_jpeg_header (STBI__SCAN_load).</summary>
        private void DecodeHeader()
        {
            _jfif = false;
            _app14ColorTransform = -1;
            _marker = MarkerNone;
            int m = GetMarker();
            if (m != 0xD8)
            {
                throw Corrupt("no SOI");
            }
            m = GetMarker();
            while (!IsSof(m))
            {
                if (!ProcessMarker(m))
                {
                    throw Corrupt(m == MarkerNone ? "expected marker" : $"unknown marker 0x{m:X2}");
                }
                m = GetMarker();
                while (m == MarkerNone)
                {
                    // some files have extra padding after their blocks, so ok, we'll scan
                    if (AtEof)
                    {
                        throw Corrupt("no SOF");
                    }
                    m = GetMarker();
                }
            }
            _progressive = m == 0xc2;
            ProcessFrameHeader();
        }

        /// <summary>stbi__skip_jpeg_junk_at_end: skip garbage after a scan, resuming at what looks like a marker.</summary>
        private int SkipJunkAtEnd()
        {
            while (!AtEof)
            {
                int x = Get8();
                while (x == 0xff)
                {
                    if (AtEof)
                    {
                        return MarkerNone;
                    }
                    x = Get8();
                    if (x != 0x00 && x != 0xff)
                    {
                        return x;
                    }
                }
            }
            return MarkerNone;
        }

        /// <summary>stbi__decode_jpeg_image: decode all scans into the per-component planes.</summary>
        private void DecodeImage()
        {
            _restartInterval = 0;
            DecodeHeader();
            int m = GetMarker();
            while (m != 0xD9)
            {
                if (m == 0xDA)
                {
                    if (!ProcessScanHeader())
                    {
                        throw Corrupt("bad SOS");
                    }
                    ParseEntropyCodedData();
                    if (_marker == MarkerNone)
                    {
                        _marker = SkipJunkAtEnd();
                    }
                    m = GetMarker();
                    if (IsRestart(m))
                    {
                        m = GetMarker();
                    }
                }
                else if (m == 0xDC)
                {
                    int ld = Get16Be();
                    int nl = Get16Be();
                    if (ld != 4)
                    {
                        throw Corrupt("bad DNL len");
                    }
                    if (nl != _imgY)
                    {
                        throw Corrupt("bad DNL height");
                    }
                    m = GetMarker();
                }
                else
                {
                    // stb treats an unexpected marker after the frame as "done" and returns what it has,
                    // without running the progressive finish pass.
                    if (!ProcessMarker(m))
                    {
                        return;
                    }
                    m = GetMarker();
                }
            }
            Finish();
        }

        // ---- output ----

        /// <summary>load_jpeg_image with req_comp = 3: decode, upsample and color-convert to packed RGB.</summary>
        public byte[] Load(out int width, out int height)
        {
            DecodeImage();

            int imgX = _imgX, imgY = _imgY, imgN = _imgN;
            bool isRgb = imgN == 3 && (_rgb == 3 || (_app14ColorTransform == 0 && !_jfif));
            int decodeN = imgN;

            var res = new ResampleState[4];
            var lineBuf = new byte[4][];
            for (int k = 0; k < decodeN; ++k)
            {
                Component comp = _comp[k];
                ref ResampleState r = ref res[k];
                lineBuf[k] = new byte[imgX + 3];
                r.Hs = _hMax / comp.H;
                r.Vs = _vMax / comp.V;
                r.YStep = r.Vs >> 1;
                r.WLores = (imgX + r.Hs - 1) / r.Hs;
                r.YPos = 0;
                r.Line0 = r.Line1 = 0;
                r.Kind = (r.Hs, r.Vs) switch
                {
                    (1, 1) => ResampleKind.Row1,
                    (1, 2) => ResampleKind.V2,
                    (2, 1) => ResampleKind.H2,
                    (2, 2) => ResampleKind.HV2,
                    _ => ResampleKind.Generic,
                };
            }

            var output = new byte[imgX * imgY * 3];
            int app14 = _app14ColorTransform;
            int stride = imgX * 3;

            for (int j = 0; j < imgY; ++j)
            {
                Span<byte> outRow = output.AsSpan(stride * j, stride);
                ReadOnlySpan<byte> c0 = default, c1 = default, c2 = default, c3 = default;
                for (int k = 0; k < decodeN; ++k)
                {
                    Component comp = _comp[k];
                    ref ResampleState r = ref res[k];
                    bool yBot = r.YStep >= (r.Vs >> 1);
                    ReadOnlySpan<byte> near = comp.Data.AsSpan(yBot ? r.Line1 : r.Line0);
                    ReadOnlySpan<byte> far = comp.Data.AsSpan(yBot ? r.Line0 : r.Line1);
                    ReadOnlySpan<byte> row = Resample(r.Kind, lineBuf[k], near, far, r.WLores, r.Hs);
                    switch (k)
                    {
                        case 0: c0 = row; break;
                        case 1: c1 = row; break;
                        case 2: c2 = row; break;
                        default: c3 = row; break;
                    }
                    if (++r.YStep >= r.Vs)
                    {
                        r.YStep = 0;
                        r.Line0 = r.Line1;
                        if (++r.YPos < comp.Y)
                        {
                            r.Line1 += comp.W2;
                        }
                    }
                }

                if (imgN == 3)
                {
                    if (isRgb)
                    {
                        for (int i = 0, o = 0; i < imgX; ++i, o += 3)
                        {
                            outRow[o] = c0[i];
                            outRow[o + 1] = c1[i];
                            outRow[o + 2] = c2[i];
                        }
                    }
                    else
                    {
                        YCbCrToRgbRow(outRow, c0, c1, c2, imgX);
                    }
                }
                else if (imgN == 4)
                {
                    if (app14 == 0)
                    {
                        // CMYK (Adobe-inverted): multiply each channel by K
                        for (int i = 0, o = 0; i < imgX; ++i, o += 3)
                        {
                            byte m = c3[i];
                            outRow[o] = Blinn8x8(c0[i], m);
                            outRow[o + 1] = Blinn8x8(c1[i], m);
                            outRow[o + 2] = Blinn8x8(c2[i], m);
                        }
                    }
                    else if (app14 == 2)
                    {
                        // YCCK
                        YCbCrToRgbRow(outRow, c0, c1, c2, imgX);
                        for (int i = 0, o = 0; i < imgX; ++i, o += 3)
                        {
                            byte m = c3[i];
                            outRow[o] = Blinn8x8((byte)(255 - outRow[o]), m);
                            outRow[o + 1] = Blinn8x8((byte)(255 - outRow[o + 1]), m);
                            outRow[o + 2] = Blinn8x8((byte)(255 - outRow[o + 2]), m);
                        }
                    }
                    else
                    {
                        // YCbCr + alpha? stb ignores the fourth channel
                        YCbCrToRgbRow(outRow, c0, c1, c2, imgX);
                    }
                }
                else
                {
                    for (int i = 0, o = 0; i < imgX; ++i, o += 3)
                    {
                        byte y = c0[i];
                        outRow[o] = y;
                        outRow[o + 1] = y;
                        outRow[o + 2] = y;
                    }
                }
            }

            width = imgX;
            height = imgY;
            return output;
        }
    }

    // ------------------------------------------------------------------------------------------
    // Upsampling and color conversion
    // ------------------------------------------------------------------------------------------

    /// <summary>
    /// Runs one row through the selected upsampler: resample_row_1, stbi__resample_row_v_2,
    /// stbi__resample_row_h_2, stbi__resample_row_hv_2 (the SSE2 variant stb uses on x86 computes
    /// exactly the same values) or stbi__resample_row_generic.
    /// </summary>
    private static ReadOnlySpan<byte> Resample(ResampleKind kind, Span<byte> output, ReadOnlySpan<byte> near, ReadOnlySpan<byte> far, int w, int hs)
    {
        switch (kind)
        {
            case ResampleKind.Row1:
                return near;

            case ResampleKind.V2:
            {
                // stbi__resample_row_v_2
                near = near[..w];
                far = far[..w];
                for (int i = 0; i < near.Length; ++i)
                {
                    output[i] = (byte)((3 * near[i] + far[i] + 2) >> 2);
                }
                return output;
            }

            case ResampleKind.H2:
            {
                // stbi__resample_row_h_2
                ReadOnlySpan<byte> input = near[..w];
                if (w == 1)
                {
                    output[0] = output[1] = input[0];
                    return output;
                }
                output[0] = input[0];
                output[1] = (byte)((input[0] * 3 + input[1] + 2) >> 2);
                int i;
                for (i = 1; i < w - 1; ++i)
                {
                    int n = 3 * input[i] + 2;
                    output[i * 2] = (byte)((n + input[i - 1]) >> 2);
                    output[i * 2 + 1] = (byte)((n + input[i + 1]) >> 2);
                }
                output[i * 2] = (byte)((input[w - 2] * 3 + input[w - 1] + 2) >> 2);
                output[i * 2 + 1] = input[w - 1];
                return output;
            }

            case ResampleKind.HV2:
            {
                // stbi__resample_row_hv_2
                near = near[..w];
                far = far[..w];
                if (w == 1)
                {
                    output[0] = output[1] = (byte)((3 * near[0] + far[0] + 2) >> 2);
                    return output;
                }
                int t1 = 3 * near[0] + far[0];
                output[0] = (byte)((t1 + 2) >> 2);
                for (int i = 1; i < w; ++i)
                {
                    int t0 = t1;
                    t1 = 3 * near[i] + far[i];
                    output[i * 2 - 1] = (byte)((3 * t0 + t1 + 8) >> 4);
                    output[i * 2] = (byte)((3 * t1 + t0 + 8) >> 4);
                }
                output[w * 2 - 1] = (byte)((t1 + 2) >> 2);
                return output;
            }

            default:
            {
                // stbi__resample_row_generic: nearest neighbour
                for (int i = 0; i < w; ++i)
                {
                    byte v = near[i];
                    for (int j = 0; j < hs; ++j)
                    {
                        output[i * hs + j] = v;
                    }
                }
                return output;
            }
        }
    }

    /// <summary>
    /// stbi__YCbCr_to_RGB_row with step 3 (stb's SSE2 kernel only vectorizes step 4, so for
    /// 3-channel output the scalar formula is what runs in the reference).
    /// </summary>
    private static void YCbCrToRgbRow(Span<byte> output, ReadOnlySpan<byte> y, ReadOnlySpan<byte> pcb, ReadOnlySpan<byte> pcr, int count)
    {
        const int Cr2R = 5743 << 8;      // stbi__float2fixed(1.40200f)
        const int Cr2G = -(2925 << 8);   // -stbi__float2fixed(0.71414f)
        const int Cb2G = -(1410 << 8);   // -stbi__float2fixed(0.34414f)
        const int Cb2B = 7258 << 8;      // stbi__float2fixed(1.77200f)
        y = y[..count];
        pcb = pcb[..count];
        pcr = pcr[..count];
        output = output[..(count * 3)];
        for (int i = 0, o = 0; i < y.Length; ++i, o += 3)
        {
            int yFixed = (y[i] << 20) + (1 << 19);
            int cr = pcr[i] - 128;
            int cb = pcb[i] - 128;
            int r = yFixed + cr * Cr2R;
            int g = yFixed + (cr * Cr2G) + ((cb * Cb2G) & unchecked((int)0xffff0000));
            int b = yFixed + cb * Cb2B;
            r >>= 20;
            g >>= 20;
            b >>= 20;
            if ((uint)r > 255)
            {
                r = r < 0 ? 0 : 255;
            }
            if ((uint)g > 255)
            {
                g = g < 0 ? 0 : 255;
            }
            if ((uint)b > 255)
            {
                b = b < 0 ? 0 : 255;
            }
            output[o] = (byte)r;
            output[o + 1] = (byte)g;
            output[o + 2] = (byte)b;
        }
    }

    /// <summary>stbi__blinn_8x8: fast rounded 0..255 x 0..255 -> 0..255 multiply.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte Blinn8x8(byte x, byte y)
    {
        uint t = (uint)(x * y) + 128;
        return (byte)((t + (t >> 8)) >> 8);
    }

    // ------------------------------------------------------------------------------------------
    // IDCT
    // ------------------------------------------------------------------------------------------

    // stbi__idct_simd constants: (even, odd) coefficient pairs for _mm_madd_epi16, built from stbi__f2f.
    private const short Rot00A = 2217, Rot00B = 2217 - 7567;           // f2f(0.5411961), f2f(0.5411961)+f2f(-1.847759065)
    private const short Rot01A = 2217 + 3135, Rot01B = 2217;           // f2f(0.5411961)+f2f(0.765366865), f2f(0.5411961)
    private const short Rot10A = 4816 - 3685, Rot10B = 4816;           // f2f(1.175875602)+f2f(-0.899976223), f2f(1.175875602)
    private const short Rot11A = 4816, Rot11B = 4816 - 10497;          // f2f(1.175875602), f2f(1.175875602)+f2f(-2.562915447)
    private const short Rot20A = -8034 + 1223, Rot20B = -8034;         // f2f(-1.961570560)+f2f(0.298631336), f2f(-1.961570560)
    private const short Rot21A = -8034, Rot21B = -8034 + 12586;        // f2f(-1.961570560), f2f(-1.961570560)+f2f(3.072711026)
    private const short Rot30A = -1597 + 8410, Rot30B = -1597;         // f2f(-0.390180644)+f2f(2.053119869), f2f(-0.390180644)
    private const short Rot31A = -1597, Rot31B = -1597 + 6149;         // f2f(-0.390180644), f2f(-0.390180644)+f2f(1.501321110)
    private const int ColumnBias = 512;
    private const int RowBias = 65536 + (128 << 17);

    /// <summary>
    /// IDCT of one dequantized 8x8 block into <paramref name="dst"/> at <paramref name="offset"/>
    /// (row stride <paramref name="stride"/>), with stbi__idct_simd semantics.
    /// </summary>
    private static unsafe void Idct(short[] block, byte[] dst, int offset, int stride)
    {
        if ((uint)offset > (uint)dst.Length || (uint)(offset + 7 * stride + 8) > (uint)dst.Length)
        {
            throw new InvalidDataException("Corrupt JPEG: block outside the component plane");
        }
        fixed (short* d = block)
        fixed (byte* o = dst)
        {
            if (Sse2.IsSupported)
            {
                IdctSse2(d, o + offset, stride);
            }
            else
            {
                IdctScalar(d, o + offset, stride);
            }
        }
    }

    /// <summary>
    /// Scalar emulation of stbi__idct_simd. It differs from the plain C stbi__idct_block only for
    /// out-of-range coefficients: sums of input pairs wrap at 16 bits (_mm_add_epi16) and the column
    /// pass output saturates to 16 bits (_mm_packs_epi32) before the row pass.
    /// </summary>
    internal static unsafe void IdctScalar(short* data, byte* output, int stride)
    {
        short* tmp = stackalloc short[64];
        int* o = stackalloc int[8];
        for (int c = 0; c < 8; ++c)
        {
            Idct1D(data[c], data[8 + c], data[16 + c], data[24 + c], data[32 + c], data[40 + c], data[48 + c], data[56 + c], ColumnBias, 10, o);
            for (int k = 0; k < 8; ++k)
            {
                int v = o[k];
                tmp[k * 8 + c] = (short)(v < short.MinValue ? short.MinValue : v > short.MaxValue ? short.MaxValue : v);
            }
        }
        for (int r = 0; r < 8; ++r)
        {
            short* s = tmp + r * 8;
            Idct1D(s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7], RowBias, 17, o);
            byte* dst = output + r * stride;
            for (int k = 0; k < 8; ++k)
            {
                int v = o[k];
                dst[k] = (byte)(v < 0 ? 0 : v > 255 ? 255 : v);
            }
        }
    }

    /// <summary>One 1-D pass of stbi__idct_simd's dct_pass (results before the final saturating pack).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe void Idct1D(int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7, int bias, int shift, int* o)
    {
        unchecked
        {
            // even part
            int t2e = s2 * Rot00A + s6 * Rot00B;
            int t3e = s2 * Rot01A + s6 * Rot01B;
            int sum04 = (short)(s0 + s4);
            int dif04 = (short)(s0 - s4);
            int t0e = sum04 << 12;
            int t1e = dif04 << 12;
            int x0 = t0e + t3e;
            int x3 = t0e - t3e;
            int x1 = t1e + t2e;
            int x2 = t1e - t2e;
            // odd part
            int y0o = s7 * Rot20A + s3 * Rot20B;
            int y2o = s7 * Rot21A + s3 * Rot21B;
            int y1o = s5 * Rot30A + s1 * Rot30B;
            int y3o = s5 * Rot31A + s1 * Rot31B;
            int sum17 = (short)(s1 + s7);
            int sum35 = (short)(s3 + s5);
            int y4o = sum17 * Rot10A + sum35 * Rot10B;
            int y5o = sum17 * Rot11A + sum35 * Rot11B;
            int x4 = y0o + y4o;
            int x5 = y1o + y5o;
            int x6 = y2o + y5o;
            int x7 = y3o + y4o;
            x0 += bias;
            x1 += bias;
            x2 += bias;
            x3 += bias;
            o[0] = (x0 + x7) >> shift;
            o[7] = (x0 - x7) >> shift;
            o[1] = (x1 + x6) >> shift;
            o[6] = (x1 - x6) >> shift;
            o[2] = (x2 + x5) >> shift;
            o[5] = (x2 - x5) >> shift;
            o[3] = (x3 + x4) >> shift;
            o[4] = (x3 - x4) >> shift;
        }
    }

    /// <summary>Direct port of stbi__idct_simd (SSE2) using .NET hardware intrinsics.</summary>
    internal static unsafe void IdctSse2(short* data, byte* output, int stride)
    {
        Vector128<short> row0 = Sse2.LoadVector128(data);
        Vector128<short> row1 = Sse2.LoadVector128(data + 8);
        Vector128<short> row2 = Sse2.LoadVector128(data + 16);
        Vector128<short> row3 = Sse2.LoadVector128(data + 24);
        Vector128<short> row4 = Sse2.LoadVector128(data + 32);
        Vector128<short> row5 = Sse2.LoadVector128(data + 40);
        Vector128<short> row6 = Sse2.LoadVector128(data + 48);
        Vector128<short> row7 = Sse2.LoadVector128(data + 56);

        // column pass
        DctPass(ref row0, ref row1, ref row2, ref row3, ref row4, ref row5, ref row6, ref row7, Vector128.Create(ColumnBias), 10);

        // 16-bit 8x8 transpose
        Interleave16(ref row0, ref row4);
        Interleave16(ref row1, ref row5);
        Interleave16(ref row2, ref row6);
        Interleave16(ref row3, ref row7);

        Interleave16(ref row0, ref row2);
        Interleave16(ref row1, ref row3);
        Interleave16(ref row4, ref row6);
        Interleave16(ref row5, ref row7);

        Interleave16(ref row0, ref row1);
        Interleave16(ref row2, ref row3);
        Interleave16(ref row4, ref row5);
        Interleave16(ref row6, ref row7);

        // row pass
        DctPass(ref row0, ref row1, ref row2, ref row3, ref row4, ref row5, ref row6, ref row7, Vector128.Create(RowBias), 17);

        // pack and 8-bit 8x8 transpose
        Vector128<byte> p0 = Sse2.PackUnsignedSaturate(row0, row1);
        Vector128<byte> p1 = Sse2.PackUnsignedSaturate(row2, row3);
        Vector128<byte> p2 = Sse2.PackUnsignedSaturate(row4, row5);
        Vector128<byte> p3 = Sse2.PackUnsignedSaturate(row6, row7);

        Interleave8(ref p0, ref p2);
        Interleave8(ref p1, ref p3);

        Interleave8(ref p0, ref p1);
        Interleave8(ref p2, ref p3);

        Interleave8(ref p0, ref p2);
        Interleave8(ref p1, ref p3);

        *(ulong*)output = p0.AsUInt64().GetElement(0);
        output += stride;
        *(ulong*)output = p0.AsUInt64().GetElement(1);
        output += stride;
        *(ulong*)output = p2.AsUInt64().GetElement(0);
        output += stride;
        *(ulong*)output = p2.AsUInt64().GetElement(1);
        output += stride;
        *(ulong*)output = p1.AsUInt64().GetElement(0);
        output += stride;
        *(ulong*)output = p1.AsUInt64().GetElement(1);
        output += stride;
        *(ulong*)output = p3.AsUInt64().GetElement(0);
        output += stride;
        *(ulong*)output = p3.AsUInt64().GetElement(1);
    }

    /// <summary>dct_interleave16.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Interleave16(ref Vector128<short> a, ref Vector128<short> b)
    {
        Vector128<short> tmp = a;
        a = Sse2.UnpackLow(a, b);
        b = Sse2.UnpackHigh(tmp, b);
    }

    /// <summary>dct_interleave8.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void Interleave8(ref Vector128<byte> a, ref Vector128<byte> b)
    {
        Vector128<byte> tmp = a;
        a = Sse2.UnpackLow(a, b);
        b = Sse2.UnpackHigh(tmp, b);
    }

    /// <summary>dct_rot: out0 = x*c0.even + y*c0.odd, out1 = x*c1.even + y*c1.odd (32-bit, low/high halves).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DctRot(Vector128<short> x, Vector128<short> y, Vector128<short> c0, Vector128<short> c1,
        out Vector128<int> out0L, out Vector128<int> out0H, out Vector128<int> out1L, out Vector128<int> out1H)
    {
        Vector128<short> lo = Sse2.UnpackLow(x, y);
        Vector128<short> hi = Sse2.UnpackHigh(x, y);
        out0L = Sse2.MultiplyAddAdjacent(lo, c0);
        out0H = Sse2.MultiplyAddAdjacent(hi, c0);
        out1L = Sse2.MultiplyAddAdjacent(lo, c1);
        out1H = Sse2.MultiplyAddAdjacent(hi, c1);
    }

    /// <summary>dct_widen: 16-bit lanes to 32-bit, scaled by 4096.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DctWiden(Vector128<short> v, out Vector128<int> lo, out Vector128<int> hi)
    {
        lo = Sse2.ShiftRightArithmetic(Sse2.UnpackLow(Vector128<short>.Zero, v).AsInt32(), 4);
        hi = Sse2.ShiftRightArithmetic(Sse2.UnpackHigh(Vector128<short>.Zero, v).AsInt32(), 4);
    }

    /// <summary>dct_bfly32o: butterfly a/b, add bias, shift and pack with signed saturation.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DctBfly(out Vector128<short> out0, out Vector128<short> out1,
        Vector128<int> aL, Vector128<int> aH, Vector128<int> bL, Vector128<int> bH, Vector128<int> bias, [ConstantExpected] byte shift)
    {
        Vector128<int> abL = Sse2.Add(aL, bias);
        Vector128<int> abH = Sse2.Add(aH, bias);
        Vector128<int> sumL = Sse2.Add(abL, bL);
        Vector128<int> sumH = Sse2.Add(abH, bH);
        Vector128<int> difL = Sse2.Subtract(abL, bL);
        Vector128<int> difH = Sse2.Subtract(abH, bH);
        out0 = Sse2.PackSignedSaturate(Sse2.ShiftRightArithmetic(sumL, shift), Sse2.ShiftRightArithmetic(sumH, shift));
        out1 = Sse2.PackSignedSaturate(Sse2.ShiftRightArithmetic(difL, shift), Sse2.ShiftRightArithmetic(difH, shift));
    }

    /// <summary>dct_pass: one 1-D IDCT pass over eight 8-lane rows.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void DctPass(ref Vector128<short> row0, ref Vector128<short> row1, ref Vector128<short> row2, ref Vector128<short> row3,
        ref Vector128<short> row4, ref Vector128<short> row5, ref Vector128<short> row6, ref Vector128<short> row7, Vector128<int> bias, [ConstantExpected] byte shift)
    {
        Vector128<short> rot00 = Pair(Rot00A, Rot00B), rot01 = Pair(Rot01A, Rot01B);
        Vector128<short> rot10 = Pair(Rot10A, Rot10B), rot11 = Pair(Rot11A, Rot11B);
        Vector128<short> rot20 = Pair(Rot20A, Rot20B), rot21 = Pair(Rot21A, Rot21B);
        Vector128<short> rot30 = Pair(Rot30A, Rot30B), rot31 = Pair(Rot31A, Rot31B);

        // even part
        DctRot(row2, row6, rot00, rot01, out var t2eL, out var t2eH, out var t3eL, out var t3eH);
        Vector128<short> sum04 = Sse2.Add(row0, row4);
        Vector128<short> dif04 = Sse2.Subtract(row0, row4);
        DctWiden(sum04, out var t0eL, out var t0eH);
        DctWiden(dif04, out var t1eL, out var t1eH);
        Vector128<int> x0L = Sse2.Add(t0eL, t3eL), x0H = Sse2.Add(t0eH, t3eH);
        Vector128<int> x3L = Sse2.Subtract(t0eL, t3eL), x3H = Sse2.Subtract(t0eH, t3eH);
        Vector128<int> x1L = Sse2.Add(t1eL, t2eL), x1H = Sse2.Add(t1eH, t2eH);
        Vector128<int> x2L = Sse2.Subtract(t1eL, t2eL), x2H = Sse2.Subtract(t1eH, t2eH);

        // odd part
        DctRot(row7, row3, rot20, rot21, out var y0oL, out var y0oH, out var y2oL, out var y2oH);
        DctRot(row5, row1, rot30, rot31, out var y1oL, out var y1oH, out var y3oL, out var y3oH);
        Vector128<short> sum17 = Sse2.Add(row1, row7);
        Vector128<short> sum35 = Sse2.Add(row3, row5);
        DctRot(sum17, sum35, rot10, rot11, out var y4oL, out var y4oH, out var y5oL, out var y5oH);
        Vector128<int> x4L = Sse2.Add(y0oL, y4oL), x4H = Sse2.Add(y0oH, y4oH);
        Vector128<int> x5L = Sse2.Add(y1oL, y5oL), x5H = Sse2.Add(y1oH, y5oH);
        Vector128<int> x6L = Sse2.Add(y2oL, y5oL), x6H = Sse2.Add(y2oH, y5oH);
        Vector128<int> x7L = Sse2.Add(y3oL, y4oL), x7H = Sse2.Add(y3oH, y4oH);

        DctBfly(out row0, out row7, x0L, x0H, x7L, x7H, bias, shift);
        DctBfly(out row1, out row6, x1L, x1H, x6L, x6H, bias, shift);
        DctBfly(out row2, out row5, x2L, x2H, x5L, x5H, bias, shift);
        DctBfly(out row3, out row4, x3L, x3H, x4L, x4H, bias, shift);
    }

    /// <summary>dct_const: (x, y) repeated four times.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector128<short> Pair(short x, short y) => Vector128.Create(x, y, x, y, x, y, x, y);
}
