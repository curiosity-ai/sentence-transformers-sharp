using System.Runtime.Intrinsics;
using SentenceTransformers.EmbeddingGemma2.LiteRt;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>
/// A per-output-channel symmetric integer weight matrix <c>W[Rows, Cols]</c> (TFLite <c>FULLY_CONNECTED</c>
/// layout: one row per output feature) with rows padded to a multiple of <see cref="Alignment"/> so SIMD kernels
/// never need a K tail. The real weight is <c>w[r, k] · Scale[r]</c>.
/// <para>
/// Weights are kept at their native width: 8-bit matrices as one signed byte per weight in row order
/// (<see cref="Data"/>), INT4-range ones as two per byte and INT2-range ones as four per byte (<see cref="Packed"/>),
/// so the 740M bundle's 2-bit audio encoder streams a quarter of the bytes per chunk.
/// </para>
/// <para>
/// Packed matrices are stored in the order the broadcast GEMM kernels read them: panels of
/// <see cref="PanelWidth"/> output channels, each laid out <c>[Stride/4][32 channels][4 consecutive k]</c>
/// (zero padded), then bit-packed per 64-byte group <c>g</c>: 4-bit: byte <c>i</c> (0..31) holds <c>v[g+i]+8</c>
/// in its low nibble and <c>v[g+32+i]+8</c> in its high nibble; 2-bit: byte <c>i</c> (0..15) holds
/// <c>v[g+16p+i]+2</c> in bits <c>2p..2p+1</c> for planes <c>p</c> = 0..3. <see cref="UnpackPanels"/> expands
/// panels right before use; <see cref="UnpackRows"/> rebuilds row order for the other kernels.
/// </para>
/// </summary>
internal sealed class QuantizedMatrix
{
    public const int Alignment = 64;
    /// <summary>Output channels per panel of the packed (broadcast-kernel) layout.</summary>
    public const int PanelWidth = 32;

    public int Rows { get; }
    public int Cols { get; }
    /// <summary>Row stride in weights (Cols rounded up to <see cref="Alignment"/>).</summary>
    public int Stride { get; }
    /// <summary>Storage width: 8, 4 or 2 bits per weight.</summary>
    public int Bits { get; }
    /// <summary>One signed byte per weight (8-bit matrices only, otherwise null).</summary>
    public sbyte[] Data { get; }
    /// <summary>Packed weights (4- and 2-bit matrices only, otherwise null); <see cref="Stride"/> · Bits / 8 bytes per row.</summary>
    public byte[] Packed { get; }
    public float[] Scale { get; }
    /// <summary>Σ_k w[r, k] - used to fold the activation zero point out of the integer dot product.</summary>
    public int[] RowSum { get; }
    public float[] Bias { get; set; }
    /// <summary>Every weight fits in [-8, 7], which lets the x86 kernel use the faster (otherwise saturating)
    /// <c>pmaddubsw</c> instruction.</summary>
    public bool SmallRange { get; }

    /// <summary>Number of <see cref="PanelWidth"/>-channel panels (the last one zero padded).</summary>
    public int Panels => (Rows + PanelWidth - 1) / PanelWidth;
    /// <summary>Bytes of one unpacked panel: <c>Stride · PanelWidth</c>.</summary>
    public int PanelBytes => Stride * PanelWidth;
    private int PackedPanelBytes => PanelBytes * Bits / 8;

    public QuantizedMatrix(int rows, int cols, sbyte[] data, float[] scale, bool packLowBits = true)
    {
        Rows = rows;
        Cols = cols;
        Stride = (cols + Alignment - 1) / Alignment * Alignment;
        Scale = scale;
        RowSum = new int[rows];
        int min = 0, max = 0;
        for (int r = 0; r < rows; r++)
        {
            int s = 0;
            foreach (var v in data.AsSpan(r * cols, cols))
            {
                s += v;
                min = Math.Min(min, v);
                max = Math.Max(max, v);
            }
            RowSum[r] = s;
        }
        SmallRange = min >= -8 && max <= 7;
        Bits = !packLowBits ? 8 : min >= -2 && max <= 1 ? 2 : SmallRange ? 4 : 8;
        if (Bits == 8)
        {
            if (Stride == cols && data.Length == rows * cols)
            {
                Data = data;
            }
            else
            {
                Data = new sbyte[rows * Stride];
                for (int r = 0; r < rows; r++)
                {
                    data.AsSpan(r * cols, cols).CopyTo(Data.AsSpan(r * Stride, cols));
                }
            }
            return;
        }
        Packed = new byte[(long)Panels * PackedPanelBytes];
        var panel = new sbyte[PanelBytes];
        for (int p = 0; p < Panels; p++)
        {
            panel.AsSpan().Clear();
            for (int c = 0; c < PanelWidth && p * PanelWidth + c < rows; c++)
            {
                var row = data.AsSpan((p * PanelWidth + c) * cols, cols);
                for (int k = 0; k < cols; k++)
                {
                    panel[(k / 4 * PanelWidth + c) * 4 + (k & 3)] = row[k];
                }
            }
            var dst = Packed.AsSpan(p * PackedPanelBytes, PackedPanelBytes);
            for (int g = 0; g < PanelBytes; g += 64)
            {
                if (Bits == 4)
                {
                    for (int i = 0; i < 32; i++)
                    {
                        dst[g / 2 + i] = (byte)((panel[g + i] + 8) | ((panel[g + 32 + i] + 8) << 4));
                    }
                }
                else
                {
                    for (int i = 0; i < 16; i++)
                    {
                        dst[g / 4 + i] = (byte)((panel[g + i] + 2) | ((panel[g + 16 + i] + 2) << 2) | ((panel[g + 32 + i] + 2) << 4) | ((panel[g + 48 + i] + 2) << 6));
                    }
                }
            }
        }
    }

    /// <summary>Expands panels <paramref name="p0"/>..<paramref name="p1"/> of a packed matrix to one signed byte per
    /// weight at <paramref name="dst"/> (<see cref="PanelBytes"/> each, in panel layout).</summary>
    public unsafe void UnpackPanels(int p0, int p1, sbyte* dst)
    {
        fixed (byte* packed = Packed)
        {
            byte* src = packed + (long)p0 * PackedPanelBytes;
            long groups = (long)(p1 - p0) * PanelBytes / 64;
            byte* d = (byte*)dst;
            if (Bits == 4)
            {
                var mask = Vector256.Create((byte)0x0F);
                var bias = Vector256.Create((byte)8);
                for (long gi = 0; gi < groups; gi++, src += 32, d += 64)
                {
                    var v = Vector256.Load(src);
                    ((v & mask) - bias).Store(d);
                    ((Vector256.ShiftRightLogical(v, 4) & mask) - bias).Store(d + 32);
                }
            }
            else
            {
                var mask = Vector128.Create((byte)0x03);
                var bias = Vector128.Create((byte)2);
                for (long gi = 0; gi < groups; gi++, src += 16, d += 64)
                {
                    var v = Vector128.Load(src);
                    ((v & mask) - bias).Store(d);
                    ((Vector128.ShiftRightLogical(v, 2) & mask) - bias).Store(d + 16);
                    ((Vector128.ShiftRightLogical(v, 4) & mask) - bias).Store(d + 32);
                    ((Vector128.ShiftRightLogical(v, 6) & mask) - bias).Store(d + 48);
                }
            }
        }
    }

    /// <summary>Rows <paramref name="r0"/>..<paramref name="r1"/> as one signed byte per weight in row order at
    /// <paramref name="dst"/> (row stride <see cref="Stride"/>).</summary>
    public unsafe void UnpackRows(int r0, int r1, sbyte* dst)
    {
        if (Bits == 8)
        {
            fixed (sbyte* src = Data)
            {
                Buffer.MemoryCopy(src + (long)r0 * Stride, dst, (long)(r1 - r0) * Stride, (long)(r1 - r0) * Stride);
            }
            return;
        }
        var panel = new sbyte[PanelBytes];
        fixed (sbyte* pp = panel)
        {
            for (int p = r0 / PanelWidth; p * PanelWidth < r1; p++)
            {
                UnpackPanels(p, p + 1, pp);
                for (int c = 0; c < PanelWidth; c++)
                {
                    int r = p * PanelWidth + c;
                    if (r < r0 || r >= r1)
                    {
                        continue;
                    }
                    int* row = (int*)(dst + (long)(r - r0) * Stride);
                    int* src = (int*)pp + c;
                    for (int k4 = 0; k4 < Stride / 4; k4++)
                    {
                        row[k4] = src[k4 * PanelWidth];
                    }
                }
            }
        }
    }

    /// <summary>Row <paramref name="r"/> as one signed byte per weight (<see cref="Stride"/> long).</summary>
    public unsafe sbyte[] Row(int r)
    {
        var row = new sbyte[Stride];
        fixed (sbyte* d = row)
        {
            UnpackRows(r, r + 1, d);
        }
        return row;
    }

    /// <summary>Builds a matrix from a quantized TFLite constant (2-D, per-row or per-tensor scale, zero point 0).</summary>
    public static QuantizedMatrix FromTensor(TfLiteModel model, TfLiteTensor tensor)
    {
        if (tensor.Shape.Length != 2)
        {
            throw new InvalidDataException($"Expected a 2-D weight, got {tensor}.");
        }
        var q = tensor.Quantization ?? throw new InvalidDataException($"{tensor} is not quantized.");
        int rows = tensor.Shape[0], cols = tensor.Shape[1];
        if (q.QuantizedDimension != 0 && q.Scale.Length != 1)
        {
            throw new NotSupportedException($"{tensor}: only per-output-channel (axis 0) quantization is supported.");
        }
        foreach (var zp in q.ZeroPoint)
        {
            if (zp != 0)
            {
                throw new NotSupportedException($"{tensor}: asymmetric weight quantization is not supported.");
            }
        }
        var scale = q.Scale.Length == rows ? q.Scale : Enumerable.Repeat(q.Scale[0], rows).ToArray();
        return new QuantizedMatrix(rows, cols, model.ReadQuantizedValues(tensor), scale);
    }

    /// <summary>Dequantized float copy of row <paramref name="r"/> (used by tests / reference paths).</summary>
    public float[] DequantizeRow(int r)
    {
        var f = new float[Cols];
        var row = Row(r);
        for (int k = 0; k < Cols; k++)
        {
            f[k] = row[k] * Scale[r];
        }
        return f;
    }

    public long ByteSize => (Data?.LongLength ?? 0) + (Packed?.LongLength ?? 0) + Scale.LongLength * 4 + RowSum.LongLength * 4;
}
