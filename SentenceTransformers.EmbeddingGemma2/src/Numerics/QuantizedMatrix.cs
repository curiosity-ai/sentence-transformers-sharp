using SentenceTransformers.EmbeddingGemma2.LiteRt;

namespace SentenceTransformers.EmbeddingGemma2.Numerics;

/// <summary>
/// A per-output-channel symmetric integer weight matrix <c>W[Rows, Cols]</c> (TFLite <c>FULLY_CONNECTED</c>
/// layout: one row per output feature), stored as one signed byte per weight with rows padded to a
/// multiple of <see cref="Alignment"/> so SIMD kernels never need a K tail. The real weight is
/// <c>Data[r, k] * Scale[r]</c>. INT4/INT2 weights are widened to bytes once at load time; the
/// <see cref="SmallRange"/> flag records that every value fits in [-8, 7], which lets the x86 kernel use
/// the faster (otherwise saturating) <c>pmaddubsw</c> instruction.
/// </summary>
internal sealed class QuantizedMatrix
{
    public const int Alignment = 64;

    public int Rows { get; }
    public int Cols { get; }
    /// <summary>Row stride in bytes (Cols rounded up to <see cref="Alignment"/>).</summary>
    public int Stride { get; }
    public sbyte[] Data { get; }
    public float[] Scale { get; }
    /// <summary>Σ_k Data[r, k] - used to fold the activation zero point out of the integer dot product.</summary>
    public int[] RowSum { get; }
    public float[] Bias { get; set; }
    public bool SmallRange { get; }

    public QuantizedMatrix(int rows, int cols, sbyte[] data, float[] scale)
    {
        Rows = rows;
        Cols = cols;
        Stride = (cols + Alignment - 1) / Alignment * Alignment;
        Scale = scale;
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
        RowSum = new int[rows];
        bool small = true;
        for (int r = 0; r < rows; r++)
        {
            int s = 0;
            var row = Data.AsSpan(r * Stride, cols);
            foreach (var v in row)
            {
                s += v;
                small &= v >= -8 && v <= 7;
            }
            RowSum[r] = s;
        }
        SmallRange = small;
    }

    public ReadOnlySpan<sbyte> Row(int r) => Data.AsSpan(r * Stride, Stride);

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
        var row = Data.AsSpan(r * Stride, Cols);
        for (int k = 0; k < Cols; k++)
        {
            f[k] = row[k] * Scale[r];
        }
        return f;
    }

    public long ByteSize => Data.LongLength + Scale.LongLength * 4 + RowSum.LongLength * 4;
}
