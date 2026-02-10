using Microsoft.ML.OnnxRuntime.Tensors;

namespace SentenceTransformers.Qwen3;

/// <summary>
/// Helpers for converting ONNX output tensors to float[][] and normalizing embeddings.
/// </summary>
public static class DenseTensorHelpers
{
    /// <summary>
    /// Copies a 2D float tensor [batch, embDim] to a jagged array.
    /// </summary>
    public static float[][] CopyToJagged(DenseTensor<float> tensor)
    {
        var dims = tensor.Dimensions.ToArray();
        int batch = dims[0];
        int embDim = dims[1];

        var result = new float[batch][];
        for (int i = 0; i < batch; i++)
        {
            var row = new float[embDim];
            for (int j = 0; j < embDim; j++)
            {
                row[j] = tensor[i, j];
            }
            result[i] = row;
        }
        return result;
    }

    /// <summary>
    /// Dequantizes a 2D uint8 tensor [batch, embDim] to float[][] using scale and zero point.
    /// </summary>
    public static float[][] DequantizeToJagged(DenseTensor<byte> tensor, float scale, int zeroPoint)
    {
        var dims = tensor.Dimensions.ToArray();
        int batch = dims[0];
        int embDim = dims[1];

        var result = new float[batch][];
        for (int i = 0; i < batch; i++)
        {
            var row = new float[embDim];
            for (int j = 0; j < embDim; j++)
            {
                row[j] = (tensor[i, j] - zeroPoint) * scale;
            }
            result[i] = row;
        }
        return result;
    }

    /// <summary>
    /// Dequantizes a 2D int8 tensor [batch, embDim] to float[][] using scale and zero point.
    /// </summary>
    public static float[][] DequantizeToJagged(DenseTensor<sbyte> tensor, float scale, int zeroPoint)
    {
        var dims = tensor.Dimensions.ToArray();
        int batch = dims[0];
        int embDim = dims[1];

        var result = new float[batch][];
        for (int i = 0; i < batch; i++)
        {
            var row = new float[embDim];
            for (int j = 0; j < embDim; j++)
            {
                row[j] = (unchecked((byte)tensor[i, j]) - zeroPoint) * scale;
            }
            result[i] = row;
        }
        return result;
    }

    /// <summary>
    /// Normalizes each row in-place to unit L2 norm.
    /// </summary>
    public static void NormalizeRows(float[][] rows, float eps = 1e-12f)
    {
        for (int r = 0; r < rows.Length; r++)
        {
            var v = rows[r];
            float sumSq = 0f;
            for (int i = 0; i < v.Length; i++)
            {
                sumSq += v[i] * v[i];
            }
            float norm = MathF.Max(MathF.Sqrt(sumSq), eps);
            float inv = 1f / norm;
            for (int i = 0; i < v.Length; i++)
            {
                v[i] *= inv;
            }
        }
    }
}
