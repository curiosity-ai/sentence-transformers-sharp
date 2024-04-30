using Microsoft.ML.OnnxRuntime.Tensors;

namespace SentenceTransformers.ArcticXs;

public static class DenseTensorHelpers
{
    public static float[][] Normalize(DenseTensor<float> input_dense, float eps = 1e-12f)
    {
        //Computes sum(abs(x)^2)^(1/2)

        const int tokenIndexForEncoding = 0;

        var sentencesCount = input_dense.Dimensions[0];
        var hiddenStates   = input_dense.Dimensions[2];

        var denom_dense = new float [sentencesCount];

        for (int s = 0; s < sentencesCount; s++)
        {
            for (int i = 0; i < hiddenStates; i++)
            {
                denom_dense[s] += input_dense[s, tokenIndexForEncoding, i] * input_dense[s, tokenIndexForEncoding, i];
            }
            denom_dense[s] = MathF.Max(MathF.Sqrt(denom_dense[s]), eps);
        }

        var outputFlatten = new float[sentencesCount][];

        for (int s = 0; s < sentencesCount; s++)
        {
            var invNorm = 1 / denom_dense[s];

            var emb = new float[hiddenStates];
            outputFlatten[s] = emb;

            for (int i = 0; i < hiddenStates; i++)
            {
                emb[i] = input_dense[s, tokenIndexForEncoding, i] * invNorm;
            }
        }

        return outputFlatten;
    }
}