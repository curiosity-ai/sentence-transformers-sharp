using Microsoft.ML.OnnxRuntime.Tensors;

namespace MiniLM;

public static class DenseTensorHelpers
{
    public static DenseTensor<float> Normalize(DenseTensor<float> input_dense, float eps = 1e-12f)
    {
        //Computes sum(abs(x)^2)^(1/2)

        var sentencesCount = input_dense.Dimensions[0];
        var hiddenStates   = input_dense.Dimensions[1];

        var denom_dense = new float [sentencesCount];

        for (int s = 0; s < sentencesCount; s++)
        {
            for (int i = 0; i < hiddenStates; i++)
            {
                denom_dense[s] += input_dense[s, i] * input_dense[s, i];
            }

            denom_dense[s] = MathF.Max(MathF.Sqrt(denom_dense[s]), eps);
        }

        for (int s = 0; s < sentencesCount; s++)
        {
            var invNorm = 1 / denom_dense[s];

            for (int i = 0; i < hiddenStates; i++)
            {
                input_dense[s, i] *= invNorm;
            }
        }

        return input_dense;
    }


    public static DenseTensor<float> MeanPooling(DenseTensor<float> token_embeddings_dense, List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> encodedSentences, float eps = 1e-9f)
    {
        var sentencesCount = token_embeddings_dense.Dimensions[0];
        var sentenceLength = token_embeddings_dense.Dimensions[1];
        var hiddenStates   = token_embeddings_dense.Dimensions[2];

        var result = new DenseTensor<float>(new[] { sentencesCount, hiddenStates });

        for (int s = 0; s < sentencesCount; s++)
        {
            var maskSum = 0f;

            var attentionMask = encodedSentences[s].AttentionMask;

            for (int t = 0; t < sentenceLength; t++)
            {
                maskSum += attentionMask[t];

                for (int i = 0; i < hiddenStates; i++)
                {
                    result[s, i] += token_embeddings_dense[s, t, i] * attentionMask[t];
                }
            }

            var invSum = 1f / MathF.Max(maskSum, eps);

            for (int i = 0; i < hiddenStates; i++)
            {
                result[s, i] *= invSum;
            }
        }

        return result;
    }
}