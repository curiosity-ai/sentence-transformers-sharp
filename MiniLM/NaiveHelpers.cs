using Microsoft.ML.OnnxRuntime.Tensors;

namespace MiniLM;

public static class NaiveHelpers
{

    public static DenseTensor<float> Normalize(DenseTensor<float> input_dense, float eps = 1e-12f)
    {
        var sentencesCount = input_dense.Dimensions[0];
        var hiddenStates = input_dense.Dimensions[1];

        var denom_dense = new float [sentencesCount];
//            sum(abs(x)**2)**(1./2)

        for (int s = 0; s < sentencesCount; s++)
        {
            for (int i = 0; i < hiddenStates; i++)
            {
                denom_dense[s] += input_dense[s, i] * input_dense[s, i];
            }
            denom_dense[s] = Math.Max((float)Math.Sqrt(denom_dense[s]), eps);

        }


        for (int s = 0; s < sentencesCount; s++)
        {
            for (int i = 0; i < hiddenStates; i++)
            {
                input_dense[s, i] /= denom_dense[s];
            }

        }

        return input_dense;
    }


    public static DenseTensor<float> MeanPooling(DenseTensor<float> token_embeddings_dense, DenseTensor<float> attentionMask_dense, float eps = 1e-9f)
    {
        var sentencesCount = token_embeddings_dense.Dimensions[0];
        var sentenceLength = token_embeddings_dense.Dimensions[1];
        var hiddenStates = token_embeddings_dense.Dimensions[2];

        var result = new DenseTensor<float>(new[] { sentencesCount, hiddenStates });

        for (int s = 0; s < sentencesCount; s++)
        {
            var maskSum = 0f;

            for (int t = 0; t < sentenceLength; t++)
            {
                maskSum += attentionMask_dense[s, t];

                for (int i = 0; i < hiddenStates; i++)
                {
                    result[s, i] += token_embeddings_dense[s, t, i] * attentionMask_dense[s, t];
                }
            }

            for (int i = 0; i < hiddenStates; i++)
            {
                result[s, i] /= Math.Max(maskSum, eps);
            }
        }


        return result;
    }
}