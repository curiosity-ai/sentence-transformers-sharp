/*
 * This file defines the DenseTensorHelpers class, which provides helper methods for working with dense tensors.
 * It includes methods for normalizing dense tensors and performing mean pooling using attention masks.
 */

using Microsoft.ML.OnnxRuntime.Tensors; // Importing the required namespace

namespace SentenceTransformers.MiniLM
{
    // Provides helper methods for working with dense tensors.
    public static class DenseTensorHelpers
    {
        // Normalizes the input dense tensor along the specified axis.
        // Returns the normalized dense tensor.
        public static DenseTensor<float> Normalize(DenseTensor<float> inputTensor, float epsilon = 1e-12f)
        {
            // Computes sum(abs(x)^2)^(1/2)

            var sentenceCount = inputTensor.Dimensions[0]; // Number of sentences
            var hiddenStates = inputTensor.Dimensions[1]; // Number of hidden states

            var denominators = new float[sentenceCount]; // Array to store normalization denominators

            // Compute normalization denominators for each sentence
            for (int sentenceIndex = 0; sentenceIndex < sentenceCount; sentenceIndex++)
            {
                for (int stateIndex = 0; stateIndex < hiddenStates; stateIndex++)
                {
                    denominators[sentenceIndex] += inputTensor[sentenceIndex, stateIndex] * inputTensor[sentenceIndex, stateIndex];
                }

                // Compute square root and apply epsilon
                denominators[sentenceIndex] = MathF.Max(MathF.Sqrt(denominators[sentenceIndex]), epsilon);
            }

            // Normalize the input dense tensor
            for (int sentenceIndex = 0; sentenceIndex < sentenceCount; sentenceIndex++)
            {
                var inverseNorm = 1 / denominators[sentenceIndex]; // Compute the inverse normalization factor

                for (int stateIndex = 0; stateIndex < hiddenStates; stateIndex++)
                {
                    inputTensor[sentenceIndex, stateIndex] *= inverseNorm; // Normalize each element of the input tensor
                }
            }

            return inputTensor; // Return the normalized dense tensor
        }

        // Computes mean pooling over the input tensor using the attention masks.
        // Returns the result tensor after mean pooling.
        public static DenseTensor<float> MeanPooling(DenseTensor<float> tokenEmbeddingsTensor, List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> encodedSentences, float epsilon = 1e-9f)
        {
            var sentenceCount = tokenEmbeddingsTensor.Dimensions[0]; // Number of sentences
            var sentenceLength = tokenEmbeddingsTensor.Dimensions[1]; // Length of each sentence
            var hiddenStates = tokenEmbeddingsTensor.Dimensions[2]; // Number of hidden states

            var resultTensor = new DenseTensor<float>(new[] { sentenceCount, hiddenStates }); // Result tensor

            // Iterate through each sentence
            for (int sentenceIndex = 0; sentenceIndex < sentenceCount; sentenceIndex++)
            {
                var maskSum = 0f; // Sum of attention mask values

                var attentionMask = encodedSentences[sentenceIndex].AttentionMask; // Get attention mask for the current sentence

                // Iterate through each token in the sentence
                for (int tokenIndex = 0; tokenIndex < sentenceLength; tokenIndex++)
                {
                    maskSum += attentionMask[tokenIndex]; // Update mask sum

                    // Apply attention mask to token embeddings
                    for (int stateIndex = 0; stateIndex < hiddenStates; stateIndex++)
                    {
                        resultTensor[sentenceIndex, stateIndex] += tokenEmbeddingsTensor[sentenceIndex, tokenIndex, stateIndex] * attentionMask[tokenIndex];
                    }
                }

                var inverseSum = 1f / MathF.Max(maskSum, epsilon); // Compute inverse sum

                // Normalize the result by the sum of attention masks
                for (int stateIndex = 0; stateIndex < hiddenStates; stateIndex++)
                {
                    resultTensor[sentenceIndex, stateIndex] *= inverseSum;
                }
            }

            return resultTensor; // Return the result tensor after mean pooling
        }
    }
}
