/*
 * This file defines the DenseTensorHelpers class, which provides helper methods for working with dense tensors.
 * The class contains a static method Normalize, which normalizes the input dense tensor along a specified axis.
 * The Normalize method computes the normalization denominators for each sentence and returns a jagged array of normalized vectors.
 * These normalized vectors can be used for various natural language processing tasks, such as text encoding or similarity calculation.
 */

using Microsoft.ML.OnnxRuntime.Tensors;

namespace SentenceTransformers.ArcticXs
{
    // Provides helper methods for working with dense tensors.
    public static class DenseTensorHelpers
    {
        // Normalizes the input dense tensor along the specified axis.
        // Returns a jagged array of normalized vectors.
        public static float[][] Normalize(DenseTensor<float> inputTensor, float epsilon = 1e-12f)
        {
            // Computes the L2 norm of each vector in the dense tensor.

            const int tokenIndex = 0; // Index for the token to be encoded

            // Get the dimensions of the input dense tensor
            var sentenceCount = inputTensor.Dimensions[0]; // Number of sentences
            var hiddenStates = inputTensor.Dimensions[2]; // Number of hidden states

            // Array to store the normalization denominators for each sentence
            var norms = new float[sentenceCount];

            // Compute the normalization denominators for each sentence
            for (int s = 0; s < sentenceCount; s++)
            {
                for (int i = 0; i < hiddenStates; i++)
                {
                    norms[s] += inputTensor[s, tokenIndex, i] * inputTensor[s, tokenIndex, i];
                }
                norms[s] = MathF.Max(MathF.Sqrt(norms[s]), epsilon);
            }

            // Array to store the output normalized vectors
            var normalizedVectors = new float[sentenceCount][];

            // Normalize the input tensor and store the normalized vectors
            for (int s = 0; s < sentenceCount; s++)
            {
                var invNorm = 1 / norms[s]; // Compute the inverse normalization factor

                // Array to store the normalized vector for the current sentence
                var normalizedVector = new float[hiddenStates];
                normalizedVectors[s] = normalizedVector;

                // Normalize each element of the input tensor for the current sentence
                for (int i = 0; i < hiddenStates; i++)
                {
                    normalizedVector[i] = inputTensor[s, tokenIndex, i] * invNorm;
                }
            }

            return normalizedVectors; // Return the normalized vectors
        }
    }
}
