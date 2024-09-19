/*
 * This file contains the implementation of a MiniLM-based sentence encoder. 
 * The MiniLMSentenceEncoder class is responsible for encoding input sentences into vectors using a pre-trained MiniLM model.
 * It utilizes an ONNX model for inference and a MiniLMTokenizer for tokenization.
 * The main method, Encode(), tokenizes the input sentences, prepares them for inference, executes the model, and performs mean pooling and normalization on the output tensor.
 * The resulting vectors represent the encoded embeddings of the input sentences.
 */

using BERTTokenizers.Base; // Importing the required namespaces
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static SentenceTransformers.MiniLM.DenseTensorHelpers; // Importing static class for tensor operations
using SentenceTransformers;

namespace SentenceTransformers.MiniLM
{
    // Represents a sentence encoder model based on MiniLM architecture.
    public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
    {
        private readonly SessionOptions   _sessionOptions; // Session options for ONNX Runtime
        private readonly InferenceSession _session;        // Inference session for the model
        private readonly TokenizerBase    _tokenizer;      // Tokenizer for tokenizing input sentences
        private readonly string[]         _outputNames;    // Names of the model's output nodes

        // Constructor for the SentenceEncoder class
        public SentenceEncoder(SessionOptions sessionOptions = null)
        {
            // Initialize session options and load the inference session with the ONNX model
            _sessionOptions = sessionOptions ?? new SessionOptions();
            _session        = new InferenceSession(ResourceLoader.GetResource(typeof(SentenceEncoder).Assembly, "model.onnx"), _sessionOptions);
            
            // Initialize tokenizer with MiniLMTokenizer
            _tokenizer      = new MiniLMTokenizer();
            
            // Get the names of the output nodes of the model
            _outputNames    = _session.OutputMetadata.Keys.ToArray();
        }

        // Dispose method to release resources
        public void Dispose()
        {
            _sessionOptions.Dispose(); // Dispose session options
            _session.Dispose();        // Dispose inference session
        }

        // Encodes an array of sentences into vectors
        public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default)
        {
            var numSentences = sentences.Length; // Number of sentences to encode

            // Tokenize the input sentences and flatten the tokens and attention masks
            var encoded    = _tokenizer.Encode(sentences); // Tokenize the input sentences
            var tokenCount = encoded.First().InputIds.Length; // Number of tokens in each sentence

            // Flatten input tokens, attention masks, and token type IDs
            long[] flattenIDs           = new long[encoded.Sum(s => s.InputIds.Length)]; // Flatten input tokens
            long[] flattenAttentionMask = new long[encoded.Sum(s => s.AttentionMask.Length)]; // Flatten attention masks
            long[] flattenTokenTypeIds  = new long[encoded.Sum(s => s.TokenTypeIds.Length)]; // Flatten token type IDs

            // Initialize spans for copying encoded tokens, attention masks, and token type IDs
            var flattenIDsSpan           = flattenIDs.AsSpan();
            var flattenAttentionMaskSpan = flattenAttentionMask.AsSpan();
            var flattenTokenTypeIdsSpan  = flattenTokenTypeIds.AsSpan();

            // Copy encoded tokens, attention masks, and token type IDs to flattened arrays
            foreach (var (InputIds, TokenTypeIds, AttentionMask) in encoded)
            {
                InputIds.AsSpan().CopyTo(flattenIDsSpan); // Copy input tokens
                flattenIDsSpan = flattenIDsSpan.Slice(InputIds.Length); // Move span to next position
                
                AttentionMask.AsSpan().CopyTo(flattenAttentionMaskSpan); // Copy attention masks
                flattenAttentionMaskSpan = flattenAttentionMaskSpan.Slice(AttentionMask.Length); // Move span to next position
                
                TokenTypeIds.AsSpan().CopyTo(flattenTokenTypeIdsSpan); // Copy token type IDs
                flattenTokenTypeIdsSpan = flattenTokenTypeIdsSpan.Slice(TokenTypeIds.Length); // Move span to next position
            }

            var dimensions = new[] { numSentences, tokenCount }; // Dimensions of the input tensor

            // Create named ONNX values for input tensors
            var input = new NamedOnnxValue[3]
            {
                NamedOnnxValue.CreateFromTensor("input_ids",      new DenseTensor<long>(flattenIDs,           dimensions)),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(flattenAttentionMask, dimensions)),
                NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(flattenTokenTypeIds,  dimensions))
            };

            // Prepare for execution
            using var runOptions   = new RunOptions(); // Options for the execution
            using var registration = cancellationToken.Register(() => runOptions.Terminate = true); // Register for cancellation

            // Execute the model with the input tensors
            using var output = _session.Run(input, _outputNames, runOptions); // Execute the model

            cancellationToken.ThrowIfCancellationRequested(); // Throw if cancellation is requested

            // Perform mean pooling and normalization on the output tensor
            var outputPooled            = MeanPooling((DenseTensor<float>)output.First().Value, encoded); // Perform mean pooling
            var outputPooledNormalized = Normalize(outputPooled); // Normalize the pooled output tensor

            const int embDim = 384; // Dimension of the embeddings

            // Flatten the normalized output tensor into a 2D array of floats
            var outputFlatten = new float[sentences.Length][]; // Initialize output array
            for (int s = 0; s < sentences.Length; s++)
            {
                var emb = new float[embDim]; // Initialize embedding array for the sentence
                outputFlatten[s] = emb;      // Assign embedding array to the output array

                // Copy normalized embeddings to the output array
                for (int i = 0; i < embDim; i++)
                {
                    emb[i] = outputPooledNormalized[s, i]; // Copy normalized embeddings
                }
            }

            return outputFlatten; // Return the flattened output array
        }
    }
}
