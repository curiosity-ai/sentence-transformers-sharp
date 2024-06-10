/*
 * This file defines the SentenceEncoder class, which is responsible for encoding sentences using the ArcticXs model.
 * It utilizes BERT tokenization to convert sentences into tokens and then performs inference using an ONNX model to generate sentence embeddings.
 * The class implements the IDisposable interface to release resources and provides a method to encode an array of sentences into vectors.
 */

using BERTTokenizers;
using BERTTokenizers.Base;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using static SentenceTransformers.ArcticXs.DenseTensorHelpers;
using System.Linq;
using System.Collections.Generic;
using SentenceTransformers;

namespace SentenceTransformers.ArcticXs
{
    // Represents an encoder for sentences using the ArcticXs model.
    public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
    {
        private readonly SessionOptions   _sessionOptions;
        private readonly InferenceSession _inferenceSession;
        private readonly TokenizerBase    _tokenizer;
        private readonly string[]         _outputNames;

        // Constructor for SentenceEncoder class
        public SentenceEncoder(SessionOptions sessionOptions = null)
        {
            _sessionOptions = sessionOptions ?? new SessionOptions();
            _inferenceSession = new InferenceSession(ResourceLoader.GetResource(typeof(SentenceEncoder).Assembly, "model.onnx"), _sessionOptions);
            _tokenizer = new ArcticTokenizer(); // Initialize tokenizer
            _outputNames = _inferenceSession.OutputMetadata.Keys.ToArray(); // Get output names from the session
        }

        // Dispose method to release resources
        public void Dispose()
        {
            _sessionOptions.Dispose();
            _inferenceSession.Dispose();
        }

        // Encodes an array of sentences into vectors
        public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default)
        {
            var numSentences = sentences.Length;

            // Tokenize sentences
            var tokenizedSentences = _tokenizer.Encode(sentences);
            var tokenCount = tokenizedSentences.First().InputIds.Length;

            // Flatten token IDs, attention masks, and token type IDs
            var flattenInputIds = new long[tokenizedSentences.Sum(s => s.InputIds.Length)];
            var flattenAttentionMask = new long[tokenizedSentences.Sum(s => s.AttentionMask.Length)];
            var flattenTokenTypeIds = new long[tokenizedSentences.Sum(s => s.TokenTypeIds.Length)];

            var flattenInputIdsSpan = flattenInputIds.AsSpan();
            var flattenAttentionMaskSpan = flattenAttentionMask.AsSpan();
            var flattenTokenTypeIdsSpan = flattenTokenTypeIds.AsSpan();

            foreach (var (inputIds, tokenTypeIds, attentionMask) in tokenizedSentences)
            {
                inputIds.AsSpan().CopyTo(flattenInputIdsSpan);
                flattenInputIdsSpan = flattenInputIdsSpan.Slice(inputIds.Length);

                attentionMask.AsSpan().CopyTo(flattenAttentionMaskSpan);
                flattenAttentionMaskSpan = flattenAttentionMaskSpan.Slice(attentionMask.Length);

                tokenTypeIds.AsSpan().CopyTo(flattenTokenTypeIdsSpan);
                flattenTokenTypeIdsSpan = flattenTokenTypeIdsSpan.Slice(tokenTypeIds.Length);
            }

            // Create NamedOnnxValue objects for input tensor
            var dimensions = new[] { numSentences, tokenCount };

            var inputTensors = new NamedOnnxValue[3]
            {
                NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(flattenInputIds, dimensions)),
                NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(flattenAttentionMask, dimensions)),
                NamedOnnxValue.CreateFromTensor("token_type_ids", new DenseTensor<long>(flattenTokenTypeIds, dimensions))
            };

            // Run inference using the input tensor
            using var runOptions = new RunOptions();
            using var registration = cancellationToken.Register(() => runOptions.Terminate = true);

            using var output = _inferenceSession.Run(inputTensors, _outputNames, runOptions);

            var outputTensor = (DenseTensor<float>)output.First().Value;

            cancellationToken.ThrowIfCancellationRequested();

            // Normalize the output tensor
            return Normalize(outputTensor);
        }
    }
}
