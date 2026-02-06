using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BERTTokenizers.Base;
using static SentenceTransformers.Qwen3.DenseTensorHelpers;

namespace SentenceTransformers.Qwen3
{
    /// <summary>
    /// Sentence encoder for Qwen3-Embedding-0.6B-onnx-uint8.
    /// Loads a quantized ONNX (uint8) model, runs it, dequantizes the uint8 output
    /// to float[] and returns (optionally) normalized float embeddings.
    /// When no model path or bytes are provided, the model is downloaded from <see cref="DefaultModelUrl"/>.
    /// </summary>
    public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
    {
        /// <summary>
        /// Default URL used to download the ONNX model when no path or bytes are provided.
        /// </summary>
        public const string DefaultModelUrl = "http://models.curiosity.ai/qwen3-06b-dynamic-uint8.onnx";

        private readonly SessionOptions _sessionOptions;
        private readonly InferenceSession _session;
        private readonly string[] _outputNames;

        public TokenizerBase Tokenizer { get; }

        public static int GetMaxChunkLength() => 32768;
        public int MaxChunkLength => GetMaxChunkLength();

        // Quantization calibration constants from electroglyph ONNX README.
        // These are the float range observed during calibration; we map uint8 -> float.
        private const float QUANT_MIN = -0.3009805381298065f;
        private const float QUANT_MAX = 0.3952634334564209f;
        private readonly float _scale;
        private readonly int _zeroPoint;

        // Toggle if you want to match “no normalization” behavior
        public bool Normalize { get; set; } = true;
        
        public SentenceEncoder(
            SessionOptions? sessionOptions = null,
            string? modelOnnxPath = null,
            string? tokenizerJsonPath = null,
            byte[]? modelBytes = null)
        {
            _sessionOptions = sessionOptions ?? new SessionOptions();

            // Optional: tweak ORT a bit
            _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            _scale = (QUANT_MAX - QUANT_MIN) / 255.0f;
            _zeroPoint = (int)Math.Round(-QUANT_MIN / _scale);

            // --- Load model ---
            if (modelBytes is { Length: > 0 })
            {
                _session = new InferenceSession(modelBytes, _sessionOptions);
            }
            else if (!string.IsNullOrWhiteSpace(modelOnnxPath))
            {
                _session = new InferenceSession(modelOnnxPath, _sessionOptions);
            }
            else
            {
                var bytes = DownloadModelFromUrl(DefaultModelUrl);
                _session = new InferenceSession(bytes, _sessionOptions);
            }

            _outputNames = _session.OutputMetadata.Keys.ToArray();

            // --- Load tokenizer ---
            string tokPathToUse;
            if (!string.IsNullOrWhiteSpace(tokenizerJsonPath))
            {
                tokPathToUse = tokenizerJsonPath!;
            }
            else
            {
                var tokBytes = ResourceLoader.GetResource(typeof(SentenceEncoder).Assembly, "tokenizer.json");
                tokPathToUse = WriteTempTokenizer(tokBytes);
            }

            Tokenizer = new QwenTokenizer(tokPathToUse, MaxChunkLength);
        }

        public void Dispose()
        {
            _session?.Dispose();
            _sessionOptions?.Dispose();
        }

        public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default)
        {
            if (sentences is null || sentences.Length == 0) return Array.Empty<float[]>();

            var encoded = Tokenizer.Encode(sentences);
            if (encoded.Count == 0) return Array.Empty<float[]>();

            int batch = encoded.Count;
            int maxLen = encoded.Max(e => e.InputIds.Length);
            if (maxLen <= 0) return Array.Empty<float[]>();

            var inputs = BuildModelInputs(encoded, batch, maxLen);

            using var runOptions = new RunOptions();
            using var registration = cancellationToken.Register(() => runOptions.Terminate = true);

            try
            {
                using var outputs = _session.Run(inputs, _outputNames, runOptions);
                var raw = outputs.First().Value;

                float[][] result = raw switch
                {
                    DenseTensor<float> floatTensor => CopyToJagged(floatTensor),
                    DenseTensor<byte> u8Tensor => DequantizeToJagged(u8Tensor, _scale, _zeroPoint),
                    DenseTensor<sbyte> s8Tensor => DequantizeToJagged(s8Tensor, _scale, _zeroPoint),
                    _ => throw new InvalidOperationException(
                        $"Unexpected ONNX output type: {raw?.GetType().FullName ?? "null"}")
                };

                if (Normalize) NormalizeRows(result);
                return result;
            }
            catch (OnnxRuntimeException e)
            {
                if (cancellationToken.IsCancellationRequested)
                    throw new OperationCanceledException("Encoding was cancelled", e, cancellationToken);
                throw;
            }
        }

        private List<NamedOnnxValue> BuildModelInputs(
            List<(long[] InputIds, long[] TokenTypeIds, long[] AttentionMask)> encoded,
            int batch,
            int maxLen)
        {
            var inputIds = new long[batch * maxLen];
            var attMask = new long[batch * maxLen];
            var typeIds = new long[batch * maxLen];

            for (int b = 0; b < batch; b++)
            {
                var (ids, tts, am) = encoded[b];
                int offset = b * maxLen;
                Array.Copy(ids, 0, inputIds, offset, ids.Length);
                Array.Copy(am, 0, attMask, offset, am.Length);
                if (tts is { Length: > 0 })
                {
                    Array.Copy(tts, 0, typeIds, offset, Math.Min(tts.Length, maxLen));
                }
                
            }

            var shape = new[] { batch, maxLen };
            string inputIdsName = FindInputName(_session, "input_ids");
            string attMaskName = FindInputName(_session, "attention_mask");
            string? typeIdsName = _session.InputMetadata.Keys.FirstOrDefault(k => k == "token_type_ids");

            var inputs = new List<NamedOnnxValue>(3)
            {
                NamedOnnxValue.CreateFromTensor(inputIdsName, new DenseTensor<long>(inputIds, shape)),
                NamedOnnxValue.CreateFromTensor(attMaskName, new DenseTensor<long>(attMask, shape)),
            };
            if (typeIdsName is not null)
            {
                inputs.Add(NamedOnnxValue.CreateFromTensor(typeIdsName, new DenseTensor<long>(typeIds, shape)));
            }
            return inputs;
        }

        private static string FindInputName(InferenceSession session, string preferred)
        {
            if (session.InputMetadata.ContainsKey(preferred)) return preferred;

            // Basic fallback heuristics
            var match = session.InputMetadata.Keys.FirstOrDefault(k => k.Contains(preferred, StringComparison.OrdinalIgnoreCase));
            if (match is not null) return match;

            throw new InvalidOperationException(
                $"Model input '{preferred}' not found. Available inputs: {string.Join(", ", session.InputMetadata.Keys)}");
        }

        private static byte[] DownloadModelFromUrl(string url)
        {
            if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme is not ("https" or "http"))
                throw new InvalidOperationException($"Invalid model URL: '{url}'. Use a valid http(s) URL.");

            using var client = new HttpClient();
            client.Timeout = TimeSpan.FromMinutes(5);
            return client.GetByteArrayAsync(uri).GetAwaiter().GetResult();
        }

        private static string WriteTempTokenizer(byte[] tokenizerJsonBytes)
        {
            // Deterministic-ish name based on length + a cheap hash
            int hash = 17;
            unchecked
            {
                foreach (var b in tokenizerJsonBytes.AsSpan(0, Math.Min(tokenizerJsonBytes.Length, 2048)))
                    hash = hash * 31 + b;
            }

            string dir = Path.Combine(Path.GetTempPath(), "SentenceTransformers.Qwen3");
            Directory.CreateDirectory(dir);

            string path = Path.Combine(dir, $"tokenizer_{tokenizerJsonBytes.Length}_{hash}.json");
            if (!File.Exists(path))
                File.WriteAllBytes(path, tokenizerJsonBytes);

            return path;
        }

    }
}