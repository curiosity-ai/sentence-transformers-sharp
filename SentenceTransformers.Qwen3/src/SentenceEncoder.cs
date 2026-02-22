using System.Net.Http.Headers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BERTTokenizers.Base;
using static SentenceTransformers.Qwen3.DenseTensorHelpers;

namespace SentenceTransformers.Qwen3
{
    /// <summary>
    /// Sentence encoder for Qwen3-Embedding-0.6B-onnx-uint8.
    /// Loads a quantized ONNX (uint8) model from a file path, runs it, dequantizes the uint8 output
    /// to float[] and returns (optionally) normalized float embeddings.
    /// Use <see cref="CreateAsync"/> to download then load the model to a path.
    /// Tokenizer is loaded from Resources/tokenizer.json (under the application base directory), or from embedded resource if not found.
    /// </summary>
    public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
    {
        /// <summary>
        /// Default URL used to download the ONNX model when no path is provided.
        /// </summary>
        public const string DefaultModelUrl = "http://models.curiosity.ai/qwen3-06b-dynamic-uint8.onnx";

        private static readonly SemaphoreSlim _oneDownloadAtATime = new(1, 1);
        private static readonly HttpClient _downloadClient = new() { Timeout = TimeSpan.FromDays(1) };

        private readonly SessionOptions _sessionOptions;
        private readonly InferenceSession _session;
        private readonly string[] _outputNames;

        public TokenizerBase Tokenizer { get; }

        public static int GetMaxChunkLength() => 32768;
        public int MaxChunkLength => GetMaxChunkLength();

        /// <summary>Returns true if a model download is currently in progress.</summary>
        public static bool IsDownloading() => _oneDownloadAtATime.CurrentCount < 1;

        /// <summary>
        /// Downloads the model from <paramref name="modelUrl"/> to <paramref name="downloadToPath"/> (or a temp path if null),
        /// then creates an encoder from that path. Tokenizer is loaded from Resources/tokenizer.json or embedded resource.
        /// </summary>
        public static async Task<SentenceEncoder> CreateAsync(
            SessionOptions sessionOptions = null,
            string modelUrl = null,
            string downloadToPath = null,
            IProgress<float> progress = null,
            CancellationToken cancellationToken = default)
        {
            var path = downloadToPath ?? Path.Combine(Path.GetTempPath(), "SentenceTransformers.Qwen3", "qwen3-model.onnx");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            await DownloadModelAsync(modelUrl ?? DefaultModelUrl, path, progress, cancellationToken);
            return new SentenceEncoder(sessionOptions, path);
        }

        // Quantization calibration constants from electroglyph ONNX README.
        // These are the float range observed during calibration; we map uint8 -> float.
        private const float QUANT_MIN = -0.3009805381298065f;
        private const float QUANT_MAX = 0.3952634334564209f;
        private readonly float _scale;
        private readonly int _zeroPoint;

        // Toggle if you want to match “no normalization” behavior
        public bool Normalize { get; set; } = true;

        /// <summary>Creates an encoder from an existing ONNX model file at <paramref name="modelOnnxPath"/>.</summary>
        /// <param name="modelOnnxPath">Path to the ONNX model file.</param>
        public SentenceEncoder(SessionOptions sessionOptions = null, string modelOnnxPath = null)
        {
            if (string.IsNullOrWhiteSpace(modelOnnxPath))
            {
                throw new ArgumentException("Model path is required.", nameof(modelOnnxPath));
            }
            _sessionOptions = sessionOptions ?? new SessionOptions();
            _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            _scale = (QUANT_MAX - QUANT_MIN) / 255.0f;
            _zeroPoint = (int)Math.Round(-QUANT_MIN / _scale);

            _session = new InferenceSession(modelOnnxPath, _sessionOptions);
            _outputNames = _session.OutputMetadata.Keys.ToArray();

            var tokPathToUse = ResolveTokenizerPath(modelOnnxPath);
            Tokenizer = new QwenTokenizer(tokPathToUse, MaxChunkLength);
        }

        /// <summary>Resolves tokenizer path: Resources/tokenizer.json (under app base directory), or embedded resource written to temp if not found.</summary>
        private static string ResolveTokenizerPath(string modelOnnxPath)
        {
            var resourcesPath = Path.Combine(AppContext.BaseDirectory, "Resources", "tokenizer.json");
            if (File.Exists(resourcesPath))
            {
                return resourcesPath;
            }

            throw new FileNotFoundException(resourcesPath);
        }

        public void Dispose()
        {
            _session?.Dispose();
            _sessionOptions?.Dispose();
        }

        public float[][] Encode(string[] sentences, CancellationToken cancellationToken = default)
        {
            if (sentences is null || sentences.Length == 0)
            {
                return Array.Empty<float[]>();
            }

            var encoded = Tokenizer.Encode(sentences);
            if (encoded.Count == 0)
            {
                return Array.Empty<float[]>();
            }

            int batch = encoded.Count;
            int maxLen = encoded.Max(e => e.InputIds.Length);
            if (maxLen <= 0)
            {
                return Array.Empty<float[]>();
            }

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

                if (Normalize)
                {
                    NormalizeRows(result);
                }
                return result;
            }
            catch (OnnxRuntimeException e)
            {
                if (cancellationToken.IsCancellationRequested)
                {
                    throw new OperationCanceledException("Encoding was cancelled", e, cancellationToken);
                }
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
            string typeIdsName = _session.InputMetadata.Keys.FirstOrDefault(k => k == "token_type_ids");

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
            if (session.InputMetadata.ContainsKey(preferred))
            {
                return preferred;
            }

            // Basic fallback heuristics
            var match = session.InputMetadata.Keys.FirstOrDefault(k => k.Contains(preferred, StringComparison.OrdinalIgnoreCase));
            if (match is not null)
            {
                return match;
            }

            throw new InvalidOperationException(
                $"Model input '{preferred}' not found. Available inputs: {string.Join(", ", session.InputMetadata.Keys)}");
        }

        /// <summary>
        /// Downloads the ONNX model from <paramref name="modelUrl"/> to <paramref name="localPath"/>.
        /// Only one download runs at a time. On failure, any partial file at <paramref name="localPath"/> is deleted.
        /// </summary>
        public static async Task DownloadModelAsync(string modelUrl, string localPath, IProgress<float> progress = null, CancellationToken cancellationToken = default)
        {
            if (File.Exists(localPath)) return;

            if (!Uri.TryCreate(modelUrl, UriKind.Absolute, out var uri) || uri.Scheme is not ("https" or "http"))
            {
                throw new InvalidOperationException($"Invalid model URL: '{modelUrl}'. Use a valid http(s) URL.");
            }

            if (string.IsNullOrWhiteSpace(localPath))
            {
                throw new ArgumentException("Local path must be non-empty.", nameof(localPath));
            }


            await _oneDownloadAtATime.WaitAsync(cancellationToken);
            try
            {
                var buffer = new byte[2 << 18]; // 512kb
                var response = await _downloadClient.GetAsync(modelUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                try
                {
                    response.EnsureSuccessStatusCode();
                    long? totalFileSize = response.Content.Headers.ContentLength;
                    var supportsRange = response.Headers.AcceptRanges.Contains("bytes");

                    await using var fileStream = new FileStream(localPath, FileMode.Create, FileAccess.Write, FileShare.None, buffer.Length, true);
                    var totalBytesRead = 0L;
                    var finished = false;

                    while (!finished)
                    {
                        try
                        {
                            await using var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken);
                            int bytesRead;
                            while ((bytesRead = await contentStream.ReadAsync(buffer, cancellationToken)) > 0)
                            {
                                await fileStream.WriteAsync(buffer.AsMemory(0, bytesRead), cancellationToken);
                                totalBytesRead += bytesRead;
                                if (totalFileSize.HasValue && progress != null)
                                {
                                    progress.Report((float)totalBytesRead / totalFileSize.Value);
                                }
                            }
                            finished = true;
                        }
                        catch (Exception ex)
                        {
                            if (ex is TaskCanceledException)
                            {
                                try { File.Delete(localPath); } catch { /* ignore */ }
                                throw new OperationCanceledException("Model download was cancelled.", ex, cancellationToken);
                            }
                            await Task.Delay(5000, cancellationToken);
                            var newRequest = new HttpRequestMessage(HttpMethod.Get, modelUrl);
                            if (supportsRange && totalBytesRead > 0)
                            {
                                newRequest.Headers.Range = new RangeHeaderValue(totalBytesRead, null);
                            }
                            else
                            {
                                totalBytesRead = 0;
                                fileStream.Position = 0;
                            }
                            response.Dispose();
                            response = await _downloadClient.SendAsync(newRequest, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                            response.EnsureSuccessStatusCode();
                            if (!supportsRange || totalBytesRead == 0)
                            {
                                totalFileSize = response.Content.Headers.ContentLength;
                            }
                        }
                    }
                }
                finally
                {
                    response?.Dispose();
                }
            }
            catch (Exception ex)
            {
                File.Delete(localPath);
                throw;
            }
            finally
            {
                _oneDownloadAtATime.Release();
            }
        }
    }
}