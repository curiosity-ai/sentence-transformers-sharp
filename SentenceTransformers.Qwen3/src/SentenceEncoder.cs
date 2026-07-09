using System.Net.Http.Headers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BERTTokenizers.Base;
using SentenceTransformers;
using UID;
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
        public const string DefaultModelUrl = "https://models.curiosity.ai/qwen3-06b-dynamic-uint8.onnx";

        private static readonly SemaphoreSlim _oneDownloadAtATime = new(1, 1);
        private static readonly HttpClient _downloadClient = new() { Timeout = TimeSpan.FromDays(1) };

        private readonly SessionOptions _sessionOptions;
        private readonly InferenceSession _session;
        private readonly string[] _outputNames;

        /// <summary>LRU cache of the last 16 encoded vectors, keyed by each input's <c>Hash128()</c> UID.</summary>
        private readonly VectorCache _vectorCache = new(16);

        public TokenizerBase Tokenizer { get; }

        public static int GetMaxChunkLength() => 32768;
        public int MaxChunkLength => GetMaxChunkLength();

        /// <summary>Returns true if a model download is currently in progress.</summary>
        public static bool IsDownloading() => _oneDownloadAtATime.CurrentCount < 1;

        /// <summary>
        /// Downloads the model from <paramref name="modelUrl"/> to <paramref name="downloadToPath"/> (or a temp path if null),
        /// then creates an encoder from that path. Tokenizer is loaded from Resources/tokenizer.json or embedded resource.
        /// </summary>
        /// <param name="reportProgress">Optional progress callback. Receives a <see cref="DownloadProgress"/>
        /// roughly twice per second while the model is downloading.</param>
        public static async Task<SentenceEncoder> CreateAsync(
            SessionOptions sessionOptions = null,
            string modelUrl = null,
            string downloadToPath = null,
            Action<DownloadProgress> reportProgress = null,
            CancellationToken cancellationToken = default)
        {
            var path = downloadToPath ?? Path.Combine(Path.GetTempPath(), "SentenceTransformers.Qwen3", "qwen3-model.onnx");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);
            await DownloadModelAsync(modelUrl ?? DefaultModelUrl, path, reportProgress, cancellationToken);
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

        /// <summary>
        /// Resolves the tokenizer.json path. Prefers the copy embedded in this assembly (extracted once to
        /// a package-specific temp path) so it can never be shadowed by another encoder package's generic
        /// <c>Resources/tokenizer.json</c> sharing the output directory. Falls back to the model-name-prefixed
        /// copy next to the application only if the embedded resource is missing.
        /// </summary>
        private static string ResolveTokenizerPath(string modelOnnxPath)
        {
            var embedded = TryExtractEmbeddedTokenizer();
            if (embedded is not null)
            {
                return embedded;
            }

            var resourcesPath = Path.Combine(AppContext.BaseDirectory, "Resources", "qwen3.tokenizer.json");
            if (File.Exists(resourcesPath))
            {
                return resourcesPath;
            }

            throw new FileNotFoundException("tokenizer.json was not found embedded in the SentenceTransformers.Qwen3 assembly or at Resources/qwen3.tokenizer.json.");
        }

        /// <summary>Writes the embedded tokenizer.json to a stable temp path (once) and returns it, or
        /// <c>null</c> when no tokenizer is embedded in this assembly.</summary>
        private static string TryExtractEmbeddedTokenizer()
        {
            var bytes = ResourceLoader.GetResource(typeof(SentenceEncoder).Assembly, "tokenizer.json");
            if (bytes is null)
            {
                return null;
            }

            var cachedPath = Path.Combine(Path.GetTempPath(), "SentenceTransformers.Qwen3", "tokenizer.json");
            Directory.CreateDirectory(Path.GetDirectoryName(cachedPath)!);

            if (!File.Exists(cachedPath) || new FileInfo(cachedPath).Length != bytes.Length)
            {
                File.WriteAllBytes(cachedPath, bytes);
            }

            return cachedPath;
        }

        public void Dispose()
        {
            _session?.Dispose();
            _sessionOptions?.Dispose();
        }

        public async Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
        {
            if (sentences is null || sentences.Length == 0)
            {
                return Array.Empty<float[]>();
            }

            var       results = new float[sentences.Length][];
            var       keys    = new UID128[sentences.Length];
            List<int> misses  = null;

            // Serve any inputs whose Hash128() UID is already cached, and collect the cache misses.
            for (int i = 0; i < sentences.Length; i++)
            {
                keys[i] = (sentences[i] ?? string.Empty).Hash128();

                if (_vectorCache.TryGet(keys[i], out var cached))
                {
                    results[i] = cached;
                }
                else
                {
                    (misses ??= new List<int>()).Add(i);
                }
            }

            if (misses is null)
            {
                return results;
            }

            var toEncode = new string[misses.Count];
            for (int i = 0; i < misses.Count; i++)
            {
                toEncode[i] = sentences[misses[i]];
            }

            var newVectors = await EncodeCoreAsync(toEncode, cancellationToken);

            // Place each freshly encoded vector back in source order and add it to the cache.
            for (int i = 0; i < misses.Count; i++)
            {
                results[misses[i]] = newVectors[i];
                _vectorCache.Set(keys[misses[i]], newVectors[i]);
            }

            return results;
        }

        /// <inheritdoc/>
        public async Task<float[]> EncodeAsync(string sentence, CancellationToken cancellationToken = default)
        {
            var vectors = await EncodeAsync(new[] { sentence ?? string.Empty }, cancellationToken);
            return vectors.Length > 0 ? vectors[0] : null;
        }

        private async Task<float[][]> EncodeCoreAsync(string[] sentences, CancellationToken cancellationToken)
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

        /// <summary>
        /// Tokenizes <paramref name="text"/> and merges tokens into chunks of at most
        /// <paramref name="chunkLength"/> tokens with <paramref name="chunkOverlap"/> tokens of
        /// overlap. Each chunk's text is reconstructed by slicing the source between the first and
        /// last token's offsets returned by the Hugging Face tokenizer, which preserves the exact
        /// original whitespace and any injected markers. This is the BPE-appropriate replacement
        /// for the WordPiece-oriented <see cref="ISentenceEncoder.ChunkTokens"/> default.
        /// </summary>
        /// <param name="text">Source text to chunk.</param>
        /// <param name="chunkLength">Maximum tokens per chunk.</param>
        /// <param name="chunkOverlap">Tokens of overlap between consecutive chunks.</param>
        /// <param name="maxChunks">Hard cap on the number of chunks returned.</param>
        /// <param name="reportProgress">Optional progress callback receiving values in <c>[0,1]</c>.</param>
        public List<string> ChunkTokens(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null)
            => BPEChunkAndEncodeHelpers.ChunkTokens(Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress);

        /// <summary>
        /// Aligned variant of <see cref="ChunkTokens"/>: each chunk carries offsets back into
        /// <paramref name="text"/>.
        /// </summary>
        /// <inheritdoc cref="ChunkTokens"/>
        public List<AlignedString> ChunkTokensAligned(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null)
            => BPEChunkAndEncodeHelpers.ChunkTokensAligned(Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress);

        /// <summary>
        /// Splits <paramref name="text"/> into BPE-token-bounded chunks and encodes each chunk to
        /// an embedding. Uses offset-based chunk reconstruction (see <see cref="ChunkTokens"/>).
        /// </summary>
        /// <param name="text">Source text. May be longer than <see cref="MaxChunkLength"/>.</param>
        /// <param name="chunkLength">Maximum tokens per chunk. Values <c>&lt;= 0</c> or above <see cref="MaxChunkLength"/> are clamped to <see cref="MaxChunkLength"/>.</param>
        /// <param name="chunkOverlap">Tokens of overlap between consecutive chunks. Out-of-range values default to <c>chunkLength / 5</c>.</param>
        /// <param name="sequentially">When true, encode chunks one-by-one; when false, encode in a single batched call.</param>
        /// <param name="maxChunks">Hard cap on chunks produced.</param>
        /// <param name="keepResultsOnCancellation">When true and cancelled, returns the chunks already encoded.</param>
        /// <param name="reportProgress">Optional progress callback receiving values in <c>[0,1]</c>.</param>
        /// <param name="cancellationToken">Cancellation token.</param>
        public Task<EncodedChunk[]> ChunkAndEncodeAsync(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
            => BPEChunkAndEncodeHelpers.ChunkAndEncodeAsync(this, text, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

        /// <summary>
        /// Aligned variant of <see cref="ChunkAndEncodeAsync"/>: each result also carries offsets
        /// back into <paramref name="text"/>.
        /// </summary>
        /// <inheritdoc cref="ChunkAndEncodeAsync"/>
        public Task<EncodedChunkAligned[]> ChunkAndEncodeAlignedAsync(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
            => BPEChunkAndEncodeHelpers.ChunkAndEncodeAlignedAsync(this, text, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

        /// <summary>
        /// Chunks <paramref name="text"/> using BPE offsets, then for each chunk runs
        /// <paramref name="stripTags"/> to remove injected markers and recover a tag before
        /// encoding the cleaned text. The chunker treats markers as ordinary tokens.
        /// </summary>
        /// <param name="text">Source text with injected markers.</param>
        /// <param name="stripTags">Callback that strips markers and produces a <see cref="TaggedChunk"/> (cleaned text + tag).</param>
        /// <inheritdoc cref="ChunkAndEncodeAsync"/>
        public Task<TaggedEncodedChunk[]> ChunkAndEncodeTaggedAsync(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
            => BPEChunkAndEncodeHelpers.ChunkAndEncodeTaggedAsync(this, text, stripTags, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

        /// <summary>
        /// Aligned variant of <see cref="ChunkAndEncodeTaggedAsync"/>: each result carries offsets
        /// into <paramref name="text"/> alongside the cleaned text, tag, and embedding.
        /// </summary>
        /// <param name="text">Source text with injected markers.</param>
        /// <param name="stripTags">Callback that strips markers from the original chunk substring and produces a <see cref="TaggedChunk"/>.</param>
        /// <inheritdoc cref="ChunkAndEncodeAsync"/>
        public Task<TaggedEncodedChunkAligned[]> ChunkAndEncodeTaggedAlignedAsync(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
            => BPEChunkAndEncodeHelpers.ChunkAndEncodeTaggedAlignedAsync(this, text, stripTags, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

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
        /// <param name="reportProgress">Optional progress callback. Receives a <see cref="DownloadProgress"/>
        /// at most twice per second while bytes are flowing. Set to <c>null</c> to skip reporting.</param>
        public static async Task DownloadModelAsync(string modelUrl, string localPath, Action<DownloadProgress> reportProgress = null, CancellationToken cancellationToken = default)
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

            var fileName = Path.GetFileName(localPath);
            // Declared in the outer scope so the Report local function (below) can read it across
            // the try block. Refreshed when the response changes on a ranged retry.
            long? totalBytes = null;

            await _oneDownloadAtATime.WaitAsync(cancellationToken);
            try
            {
                var buffer = new byte[2 << 18]; // 512kb
                var response = await _downloadClient.GetAsync(modelUrl, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                try
                {
                    response.EnsureSuccessStatusCode();
                    var supportsRange = response.Headers.AcceptRanges.Contains("bytes");
                    totalBytes = response.Content.Headers.ContentLength;

                    await using var fileStream = new FileStream(localPath, FileMode.Create, FileAccess.Write, FileShare.None, buffer.Length, true);
                    var totalBytesRead = 0L;
                    var finished = false;
                    var lastReportTicks = 0L;
                    Report(0L); // initial 0% notification so callers can show "starting…"

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
                                var nowTicks = Environment.TickCount64;
                                if (nowTicks - lastReportTicks >= 500)
                                {
                                    lastReportTicks = nowTicks;
                                    Report(totalBytesRead);
                                }
                            }
                            finished = true;
                            Report(totalBytesRead); // final 100% notification
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
                            if (response.Content.Headers.ContentLength is long len)
                            {
                                totalBytes = totalBytesRead + len;
                            }
                        }
                    }
                }
                finally
                {
                    response?.Dispose();
                }
            }
            catch (Exception)
            {
                File.Delete(localPath);
                throw;
            }
            finally
            {
                _oneDownloadAtATime.Release();
            }

            void Report(long received)
            {
                if (reportProgress is null) return;
                var fraction = totalBytes is long total && total > 0
                    ? Math.Clamp(received / (float)total, 0f, 1f)
                    : 0f;
                reportProgress(new DownloadProgress(received, totalBytes, fraction, fileName));
            }
        }
    }
}