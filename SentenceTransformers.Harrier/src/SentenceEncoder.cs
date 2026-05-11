using System.Net.Http.Headers;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using BERTTokenizers.Base;
using SentenceTransformers;
using static SentenceTransformers.Harrier.DenseTensorHelpers;

namespace SentenceTransformers.Harrier
{
    /// <summary>
    /// Sentence encoder for harrier-oss-v1-0.6b ONNX (Microsoft's Harrier multilingual embedding model).
    /// Loads an ONNX model (with optional external data file) from a file path, runs it, and returns
    /// L2-normalized float embeddings (1024-dimensional). The model performs last-token pooling and
    /// L2 normalization internally and exposes a 'sentence_embedding' output.
    /// Use <see cref="CreateAsync"/> to download then load the model to a path.
    /// Tokenizer is loaded from Resources/tokenizer.json (under the application base directory).
    /// </summary>
    public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
    {
        /// <summary>
        /// Default URL used to download the ONNX model when no path is provided.
        /// Points to the standard quantized variant (~706 MB external data) which produces
        /// float32 output and is the most broadly compatible with onnxruntime CPU/GPU providers.
        /// </summary>
        public const string DefaultModelUrl = "https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/resolve/main/onnx/model_quantized.onnx";

        /// <summary>
        /// Default URL for the external ONNX data file that accompanies <see cref="DefaultModelUrl"/>.
        /// Hugging Face splits Harrier ONNX models into a small graph file plus a separate weights file.
        /// </summary>
        public const string DefaultModelDataUrl = "https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX/resolve/main/onnx/model_quantized.onnx_data";

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
        /// Downloads the model (and its external data file, when applicable) to <paramref name="downloadToPath"/>
        /// (or a temp path if null), then creates an encoder from that path. Tokenizer is loaded from
        /// Resources/tokenizer.json.
        /// </summary>
        /// <param name="modelUrl">URL for the .onnx graph file. Defaults to <see cref="DefaultModelUrl"/>.</param>
        /// <param name="modelDataUrl">URL for the accompanying .onnx_data weights file. Pass null to skip.
        /// Defaults to <see cref="DefaultModelDataUrl"/> when <paramref name="modelUrl"/> is also null.</param>
        public static async Task<SentenceEncoder> CreateAsync(
            SessionOptions sessionOptions = null,
            string modelUrl = null,
            string modelDataUrl = null,
            string downloadToPath = null,
            CancellationToken cancellationToken = default)
        {
            var path = downloadToPath ?? Path.Combine(Path.GetTempPath(), "SentenceTransformers.Harrier", "harrier-model.onnx");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            var resolvedModelUrl = modelUrl ?? DefaultModelUrl;
            // If user supplied modelUrl but no data url, do not auto-download data file.
            // If user left both null, use the default pair.
            var resolvedDataUrl = modelDataUrl ?? (modelUrl is null ? DefaultModelDataUrl : null);

            await DownloadModelAsync(resolvedModelUrl, path, cancellationToken);
            if (!string.IsNullOrEmpty(resolvedDataUrl))
            {
                // ONNX runtime expects the external data file to live next to the .onnx file
                // with the exact name referenced in the model's external_data entry. The
                // upstream models reference "model_q4f16.onnx_data" (or similar). Preserve
                // that filename to keep the reference resolvable.
                var dataFileName = Path.GetFileName(new Uri(resolvedDataUrl).LocalPath);
                var dataPath = Path.Combine(Path.GetDirectoryName(path)!, dataFileName);
                await DownloadModelAsync(resolvedDataUrl, dataPath, cancellationToken);
            }
            return new SentenceEncoder(sessionOptions, path);
        }

        // Toggle if you want to match "no normalization" behavior. The model already L2-normalizes
        // its output; this flag re-normalizes float[][] after copy as a safety net.
        public bool Normalize { get; set; } = true;

        /// <summary>Creates an encoder from an existing ONNX model file at <paramref name="modelOnnxPath"/>.</summary>
        /// <param name="modelOnnxPath">Path to the ONNX model file. If the model uses external data,
        /// the data file must live in the same directory under the name referenced by the graph.</param>
        public SentenceEncoder(SessionOptions sessionOptions = null, string modelOnnxPath = null)
        {
            if (string.IsNullOrWhiteSpace(modelOnnxPath))
            {
                throw new ArgumentException("Model path is required.", nameof(modelOnnxPath));
            }
            _sessionOptions = sessionOptions ?? new SessionOptions();
            _sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;

            _session = new InferenceSession(modelOnnxPath, _sessionOptions);
            _outputNames = ResolveOutputNames(_session);

            var tokPathToUse = ResolveTokenizerPath(modelOnnxPath);
            Tokenizer = new HarrierTokenizer(tokPathToUse, MaxChunkLength);
        }

        /// <summary>
        /// Picks the sentence-embedding output. Harrier exports an output named
        /// <c>sentence_embedding</c>; if absent, fall back to the first output.
        /// </summary>
        private static string[] ResolveOutputNames(InferenceSession session)
        {
            var keys = session.OutputMetadata.Keys.ToArray();
            var preferred = keys.FirstOrDefault(k => string.Equals(k, "sentence_embedding", StringComparison.Ordinal))
                         ?? keys.FirstOrDefault(k => k.Contains("sentence_embedding", StringComparison.OrdinalIgnoreCase))
                         ?? keys.FirstOrDefault();
            if (preferred is null)
            {
                throw new InvalidOperationException("ONNX model has no outputs.");
            }
            return new[] { preferred };
        }

        /// <summary>Resolves tokenizer path: Resources/tokenizer.json (under app base directory).</summary>
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

        public async Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
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
                    DenseTensor<Float16> halfTensor => CopyToJagged(halfTensor),
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
            // Harrier uses pad_token_id = 151643 (<|endoftext|>). Pre-fill input_ids with the pad id
            // so right-padding tokens are recognized by the model; attention_mask remains 0 there.
            const long PadTokenId = 151643L;
            var inputIds = new long[batch * maxLen];
            Array.Fill(inputIds, PadTokenId);
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
        /// Downloads a file from <paramref name="modelUrl"/> to <paramref name="localPath"/>.
        /// Only one download runs at a time. On failure, any partial file at <paramref name="localPath"/> is deleted.
        /// Re-used for both the .onnx graph and its accompanying .onnx_data file.
        /// </summary>
        public static async Task DownloadModelAsync(string modelUrl, string localPath, CancellationToken cancellationToken = default)
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
        }
    }
}
