using System.Net.Http.Headers;
using BERTTokenizers.Base;
using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Harrier.Small.Pure.Numerics;
using SentenceTransformers.Harrier.Small.Pure.Tokenizer;

namespace SentenceTransformers.Harrier.Small.Pure
{
    /// <summary>
    /// A 100% managed sentence encoder for harrier-oss-v1-270m (Microsoft's small Harrier multilingual
    /// embedding model) with <b>no native dependencies</b> - no ONNX Runtime and no native tokenizer.
    /// The Gemma3 forward pass and Gemma BPE tokenizer are implemented in pure C# on top of
    /// <see cref="System.Numerics.Tensors.TensorPrimitives"/>, so the package is trim/AOT friendly and
    /// runs anywhere .NET runs (including Blazor WASM and mobile) without shipping a single .so/.dll/.dylib.
    ///
    /// The model is decoder-only with last-token pooling; embeddings are L2-normalized and 640-dimensional.
    /// Use <see cref="CreateAsync"/> to download the original safetensors weights on first use and load them.
    /// </summary>
    public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
    {
        /// <summary>Task-specific instruction prompts published with the model. Queries are encoded with
        /// one of these prefixes; documents are encoded as-is. Format: <c>"Instruct: {task}\nQuery: {text}"</c>.</summary>
        public static class Prompts
        {
            public const string WebSearchQuery = "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery: ";
            public const string SemanticSimilarityQuery = "Instruct: Retrieve semantically similar text\nQuery: ";
            public const string BitextQuery = "Instruct: Retrieve parallel sentences\nQuery: ";
        }

        /// <summary>Default download URL for the bfloat16 safetensors weights (~540 MB). Points at the
        /// original Microsoft checkpoint on hosted by Curiosity; override <see cref="CreateAsync"/>'s
        /// <c>weightsUrl</c> to use a mirror.</summary>
        public const string DefaultWeightsUrl = "https://models.curiosity.ai/harrier-oss-v1-270m-safetensors/harrier-oss-v1-270m.safetensors";

        private static readonly SemaphoreSlim _oneDownloadAtATime = new(1, 1);
        private static readonly HttpClient _downloadClient = new() { Timeout = TimeSpan.FromDays(1) };

        private readonly Gemma3Model _model;

        public TokenizerBase Tokenizer { get; }
        private readonly HarrierSmallPureTokenizer _tokenizer;

        /// <summary>Maximum tokens per encoded input. The model itself supports 32768 positions, but the
        /// pure-managed forward pass uses full (non-windowed) causal attention, so cost grows with the
        /// square of the sequence length - a 32k-token input takes hours of single-threaded CPU. 2048 keeps
        /// a single encode in the seconds range; longer texts should be chunked (see <c>ChunkAndEncode*</c>).</summary>
        public static int GetMaxChunkLength() => 2048;
        public int MaxChunkLength => GetMaxChunkLength();

        /// <summary>The embedding dimension produced by this model.</summary>
        public int EmbeddingDimension => 640;

        /// <summary>Re-normalize embeddings to unit L2 length after pooling (default true).</summary>
        public bool Normalize { get; set; } = true;

        /// <summary>Returns true if a model download is currently in progress.</summary>
        public static bool IsDownloading() => _oneDownloadAtATime.CurrentCount < 1;

        /// <summary>
        /// Downloads the safetensors weights to <paramref name="downloadToPath"/> (or a temp path if null),
        /// then creates an encoder from them. The Gemma BPE tokenizer is loaded from the embedded
        /// <c>tokenizer.json</c>.
        /// </summary>
        /// <param name="weightsUrl">URL of the <c>.safetensors</c> file. Defaults to <see cref="DefaultWeightsUrl"/>.</param>
        /// <param name="downloadToPath">Where to cache the weights. Defaults to a temp folder.</param>
        /// <param name="quantization">Weight precision for the transformer layers. <see cref="Quantization.None"/>
        /// (float32) is the default and exactly matches the reference; <see cref="Quantization.Int8"/> and
        /// <see cref="Quantization.Int4"/> shrink the in-memory footprint ~4x / ~8x respectively.</param>
        /// <param name="reportProgress">Optional download progress callback (~2 Hz).</param>
        public static async Task<SentenceEncoder> CreateAsync(
            string weightsUrl = null,
            string downloadToPath = null,
            Quantization quantization = Quantization.None,
            Action<DownloadProgress> reportProgress = null,
            ParallelOptions parallelOptions = null)
        {

            parallelOptions ??= new ParallelOptions()
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount / 2
            };

            var path = downloadToPath ?? Path.Combine(Path.GetTempPath(), "SentenceTransformers.Harrier.Small.Pure", "harrier-small.safetensors");
            Directory.CreateDirectory(Path.GetDirectoryName(path)!);

            await DownloadFileAsync(weightsUrl ?? DefaultWeightsUrl, path, reportProgress, parallelOptions.CancellationToken).ConfigureAwait(false);
            return await LoadAsync(path, quantization: quantization, parallelOptions: parallelOptions).ConfigureAwait(false);
        }

        /// <summary>Creates an encoder from an existing safetensors file on disk. Loading (including the
        /// file read and any weight quantization) is fully asynchronous and non-blocking.</summary>
        /// <param name="safetensorsPath">Path to the harrier-oss-v1-270m <c>model.safetensors</c>.</param>
        /// <param name="tokenizerJsonPath">Optional path to <c>tokenizer.json</c>; when null the embedded copy is used.</param>
        /// <param name="quantization">Weight precision for the transformer layers (default float32).</param>
        public static async Task<SentenceEncoder> LoadAsync(
            string safetensorsPath,
            string tokenizerJsonPath = null,
            Quantization quantization = Quantization.None,
            ParallelOptions parallelOptions = null)
        {
            parallelOptions ??= new ParallelOptions()
            {
                MaxDegreeOfParallelism = Environment.ProcessorCount / 2
            };

            if (string.IsNullOrWhiteSpace(safetensorsPath))
            {
                throw new ArgumentException("Weights path is required.", nameof(safetensorsPath));
            }

            var model = await Gemma3Model.LoadAsync(safetensorsPath, new Gemma3Config(), quantization, parallelOptions).ConfigureAwait(false);
            var tokenizer = LoadTokenizer(tokenizerJsonPath, GetMaxChunkLength());
            return new SentenceEncoder(model, tokenizer);
        }

        private SentenceEncoder(Gemma3Model model, HarrierSmallPureTokenizer tokenizer)
        {
            _model = model;
            _tokenizer = tokenizer;
            Tokenizer = tokenizer;
        }

        /// <summary>Name of the tokenizer copied next to the app. Prefixed with the model name so it
        /// never collides with another encoder package's <c>Resources/tokenizer.json</c> in a shared
        /// output directory.</summary>
        private const string TokenizerFileName = "harrier-small-pure.tokenizer.json";

        private static HarrierSmallPureTokenizer LoadTokenizer(string tokenizerJsonPath, int maxTokens)
        {
            // 1. An explicit caller-provided path always wins.
            if (!string.IsNullOrWhiteSpace(tokenizerJsonPath) && File.Exists(tokenizerJsonPath))
            {
                return HarrierSmallPureTokenizer.FromFile(tokenizerJsonPath, maxTokens);
            }

            // 2. Prefer the tokenizer embedded in this assembly. It ships with the package and is
            //    guaranteed to be the correct Gemma tokenizer, so it can never be shadowed by another
            //    model's generic Resources/tokenizer.json that happens to share the output directory
            //    (which happens when several encoder packages are referenced from the same app).
            var stream = typeof(SentenceEncoder).Assembly.GetManifestResourceStream("tokenizer.json");
            if (stream is not null)
            {
                using (stream)
                {
                    return HarrierSmallPureTokenizer.FromStream(stream, maxTokens);
                }
            }

            // 3. Fall back to the model-name-prefixed copy next to the app (only reached if the embedded
            //    resource is somehow missing).
            var resourcesPath = Path.Combine(AppContext.BaseDirectory, "Resources", TokenizerFileName);
            if (File.Exists(resourcesPath))
            {
                return HarrierSmallPureTokenizer.FromFile(resourcesPath, maxTokens);
            }

            throw new FileNotFoundException($"tokenizer.json was not found embedded in the assembly or at Resources/{TokenizerFileName}.");
        }

        public Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default) => 
            EncodeAsync(sentences, new ParallelOptions() { CancellationToken = cancellationToken });

        /// <summary>Encodes a batch of texts into embeddings. Each input must fit in
        /// <see cref="MaxChunkLength"/> tokens; for longer text use the <c>ChunkAndEncode*</c> helpers.</summary>
        public async Task<float[][]> EncodeAsync(string[] sentences, ParallelOptions parallelOptions)
        {
            if (sentences is null || sentences.Length == 0)
            {
                return Array.Empty<float[]>();
            }

            var result = new float[sentences.Length][];
            for (int i = 0; i < sentences.Length; i++)
            {
                parallelOptions.CancellationToken.ThrowIfCancellationRequested();
                var ids = _tokenizer.EncodeIds(sentences[i] ?? string.Empty);
                if (ids.Length == 0)
                {
                    // <bos>/<eos> are always added, so this only happens for a deliberately empty tokenizer;
                    // return a zero vector to keep batch shape stable.
                    result[i] = new float[EmbeddingDimension];
                    continue;
                }
                var embedding = await _model.ForwardAsync(ids, parallelOptions).ConfigureAwait(false);
                if (Normalize)
                {
                    Ops.L2NormalizeInPlace(embedding);
                }
                result[i] = embedding;
            }
            return result;
        }

        public Task<float[][]> EncodeQueriesAsync(string[] queries, string promptPrefix = Prompts.WebSearchQuery, CancellationToken cancellationToken = default) =>
                EncodeQueriesAsync(queries, new ParallelOptions() { CancellationToken = cancellationToken, MaxDegreeOfParallelism = Environment.ProcessorCount }, promptPrefix);

        /// <summary>Encodes queries with the given instruction prompt prefix (see <see cref="Prompts"/>),
        /// then embeds them. Documents should be encoded with the plain <see cref="EncodeAsync(string[], CancellationToken)"/>.</summary>
        public Task<float[][]> EncodeQueriesAsync(string[] queries, ParallelOptions parallelOptions, string promptPrefix = Prompts.WebSearchQuery)
        {
            if (queries is null || queries.Length == 0)
            {
                return Task.FromResult(Array.Empty<float[]>());
            }
            var prompted = new string[queries.Length];
            for (int i = 0; i < queries.Length; i++)
            {
                prompted[i] = (promptPrefix ?? string.Empty) + (queries[i] ?? string.Empty);
            }
            return EncodeAsync(prompted, parallelOptions);
        }

        public List<string> ChunkTokens(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null)
            => BPEChunkAndEncodeHelpers.ChunkTokens(Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress);

        public List<AlignedString> ChunkTokensAligned(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null)
            => BPEChunkAndEncodeHelpers.ChunkTokensAligned(Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress);

        public Task<EncodedChunk[]> ChunkAndEncodeAsync(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
            => BPEChunkAndEncodeHelpers.ChunkAndEncodeAsync(this, text, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

        public Task<EncodedChunkAligned[]> ChunkAndEncodeAlignedAsync(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
            => BPEChunkAndEncodeHelpers.ChunkAndEncodeAlignedAsync(this, text, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

        public Task<TaggedEncodedChunk[]> ChunkAndEncodeTaggedAsync(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
            => BPEChunkAndEncodeHelpers.ChunkAndEncodeTaggedAsync(this, text, stripTags, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

        public Task<TaggedEncodedChunkAligned[]> ChunkAndEncodeTaggedAlignedAsync(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
            => BPEChunkAndEncodeHelpers.ChunkAndEncodeTaggedAlignedAsync(this, text, stripTags, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

        public void Dispose()
        {
            // No unmanaged resources; the large weight arrays are released with the instance.
        }

        /// <summary>
        /// Downloads a file from <paramref name="url"/> to <paramref name="localPath"/> with resume support.
        /// Only one download runs at a time process-wide. On failure, any partial file is deleted.
        /// </summary>
        public static async Task DownloadFileAsync(string url, string localPath, Action<DownloadProgress> reportProgress = null, CancellationToken cancellationToken = default)
        {
            if (File.Exists(localPath))
            {
                return;
            }
            if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme is not ("https" or "http"))
            {
                throw new InvalidOperationException($"Invalid weights URL: '{url}'. Use a valid http(s) URL.");
            }
            if (string.IsNullOrWhiteSpace(localPath))
            {
                throw new ArgumentException("Local path must be non-empty.", nameof(localPath));
            }

            var fileName = Path.GetFileName(localPath);
            long? totalBytes = null;

            await _oneDownloadAtATime.WaitAsync(cancellationToken);
            try
            {
                var buffer = new byte[2 << 18]; // 512 KB
                var response = await _downloadClient.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
                try
                {
                    response.EnsureSuccessStatusCode();
                    var supportsRange = response.Headers.AcceptRanges.Contains("bytes");
                    totalBytes = response.Content.Headers.ContentLength;

                    await using var fileStream = new FileStream(localPath, FileMode.Create, FileAccess.Write, FileShare.None, buffer.Length, true);
                    var totalBytesRead = 0L;
                    var finished = false;
                    var lastReportTicks = 0L;
                    Report(0L);

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
                            Report(totalBytesRead);
                        }
                        catch (Exception ex)
                        {
                            if (ex is TaskCanceledException)
                            {
                                try { File.Delete(localPath); } catch { /* ignore */ }
                                throw new OperationCanceledException("Weights download was cancelled.", ex, cancellationToken);
                            }
                            await Task.Delay(5000, cancellationToken);
                            var newRequest = new HttpRequestMessage(HttpMethod.Get, url);
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
                try { File.Delete(localPath); } catch { /* ignore */ }
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
