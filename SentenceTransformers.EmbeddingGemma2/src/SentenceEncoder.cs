using BERTTokenizers.Base;
using SentenceTransformers.EmbeddingGemma2.Audio;
using SentenceTransformers.EmbeddingGemma2.LiteRt;
using SentenceTransformers.EmbeddingGemma2.Model;
using SentenceTransformers.EmbeddingGemma2.Numerics;
using SentenceTransformers.EmbeddingGemma2.Tokenizer;
using SentenceTransformers.EmbeddingGemma2.Vision;
using UID;

namespace SentenceTransformers.EmbeddingGemma2;

/// <summary>What to do with inputs longer than the model's context (1024 tokens incl. BOS/EOS).</summary>
public enum InputOverflowStrategy
{
    /// <summary>Keep <c>&lt;bos&gt;</c>, the first tokens and <c>&lt;eos&gt;</c> (LiteRT-LM <c>TRUNCATE</c>). Default.</summary>
    Truncate,

    /// <summary>Split the sequence into context-sized chunks and average their (unnormalized) embeddings
    /// (LiteRT-LM <c>CHUNK_AND_AVERAGE</c>).</summary>
    ChunkAndAverage,

    /// <summary>Throw (LiteRT-LM's default <c>ERROR</c>).</summary>
    Error,
}

/// <summary>
/// A 100% managed sentence encoder for Google's <b>EmbeddingGemma 2</b> family
/// (<c>embeddinggemma-2-text-270m</c>, <c>embeddinggemma-2-text-vision-440m</c>, <c>embeddinggemma-2-740m</c>)
/// with <b>no native dependencies</b>: it reads the official LiteRT-LM <c>.litertlm</c> bundles and runs the
/// SentencePiece tokenizer, the int4 text transformer, the int8 vision transformer and the audio encoder
/// in SIMD-vectorized C# - a port of the <c>litert-lm</c> runtime's embedding engine that reproduces its
/// numerics (dynamically quantized int8 activations against int4 weights, static int8 vision activations).
/// <para>
/// Embeddings are 768-dimensional, mean-pooled and L2-normalized, and support Matryoshka truncation to
/// 512 / 256 / 128 dimensions via <see cref="OutputDimension"/>. Use the task prompts in <see cref="Prompts"/>
/// for queries and <see cref="Prompts.Document"/> (or <see cref="Prompts.DocumentWithTitle"/>) for documents.
/// Images (440M / 740M) and audio (740M) are embedded into the same space, alone or interleaved with text,
/// via <see cref="EncodeContentAsync(IReadOnlyList{EmbeddingGemma2Content}, CancellationToken)"/>.
/// </para>
/// </summary>
public sealed class SentenceEncoder : IDisposable, ISentenceEncoder
{
    /// <summary>EmbeddingGemma 2 task prompts (<c>config_sentence_transformers.json</c>). Queries use
    /// <c>task: … | query: </c>; documents use <c>title: … | text: </c>.</summary>
    public static class Prompts
    {
        public const string SearchQuery = "task: search result | query: ";
        public const string QuestionAnswering = "task: question answering | query: ";
        public const string FactChecking = "task: fact checking | query: ";
        public const string CodeRetrieval = "task: code retrieval | query: ";
        public const string Classification = "task: classification | query: ";
        public const string Clustering = "task: clustering | query: ";
        public const string SentenceSimilarity = "task: sentence similarity | query: ";
        /// <summary>Document prefix when no title is available.</summary>
        public const string Document = "title: none | text: ";

        /// <summary>Document prefix with a real title: <c>title: {title} | text: </c>.</summary>
        public static string DocumentWithTitle(string title) => $"title: {(string.IsNullOrWhiteSpace(title) ? "none" : title)} | text: ";
    }

    /// <summary>Native embedding size.</summary>
    public const int NativeDimension = 768;

    private readonly LiteRtLmFile _file;
    private readonly TextEncoder _text;
    private readonly EmbeddingGemma2Tokenizer _tokenizer;
    private readonly EmbeddingMetadata _metadata;
    private readonly Lazy<VisionEncoder> _vision;
    private readonly Lazy<AudioEncoder> _audio;
    private readonly ParallelOptions _defaultParallelOptions;
    private readonly VectorCache _vectorCache = new(16);
    private readonly int _startOfImageId, _startOfAudioId, _endOfImageId;
    private int _outputDimension = NativeDimension;
    private int _visionTokensPerImage;

    public TokenizerBase Tokenizer => _tokenizer;

    /// <summary>The bundle this encoder was loaded from (null when loaded from an unknown file).</summary>
    public EmbeddingGemma2Model? Model { get; }

    /// <summary>Path of the loaded <c>.litertlm</c> bundle.</summary>
    public string ModelPath => _file.Path;

    /// <summary>Maximum sequence length (tokens incl. BOS/EOS) of the text transformer.</summary>
    public int MaxInputLength { get; }

    public int MaxChunkLength => MaxInputLength - 2;

    /// <summary>The embedding size returned (Matryoshka truncation). One of 768 (default), 512, 256, 128
    /// - any size in [1, 768] is accepted; vectors are re-normalized after truncation.</summary>
    public int OutputDimension
    {
        get => _outputDimension;
        set
        {
            if (value < 1 || value > NativeDimension)
            {
                throw new ArgumentOutOfRangeException(nameof(value), $"OutputDimension must be in [1, {NativeDimension}].");
            }
            _outputDimension = value;
        }
    }

    /// <summary>Same as <see cref="OutputDimension"/>.</summary>
    public int EmbeddingDimension => _outputDimension;

    /// <summary>L2-normalize embeddings (default true, like the reference engine).</summary>
    public bool Normalize { get; set; } = true;

    /// <summary>How over-long inputs are handled (default <see cref="InputOverflowStrategy.Truncate"/>).</summary>
    public InputOverflowStrategy OverflowStrategy { get; set; } = InputOverflowStrategy.Truncate;

    /// <summary>True when the bundle contains the vision encoder (440M, 740M).</summary>
    public bool SupportsImages { get; }

    /// <summary>True when the bundle contains the audio encoder (740M).</summary>
    public bool SupportsAudio { get; }

    /// <summary>Soft tokens per image (the vision signature): 140 (default, 1260 patches) or 70 (630 patches).</summary>
    public int VisionTokensPerImage
    {
        get => _visionTokensPerImage;
        set
        {
            if (value != 70 && value != 140)
            {
                throw new ArgumentOutOfRangeException(nameof(value), "Supported vision token budgets are 70 and 140.");
            }
            _visionTokensPerImage = value;
        }
    }

    private SentenceEncoder(LiteRtLmFile file, EmbeddingGemma2Model? model, ParallelOptions defaultParallelOptions)
    {
        _file = file;
        Model = model;
        _metadata = file.ReadEmbeddingMetadata();
        MaxInputLength = _metadata.MaxInputLength > 0 ? _metadata.MaxInputLength : 1024;
        _tokenizer = new EmbeddingGemma2Tokenizer(SentencePieceBpe.FromModelProto(file.ReadSection(LiteRtLmSectionType.SP_Tokenizer)), _metadata.BosId, _metadata.EosId, MaxInputLength);
        _text = TextEncoder.Load(file);
        SupportsImages = file.HasModel(VisionEncoder.EncoderModelType);
        SupportsAudio = file.HasModel(AudioEncoder.EncoderModelType);
        _vision = new Lazy<VisionEncoder>(() => VisionEncoder.Load(_file), LazyThreadSafetyMode.ExecutionAndPublication);
        _audio = new Lazy<AudioEncoder>(() => AudioEncoder.Load(_file, _metadata), LazyThreadSafetyMode.ExecutionAndPublication);
        _defaultParallelOptions = defaultParallelOptions;
        _startOfImageId = SpecialTokenId(_metadata.StartOfImageToken);
        _endOfImageId = SpecialTokenId(_metadata.EndOfImageToken);
        _startOfAudioId = SpecialTokenId(_metadata.StartOfAudioToken);
        // LiteRT-LM default: max_num_patches / k² soft tokens per image.
        int k = Math.Max(1, _metadata.PoolingKernelSize);
        _visionTokensPerImage = _metadata.MaxNumPatches > 0 ? _metadata.MaxNumPatches / (k * k) : 140;
    }

    private int SpecialTokenId(string token)
    {
        if (string.IsNullOrEmpty(token))
        {
            return -1;
        }
        var ids = _tokenizer.EncodeIdsWithoutSpecialTokens(token);
        return ids.Length == 1 ? ids[0] : -1;
    }

    /// <summary>
    /// Downloads the requested EmbeddingGemma 2 bundle (once; it is cached at <paramref name="downloadToPath"/>,
    /// by default under the temp folder) and loads it.
    /// </summary>
    /// <param name="model">Which bundle: text-only 270M (default), text+vision 440M, or text+vision+audio 740M.</param>
    /// <param name="modelUrl">Override the download URL (defaults to the Hugging Face <c>litert-community</c> repo).</param>
    /// <param name="downloadToPath">Where to cache the <c>.litertlm</c> file.</param>
    /// <param name="reportProgress">Optional download progress callback (~2 Hz).</param>
    /// <param name="parallelOptions">Default concurrency for encoding (defaults to all cores). Use
    /// <c>MaxDegreeOfParallelism = 1</c> when many requests are encoded concurrently.</param>
    public static async Task<SentenceEncoder> CreateAsync(
        EmbeddingGemma2Model model = EmbeddingGemma2Model.Text270M,
        string modelUrl = null,
        string downloadToPath = null,
        Action<DownloadProgress> reportProgress = null,
        ParallelOptions parallelOptions = null,
        CancellationToken cancellationToken = default)
    {
        var path = downloadToPath ?? EmbeddingGemma2Models.GetDefaultCachePath(model);
        var url = modelUrl ?? EmbeddingGemma2Models.GetDownloadUrl(model);
        long? size = modelUrl is null ? EmbeddingGemma2Models.GetFileSize(model) : null;
        await ModelDownloader.DownloadFileAsync(url, path, reportProgress, size, cancellationToken).ConfigureAwait(false);
        try
        {
            return await LoadAsync(path, parallelOptions, model).ConfigureAwait(false);
        }
        catch (InvalidDataException)
        {
            // A corrupt cached bundle (e.g. a truncated file left by an older version): fetch it again once.
            try { File.Delete(path); } catch { /* ignore */ }
            await ModelDownloader.DownloadFileAsync(url, path, reportProgress, size, cancellationToken).ConfigureAwait(false);
            return await LoadAsync(path, parallelOptions, model).ConfigureAwait(false);
        }
    }

    /// <summary>Downloads a bundle without loading it (e.g. to pre-populate a cache).</summary>
    public static Task DownloadModelAsync(EmbeddingGemma2Model model, string downloadToPath = null, Action<DownloadProgress> reportProgress = null, CancellationToken cancellationToken = default)
        => ModelDownloader.DownloadFileAsync(EmbeddingGemma2Models.GetDownloadUrl(model), downloadToPath ?? EmbeddingGemma2Models.GetDefaultCachePath(model), reportProgress, EmbeddingGemma2Models.GetFileSize(model), cancellationToken);

    /// <summary>Loads an encoder from a local <c>.litertlm</c> bundle (any of the three variants).</summary>
    public static Task<SentenceEncoder> LoadAsync(string litertlmPath, ParallelOptions parallelOptions = null, EmbeddingGemma2Model? model = null)
    {
        if (string.IsNullOrWhiteSpace(litertlmPath))
        {
            throw new ArgumentException("Model path is required.", nameof(litertlmPath));
        }
        var po = parallelOptions ?? new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount };
        return Task.Run(() =>
        {
            var file = LiteRtLmFile.Open(litertlmPath);
            try
            {
                return new SentenceEncoder(file, model ?? Detect(file), po);
            }
            catch
            {
                file.Dispose();
                throw;
            }
        });
    }

    private static EmbeddingGemma2Model? Detect(LiteRtLmFile file)
    {
        if (file.HasModel(AudioEncoder.EncoderModelType))
        {
            return EmbeddingGemma2Model.Multimodal740M;
        }
        if (file.HasModel(VisionEncoder.EncoderModelType))
        {
            return EmbeddingGemma2Model.TextVision440M;
        }
        return file.HasModel(TextEncoder.EncoderModelType) ? EmbeddingGemma2Model.Text270M : null;
    }

    private ParallelOptions Options(CancellationToken ct) => new()
    {
        MaxDegreeOfParallelism = _defaultParallelOptions.MaxDegreeOfParallelism,
        CancellationToken = ct,
        TaskScheduler = _defaultParallelOptions.TaskScheduler ?? TaskScheduler.Default,
    };

    // ---------------------------------------------------------------------------------------------
    // Text
    // ---------------------------------------------------------------------------------------------

    /// <summary>Encodes texts as-is (no prompt). For retrieval, prefer <see cref="EncodeQueriesAsync(string[], string, CancellationToken)"/>
    /// and <see cref="EncodeDocumentsAsync"/>, which add the task prompts the model was trained with.</summary>
    public Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
        => EncodeAsync(sentences, Options(cancellationToken));

    public async Task<float[][]> EncodeAsync(string[] sentences, ParallelOptions parallelOptions)
    {
        if (sentences is null || sentences.Length == 0)
        {
            return Array.Empty<float[]>();
        }
        var raw = new float[sentences.Length][];
        var keys = new UID128[sentences.Length];
        var misses = new List<int>();
        for (int i = 0; i < sentences.Length; i++)
        {
            // The raw (untruncated, unnormalized) vector depends on the overflow strategy only.
            keys[i] = ((char)('0' + (int)OverflowStrategy) + (sentences[i] ?? string.Empty)).Hash128();
            if (_vectorCache.TryGet(keys[i], out var cached))
            {
                raw[i] = cached;
            }
            else
            {
                misses.Add(i);
            }
        }
        if (misses.Count > 0)
        {
            var computed = await Task.Run(() => EncodeTexts(misses.Select(i => sentences[i] ?? string.Empty).ToArray(), parallelOptions)).ConfigureAwait(false);
            for (int m = 0; m < misses.Count; m++)
            {
                raw[misses[m]] = computed[m];
                _vectorCache.Set(keys[misses[m]], computed[m]);
            }
        }
        var result = new float[sentences.Length][];
        for (int i = 0; i < raw.Length; i++)
        {
            result[i] = Finish(raw[i]);
        }
        return result;
    }

    public async Task<float[]> EncodeAsync(string sentence, CancellationToken cancellationToken = default)
        => (await EncodeAsync(new[] { sentence ?? string.Empty }, cancellationToken).ConfigureAwait(false))[0];

    /// <summary>Encodes queries with a task prompt (default <see cref="Prompts.SearchQuery"/>).</summary>
    public Task<float[][]> EncodeQueriesAsync(string[] queries, string promptPrefix = Prompts.SearchQuery, CancellationToken cancellationToken = default)
        => EncodeAsync(Prefix(queries, _ => promptPrefix), cancellationToken);

    /// <summary>Encodes documents as <c>title: {title} | text: {document}</c> (<c>title: none</c> when no title is given).</summary>
    public Task<float[][]> EncodeDocumentsAsync(string[] documents, string[] titles = null, CancellationToken cancellationToken = default)
    {
        if (titles is not null && documents is not null && titles.Length != documents.Length)
        {
            throw new ArgumentException("titles must be null or have one entry per document.", nameof(titles));
        }
        return EncodeAsync(Prefix(documents, i => titles is null ? Prompts.Document : Prompts.DocumentWithTitle(titles[i])), cancellationToken);
    }

    private static string[] Prefix(string[] texts, Func<int, string> prefix)
    {
        if (texts is null)
        {
            return Array.Empty<string>();
        }
        var r = new string[texts.Length];
        for (int i = 0; i < texts.Length; i++)
        {
            r[i] = (prefix(i) ?? string.Empty) + (texts[i] ?? string.Empty);
        }
        return r;
    }

    /// <summary>Token budget of one stacked forward pass (bounds scratch memory for large batches).</summary>
    private const int TokensPerBatch = 8192;

    /// <summary>Unnormalized 768-d embeddings for raw texts (BOS/EOS added, overflow handled).</summary>
    private float[][] EncodeTexts(string[] texts, ParallelOptions po)
    {
        var sequences = new List<int[]>();
        var owner = new List<int>();
        var chunkCounts = new int[texts.Length];
        for (int i = 0; i < texts.Length; i++)
        {
            var content = _tokenizer.EncodeIdsWithoutSpecialTokens(texts[i]);
            foreach (var seq in ApplyOverflow(content))
            {
                sequences.Add(seq);
                owner.Add(i);
                chunkCounts[i]++;
            }
        }
        var outputs = RunBatched(sequences, po);
        var result = new float[texts.Length][];
        for (int s = 0; s < outputs.Length; s++)
        {
            int i = owner[s];
            if (result[i] is null)
            {
                result[i] = outputs[s];
            }
            else
            {
                System.Numerics.Tensors.TensorPrimitives.Add(result[i], outputs[s], result[i]);
            }
        }
        for (int i = 0; i < texts.Length; i++)
        {
            if (chunkCounts[i] > 1)
            {
                System.Numerics.Tensors.TensorPrimitives.Divide(result[i], chunkCounts[i], result[i]);
            }
        }
        return result;
    }

    private IEnumerable<int[]> ApplyOverflow(int[] content)
    {
        int bos = _tokenizer.BosId, eos = _tokenizer.EosId;
        int total = content.Length + 2;
        if (total <= MaxInputLength)
        {
            yield return Wrap(content, 0, content.Length, true, true);
            yield break;
        }
        switch (OverflowStrategy)
        {
            case InputOverflowStrategy.Error:
                throw new ArgumentException($"Input sequence length ({total}) exceeds the maximum supported length ({MaxInputLength}).");
            case InputOverflowStrategy.Truncate:
                yield return Wrap(content, 0, MaxInputLength - 2, true, true);
                yield break;
            default:
                // Chunk the full [bos, content..., eos] sequence into context-sized pieces.
                var full = Wrap(content, 0, content.Length, true, true);
                for (int s = 0; s < full.Length; s += MaxInputLength)
                {
                    yield return full.AsSpan(s, Math.Min(MaxInputLength, full.Length - s)).ToArray();
                }
                yield break;
        }

        int[] Wrap(int[] ids, int start, int count, bool addBos, bool addEos)
        {
            var r = new int[count + (addBos ? 1 : 0) + (addEos ? 1 : 0)];
            int o = 0;
            if (addBos)
            {
                r[o++] = bos;
            }
            Array.Copy(ids, start, r, o, count);
            if (addEos)
            {
                r[^1] = eos;
            }
            return r;
        }
    }

    private float[][] RunBatched(List<int[]> sequences, ParallelOptions po)
    {
        var outputs = new float[sequences.Count][];
        int start = 0;
        while (start < sequences.Count)
        {
            int tokens = 0, end = start;
            while (end < sequences.Count && (end == start || tokens + sequences[end].Length <= TokensPerBatch))
            {
                tokens += sequences[end].Length;
                end++;
            }
            po.CancellationToken.ThrowIfCancellationRequested();
            var batch = _text.Forward(sequences.GetRange(start, end - start), po);
            Array.Copy(batch, 0, outputs, start, batch.Length);
            start = end;
        }
        return outputs;
    }

    /// <summary>Matryoshka truncation + optional L2 normalization of a raw 768-d embedding (copy).</summary>
    private float[] Finish(float[] raw)
    {
        var v = raw.AsSpan(0, _outputDimension).ToArray();
        if (Normalize)
        {
            Ops.L2NormalizeInPlace(v);
        }
        return v;
    }

    // ---------------------------------------------------------------------------------------------
    // Multimodal
    // ---------------------------------------------------------------------------------------------

    /// <summary>Embeds an image (440M / 740M bundles).</summary>
    public async Task<float[]> EncodeImageAsync(EmbeddingGemma2Image image, CancellationToken cancellationToken = default)
        => await EncodeContentAsync(new EmbeddingGemma2Content[] { image }, cancellationToken).ConfigureAwait(false);

    /// <summary>Embeds several images, one vector each.</summary>
    public async Task<float[][]> EncodeImagesAsync(IEnumerable<EmbeddingGemma2Image> images, CancellationToken cancellationToken = default)
    {
        var list = new List<float[]>();
        foreach (var image in images)
        {
            list.Add(await EncodeImageAsync(image, cancellationToken).ConfigureAwait(false));
        }
        return list.ToArray();
    }

    /// <summary>Embeds an audio clip (740M bundle).</summary>
    public async Task<float[]> EncodeAudioAsync(EmbeddingGemma2Audio audio, CancellationToken cancellationToken = default)
        => await EncodeContentAsync(new EmbeddingGemma2Content[] { audio }, cancellationToken).ConfigureAwait(false);

    /// <summary>Embeds an interleaved sequence of text / image / audio items into one vector, e.g.
    /// <c>EncodeContentAsync(new EmbeddingGemma2Content[] { "A product photo: ", image, " waterproof shoes" })</c>.</summary>
    public Task<float[]> EncodeContentAsync(IReadOnlyList<EmbeddingGemma2Content> contents, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(contents);
        var po = Options(cancellationToken);
        return Task.Run(() => Finish(EncodeContent(contents, po)), cancellationToken);
    }

    /// <summary>Builds the input-embedding sequence for interleaved content exactly like LiteRT-LM's
    /// <c>InsertSpecialTokens</c> + <c>ProcessAndCombineContents</c>, then runs the text transformer.</summary>
    private float[] EncodeContent(IReadOnlyList<EmbeddingGemma2Content> contents, ParallelOptions po)
    {
        int d = _text.HiddenSize;
        var rows = new List<float[]>();
        void AddTokens(IEnumerable<int> ids)
        {
            foreach (var id in ids)
            {
                var e = new float[d];
                _text.Embedder.Lookup(id, e);
                rows.Add(e);
            }
        }

        AddTokens(new[] { _tokenizer.BosId });
        foreach (var item in contents)
        {
            po.CancellationToken.ThrowIfCancellationRequested();
            switch (item)
            {
                case EmbeddingGemma2Text t:
                    AddTokens(_tokenizer.EncodeIdsWithoutSpecialTokens(t.Value));
                    break;
                case EmbeddingGemma2Image image:
                {
                    if (!SupportsImages)
                    {
                        throw new NotSupportedException($"This bundle ({Model}) has no vision encoder; use {EmbeddingGemma2Model.TextVision440M} or {EmbeddingGemma2Model.Multimodal740M}.");
                    }
                    var vision = _vision.Value;
                    if (_startOfImageId >= 0)
                    {
                        AddTokens(new[] { _startOfImageId });
                    }
                    var patches = ImagePreprocessor.Preprocess(image, _visionTokensPerImage, vision.PatchSize, vision.PoolingKernel);
                    var soft = vision.Encode(patches.Patches, patches.Positions, patches.NumPatches, po);
                    for (int t = 0; t < soft.Count; t++)
                    {
                        rows.Add(soft.Embeddings.AsSpan(t * soft.Cols, soft.Cols).ToArray());
                    }
                    rows.Add((float[])vision.EndOfVision.Clone());
                    break;
                }
                case EmbeddingGemma2Audio audio:
                {
                    if (!SupportsAudio)
                    {
                        throw new NotSupportedException($"This bundle ({Model}) has no audio encoder; use {EmbeddingGemma2Model.Multimodal740M}.");
                    }
                    var encoder = _audio.Value;
                    if (_startOfAudioId >= 0)
                    {
                        AddTokens(new[] { _startOfAudioId });
                    }
                    var soft = encoder.Encode(audio.Samples, out int count, po);
                    for (int t = 0; t < count; t++)
                    {
                        rows.Add(soft.AsSpan(t * encoder.OutputSize, encoder.OutputSize).ToArray());
                    }
                    rows.Add((float[])encoder.EndOfAudio.Clone());
                    break;
                }
                default:
                    throw new NotSupportedException($"Unsupported content item {item?.GetType().Name ?? "null"}.");
            }
        }
        AddTokens(new[] { _tokenizer.EosId });

        if (rows.Count > MaxInputLength)
        {
            switch (OverflowStrategy)
            {
                case InputOverflowStrategy.Error:
                    throw new ArgumentException($"Input sequence length ({rows.Count}) exceeds the maximum supported length ({MaxInputLength}).");
                case InputOverflowStrategy.Truncate:
                    rows.RemoveRange(MaxInputLength - 1, rows.Count - MaxInputLength);   // keep the trailing EOS
                    break;
                default:
                {
                    var sum = new float[_text.OutputSize];
                    int chunks = 0;
                    for (int s = 0; s < rows.Count; s += MaxInputLength)
                    {
                        var part = rows.GetRange(s, Math.Min(MaxInputLength, rows.Count - s));
                        System.Numerics.Tensors.TensorPrimitives.Add(sum, Run(part), sum);
                        chunks++;
                    }
                    System.Numerics.Tensors.TensorPrimitives.Divide(sum, chunks, sum);
                    return sum;
                }
            }
        }
        return Run(rows);

        float[] Run(List<float[]> seq)
        {
            var x = new float[seq.Count * d];
            for (int i = 0; i < seq.Count; i++)
            {
                seq[i].CopyTo(x, i * d);
            }
            return _text.ForwardEmbeddings(x, new[] { 0, seq.Count }, po)[0];
        }
    }

    // ---------------------------------------------------------------------------------------------
    // Chunking helpers (ISentenceEncoder)
    // ---------------------------------------------------------------------------------------------

    public List<string> ChunkTokens(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null, ChunkPrefix prefix = null)
        => BPEChunkAndEncodeHelpers.ChunkTokens(Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress, prefix);

    public List<AlignedString> ChunkTokensAligned(string text, int chunkLength = 500, int chunkOverlap = 100, int maxChunks = int.MaxValue, Action<float> reportProgress = null, ChunkPrefix prefix = null)
        => BPEChunkAndEncodeHelpers.ChunkTokensAligned(Tokenizer, text, chunkLength, chunkOverlap, maxChunks, reportProgress, prefix);

    public Task<EncodedChunk[]> ChunkAndEncodeAsync(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default, ChunkPrefix prefix = null)
        => BPEChunkAndEncodeHelpers.ChunkAndEncodeAsync(this, text, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken, prefix);

    public Task<EncodedChunkAligned[]> ChunkAndEncodeAlignedAsync(string text, int chunkLength = -1, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default, ChunkPrefix prefix = null)
        => BPEChunkAndEncodeHelpers.ChunkAndEncodeAlignedAsync(this, text, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken, prefix);

    public Task<TaggedEncodedChunk[]> ChunkAndEncodeTaggedAsync(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
        => BPEChunkAndEncodeHelpers.ChunkAndEncodeTaggedAsync(this, text, stripTags, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

    public Task<TaggedEncodedChunkAligned[]> ChunkAndEncodeTaggedAlignedAsync(string text, Func<string, TaggedChunk> stripTags, int chunkLength = 500, int chunkOverlap = 100, bool sequentially = true, int maxChunks = int.MaxValue, bool keepResultsOnCancellation = false, Action<float> reportProgress = null, CancellationToken cancellationToken = default)
        => BPEChunkAndEncodeHelpers.ChunkAndEncodeTaggedAlignedAsync(this, text, stripTags, chunkLength, chunkOverlap, sequentially, maxChunks, keepResultsOnCancellation, reportProgress, cancellationToken);

    public void Dispose() => _file.Dispose();
}
