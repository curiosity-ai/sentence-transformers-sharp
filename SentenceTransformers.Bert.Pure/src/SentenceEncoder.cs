using System.Numerics.Tensors;
using BERTTokenizers.Base;
using SentenceTransformers.Bert.Pure.Model;
using SentenceTransformers.Training.Autograd;
using UID;

namespace SentenceTransformers.Bert.Pure;

/// <summary>
/// A 100% managed BERT sentence encoder (no ONNX Runtime) for the all-MiniLM-L6-v2 /
/// snowflake-arctic-embed-xs family. Loads the original HuggingFace safetensors weights and runs the
/// <see cref="BertModel">BertModel</see> forward pass in pure C#. Optionally applies a trained
/// <see cref="LoraAdapter"/> (real weight-space LoRA injected into the transformer's linears) so a
/// fine-tuned model is a drop-in <see cref="ISentenceEncoder"/>.
/// </summary>
public sealed class SentenceEncoder : ISentenceEncoder
{
    /// <summary>Default full-precision weights for all-MiniLM-L6-v2 (mean pooling, 256-token window).</summary>
    public const string MiniLMWeightsUrl = "https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/model.safetensors";

    /// <summary>Default full-precision weights for snowflake-arctic-embed-xs (CLS pooling, 512-token window).</summary>
    public const string ArcticXsWeightsUrl = "https://huggingface.co/Snowflake/snowflake-arctic-embed-xs/resolve/main/model.safetensors";

    private static readonly SemaphoreSlim _oneDownloadAtATime = new(1, 1);
    private static readonly HttpClient    _http = new() { Timeout = TimeSpan.FromHours(1) };

    private readonly BertModel    _model;
    private readonly LoraRuntime  _adapterRuntime;   // null = base model
    private readonly LoraAdapter  _adapter;          // for whitening arrays (null = none)
    private readonly VectorCache  _vectorCache = new(16);

    public TokenizerBase Tokenizer { get; }
    public int           MaxChunkLength { get; }

    /// <summary>Embedding dimension (384 for both supported models).</summary>
    public int EmbeddingDimension => _model.Config.HiddenSize;

    /// <summary>The applied adapter, or null when running the frozen base encoder.</summary>
    public LoraAdapter Adapter => _adapter;

    /// <summary>The frozen base model — used by the trainer to run training forward/backward passes.</summary>
    internal BertModel ModelInternal => _model;

    private SentenceEncoder(BertModel model, TokenizerBase tokenizer, int maxTokens, LoraAdapter adapter)
    {
        _model          = model;
        Tokenizer       = tokenizer;
        MaxChunkLength  = maxTokens;
        _adapter        = adapter;
        _adapterRuntime = adapter?.ToRuntime();
        Tokenizer.SetMaxTokens(maxTokens);
    }

    // ----- construction ---------------------------------------------------------------------------

    /// <summary>Creates the all-MiniLM-L6-v2 encoder, downloading its weights on first use.</summary>
    public static Task<SentenceEncoder> CreateMiniLMAsync(LoraAdapter adapter = null, string weightsPath = null, Action<DownloadProgress> reportProgress = null, CancellationToken ct = default)
        => CreateAsync(MiniLMWeightsUrl, BertConfig.MiniLM, 256, adapter, weightsPath, reportProgress, ct);

    /// <summary>Creates the snowflake-arctic-embed-xs encoder, downloading its weights on first use.</summary>
    public static Task<SentenceEncoder> CreateArcticXsAsync(LoraAdapter adapter = null, string weightsPath = null, Action<DownloadProgress> reportProgress = null, CancellationToken ct = default)
        => CreateAsync(ArcticXsWeightsUrl, BertConfig.ArcticXs, 512, adapter, weightsPath, reportProgress, ct);

    /// <summary>Creates an encoder for any BertModel-shaped checkpoint from a weights URL.</summary>
    public static async Task<SentenceEncoder> CreateAsync(string weightsUrl, BertConfig config, int maxTokens, LoraAdapter adapter = null, string weightsPath = null, Action<DownloadProgress> reportProgress = null, CancellationToken ct = default)
    {
        var path = weightsPath ?? Path.Combine(Path.GetTempPath(), "SentenceTransformers.Bert.Pure", CacheFileName(weightsUrl));
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await DownloadFileAsync(weightsUrl, path, reportProgress, ct).ConfigureAwait(false);
        BertModel model;
        try
        {
            model = await BertModel.LoadAsync(path, config, ct).ConfigureAwait(false);
        }
        catch (Exception ex) when (ex is InvalidDataException or ArgumentOutOfRangeException)
        {
            // A corrupt cached weights file (e.g. a truncated download left behind by an older
            // package version or an interrupted process) fails while parsing/reading tensors.
            // Delete it, download once more and retry before giving up.
            try { File.Delete(path); } catch { /* ignore */ }
            await DownloadFileAsync(weightsUrl, path, reportProgress, ct).ConfigureAwait(false);
            model = await BertModel.LoadAsync(path, config, ct).ConfigureAwait(false);
        }
        return new SentenceEncoder(model, new BertWordPieceTokenizer(), maxTokens, adapter);
    }

    /// <summary>Creates an encoder from an existing safetensors file on disk (no download).</summary>
    public static SentenceEncoder Load(string safetensorsPath, BertConfig config, int maxTokens, LoraAdapter adapter = null)
        => new SentenceEncoder(BertModel.Load(safetensorsPath, config), new BertWordPieceTokenizer(), maxTokens, adapter);

    /// <summary>Creates an encoder from the fp32 weights embedded in an ONNX graph (self-contained, no
    /// download and no ONNX Runtime — just reads the bytes). This is how the training CLI reuses the
    /// weights that already ship inside the MiniLM / ArcticXs packages.</summary>
    public static SentenceEncoder LoadFromOnnx(byte[] onnx, BertConfig config, int maxTokens, LoraAdapter adapter = null)
        => new SentenceEncoder(BertModel.LoadFromOnnx(onnx, config), new BertWordPieceTokenizer(), maxTokens, adapter);

    /// <summary>Returns a copy of this encoder with a (different) adapter applied, sharing the frozen base model.</summary>
    public SentenceEncoder WithAdapter(LoraAdapter adapter)
        => new SentenceEncoder(_model, Tokenizer, MaxChunkLength, adapter);

    // ----- encoding -------------------------------------------------------------------------------

    /// <inheritdoc/>
    public Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
    {
        if (sentences is null || sentences.Length == 0) return Task.FromResult(Array.Empty<float[]>());

        var results = new float[sentences.Length][];
        var keys    = new UID128[sentences.Length];
        List<int> misses = null;

        for (int i = 0; i < sentences.Length; i++)
        {
            keys[i] = (sentences[i] ?? string.Empty).Hash128();
            if (_vectorCache.TryGet(keys[i], out var cached)) results[i] = cached;
            else (misses ??= new List<int>()).Add(i);
        }

        // Encoding is CPU-bound managed work; run it on the caller's thread rather than faking async.
        if (misses != null)
        {
            foreach (int idx in misses)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var vec = EncodeOne(sentences[idx] ?? string.Empty);
                results[idx] = vec;
                _vectorCache.Set(keys[idx], vec);
            }
        }

        return Task.FromResult(results);
    }

    /// <inheritdoc/>
    public Task<float[]> EncodeAsync(string sentence, CancellationToken cancellationToken = default)
    {
        sentence ??= string.Empty;
        var key = sentence.Hash128();
        if (_vectorCache.TryGet(key, out var cached)) return Task.FromResult(cached);
        cancellationToken.ThrowIfCancellationRequested();
        var vec = EncodeOne(sentence);
        _vectorCache.Set(key, vec);
        return Task.FromResult(vec);
    }

    /// <summary>Runs the frozen (un-adapted) base encoder regardless of any applied adapter — used to
    /// compare a tuned adapter against its baseline.</summary>
    public float[] EncodeBase(string sentence) => EncodeOne(sentence ?? string.Empty, useAdapter: false);

    /// <summary>The adapted pooled vector before normalization/whitening — used when fitting whitening.</summary>
    internal float[] PooledPreNormalize(string sentence)
    {
        int[] ids = TokenIds(sentence ?? string.Empty);
        if (ids.Length == 0) return new float[EmbeddingDimension];
        var g = new Graph();
        var pooled = _model.Forward(g, ids, _adapterRuntime);
        return (float[])pooled.Data.Clone();
    }

    private float[] EncodeOne(string sentence, bool useAdapter = true)
    {
        int[] ids = TokenIds(sentence);
        if (ids.Length == 0) return new float[EmbeddingDimension];

        var g = new Graph();
        var pooled = _model.Forward(g, ids, useAdapter ? _adapterRuntime : null);

        var z = (float[])pooled.Data.Clone();
        if (useAdapter && _adapter?.WhiteningMatrix != null) z = ApplyWhitening(z);
        L2NormalizeInPlace(z);
        return z;
    }

    /// <summary>Tokenizes one sentence to input ids (with [CLS]/[SEP]); returns int[] with no padding.</summary>
    internal int[] TokenIds(string sentence)
    {
        var enc = Tokenizer.Encode(sentence);
        if (enc.Count == 0) return Array.Empty<int>();
        var input = enc[0].InputIds;
        var ids = new int[input.Length];
        for (int i = 0; i < input.Length; i++) ids[i] = (int)input[i];
        return ids;
    }

    private float[] ApplyWhitening(float[] z)
    {
        int d = EmbeddingDimension;
        var mean = _adapter.WhiteningMean;
        var W = _adapter.WhiteningMatrix; // [d,d] row-major
        var centered = new float[d];
        for (int i = 0; i < d; i++) centered[i] = z[i] - mean[i];
        var outv = new float[d];
        for (int i = 0; i < d; i++) outv[i] = TensorPrimitives.Dot(W.AsSpan(i * d, d), centered);
        return outv;
    }

    private static void L2NormalizeInPlace(float[] v)
    {
        float norm = MathF.Max(MathF.Sqrt(TensorPrimitives.Dot(v, v)), 1e-12f);
        TensorPrimitives.Divide(v, norm, v);
    }

    public void Dispose() { /* only managed weight arrays; released with the instance */ }

    // ----- download -------------------------------------------------------------------------------

    private static string CacheFileName(string url)
    {
        // Keep the file name readable but unique per URL so different models don't collide in the cache.
        string name = Path.GetFileName(new Uri(url).AbsolutePath);
        uint h = 2166136261u;
        foreach (char c in url) { h = (h ^ c) * 16777619u; }
        return $"{h:x8}-{name}";
    }

    private static async Task DownloadFileAsync(string url, string localPath, Action<DownloadProgress> reportProgress, CancellationToken ct)
    {
        if (File.Exists(localPath) && new FileInfo(localPath).Length > 0) return;
        if (!Uri.TryCreate(url, UriKind.Absolute, out var uri) || uri.Scheme is not ("https" or "http"))
            throw new InvalidOperationException($"Invalid weights URL: '{url}'.");

        await _oneDownloadAtATime.WaitAsync(ct).ConfigureAwait(false);
        try
        {
            if (File.Exists(localPath) && new FileInfo(localPath).Length > 0) return;
            string tmp = localPath + ".partial";
            using (var resp = await _http.GetAsync(url, HttpCompletionOption.ResponseHeadersRead, ct).ConfigureAwait(false))
            {
                resp.EnsureSuccessStatusCode();
                long? total = resp.Content.Headers.ContentLength;
                await using var src = await resp.Content.ReadAsStreamAsync(ct).ConfigureAwait(false);
                await using var dst = File.Create(tmp);
                var buffer = new byte[1 << 20];
                long read = 0; int n;
                while ((n = await src.ReadAsync(buffer, ct).ConfigureAwait(false)) > 0)
                {
                    await dst.WriteAsync(buffer.AsMemory(0, n), ct).ConfigureAwait(false);
                    read += n;
                    if (reportProgress != null && total is > 0)
                        reportProgress(new DownloadProgress(read, total, (float)read / total.Value, Path.GetFileName(localPath)));
                }
                if (total is long expected && read < expected)
                    throw new IOException($"Download of '{Path.GetFileName(localPath)}' ended prematurely: received {read} of {expected} bytes.");
            }
            File.Move(tmp, localPath, overwrite: true);
        }
        finally
        {
            _oneDownloadAtATime.Release();
        }
    }
}
