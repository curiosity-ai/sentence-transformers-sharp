using System.Numerics.Tensors;
using BERTTokenizers.Base;
using SentenceTransformers.Harrier.Small.Pure.Tokenizer;
using SentenceTransformers.Training.Autograd;
using UID;

namespace SentenceTransformers.Harrier.Small.Pure.Training;

/// <summary>
/// A trainable/adaptable Gemma3 (Harrier Small) sentence encoder built on the autograd
/// <see cref="Gemma3LoraModel"/>. It is the counterpart to the optimized inference
/// <see cref="SentenceTransformers.Harrier.Small.Pure.SentenceEncoder"/>: slower, but it can apply a
/// trained <see cref="GemmaLoraAdapter"/> and expose the internals the LoRA trainer needs. Used both to
/// train an adapter and to evaluate it as a drop-in <see cref="ISentenceEncoder"/>.
/// </summary>
public sealed class Gemma3LoraEncoder : ISentenceEncoder
{
    private readonly Gemma3LoraModel  _model;
    private readonly GemmaLoraAdapter _adapter;
    private readonly GemmaLoraRuntime _rt;
    private readonly HarrierSmallPureTokenizer _tokenizer;
    private readonly VectorCache _vectorCache = new(16);

    public TokenizerBase Tokenizer => _tokenizer;
    public int MaxChunkLength => 8192;
    public int EmbeddingDimension => _model.Dimension;
    public GemmaLoraAdapter Adapter => _adapter;

    internal Gemma3LoraModel ModelInternal => _model;

    private Gemma3LoraEncoder(Gemma3LoraModel model, HarrierSmallPureTokenizer tokenizer, GemmaLoraAdapter adapter)
    {
        _model = model; _tokenizer = tokenizer; _adapter = adapter; _rt = adapter?.ToRuntime();
    }

    /// <summary>Downloads the Harrier Small weights (bf16 safetensors) if needed and builds a trainable encoder.</summary>
    public static async Task<Gemma3LoraEncoder> CreateAsync(GemmaLoraAdapter adapter = null, string weightsUrl = null, string weightsPath = null, Action<DownloadProgress> reportProgress = null, CancellationToken ct = default)
    {
        var path = weightsPath ?? Path.Combine(Path.GetTempPath(), "SentenceTransformers.Harrier.Small.Pure", "harrier-small.safetensors");
        Directory.CreateDirectory(Path.GetDirectoryName(path)!);
        await SentenceEncoder.DownloadFileAsync(weightsUrl ?? SentenceEncoder.DefaultWeightsUrl, path, reportProgress, ct).ConfigureAwait(false);
        var model = await Gemma3LoraModel.LoadAsync(path, ct).ConfigureAwait(false);
        return new Gemma3LoraEncoder(model, LoadTokenizer(), adapter);
    }

    /// <summary>Builds a trainable encoder from a safetensors file already on disk.</summary>
    public static async Task<Gemma3LoraEncoder> LoadAsync(string safetensorsPath, GemmaLoraAdapter adapter = null, CancellationToken ct = default)
        => new Gemma3LoraEncoder(await Gemma3LoraModel.LoadAsync(safetensorsPath, ct).ConfigureAwait(false), LoadTokenizer(), adapter);

    /// <summary>Returns a sibling encoder with a (different) adapter applied, sharing the frozen base model.</summary>
    public Gemma3LoraEncoder WithAdapter(GemmaLoraAdapter adapter) => new Gemma3LoraEncoder(_model, _tokenizer, adapter);

    private static HarrierSmallPureTokenizer LoadTokenizer()
    {
        var stream = typeof(SentenceEncoder).Assembly.GetManifestResourceStream("tokenizer.json")
                     ?? throw new FileNotFoundException("Embedded tokenizer.json not found for Harrier Small.");
        using (stream) return HarrierSmallPureTokenizer.FromStream(stream, 8192);
    }

    public async Task<float[][]> EncodeAsync(string[] sentences, CancellationToken cancellationToken = default)
    {
        if (sentences is null || sentences.Length == 0) return Array.Empty<float[]>();
        var results = new float[sentences.Length][];
        var keys = new UID128[sentences.Length];
        List<int> misses = null;
        for (int i = 0; i < sentences.Length; i++)
        {
            keys[i] = (sentences[i] ?? string.Empty).Hash128();
            if (_vectorCache.TryGet(keys[i], out var cached)) results[i] = cached;
            else (misses ??= new List<int>()).Add(i);
        }
        if (misses is null) return results;

        await Task.Run(() =>
        {
            foreach (int idx in misses)
            {
                cancellationToken.ThrowIfCancellationRequested();
                var v = EncodeOne(sentences[idx] ?? string.Empty);
                results[idx] = v;
                _vectorCache.Set(keys[idx], v);
            }
        }, cancellationToken).ConfigureAwait(false);
        return results;
    }

    private float[] EncodeOne(string sentence)
    {
        int[] ids = TokenIds(sentence);
        if (ids.Length == 0) return new float[EmbeddingDimension];
        var g = new Graph();
        var pooled = _model.Forward(g, ids, _rt);
        var z = (float[])pooled.Data.Clone();
        if (_adapter?.WhiteningMatrix != null) z = ApplyWhitening(z);
        float norm = MathF.Max(MathF.Sqrt(TensorPrimitives.Dot(z, z)), 1e-12f);
        TensorPrimitives.Divide(z, norm, z);
        return z;
    }

    internal int[] TokenIds(string sentence) => _tokenizer.EncodeIds(sentence ?? string.Empty);

    internal float[] PooledPreNormalize(string sentence)
    {
        int[] ids = TokenIds(sentence ?? string.Empty);
        if (ids.Length == 0) return new float[EmbeddingDimension];
        var g = new Graph();
        return (float[])_model.Forward(g, ids, _rt).Data.Clone();
    }

    private float[] ApplyWhitening(float[] z)
    {
        int d = EmbeddingDimension;
        var mean = _adapter.WhiteningMean; var W = _adapter.WhiteningMatrix;
        var centered = new float[d];
        for (int i = 0; i < d; i++) centered[i] = z[i] - mean[i];
        var outv = new float[d];
        for (int i = 0; i < d; i++) outv[i] = TensorPrimitives.Dot(W.AsSpan(i * d, d), centered);
        return outv;
    }

    public void Dispose() { }
}
