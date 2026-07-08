using System.Buffers.Binary;
using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Training.Autograd;

namespace SentenceTransformers.Harrier.Small.Pure.Training;

/// <summary>Which of a Gemma3 layer's linear projections carry a trainable low-rank adapter.</summary>
[Flags]
public enum GemmaLoraTargets
{
    None       = 0,
    Query      = 1 << 0,
    Key        = 1 << 1,
    Value      = 1 << 2,
    AttnOutput = 1 << 3,
    Gate       = 1 << 4,
    Up         = 1 << 5,
    Down       = 1 << 6,

    /// <summary>All attention projections (q, k, v, o).</summary>
    Attention  = Query | Key | Value | AttnOutput,
    /// <summary>All feed-forward projections (gate, up, down).</summary>
    Mlp        = Gate | Up | Down,
    All        = Attention | Mlp,
}

/// <summary>
/// A trained (or freshly initialized) set of real weight-space LoRA parameters for the Gemma3 encoder:
/// per targeted linear per layer a pair of low-rank factors whose scaled product <c>(α/r)·B·A</c> is added
/// to the frozen weight. Optionally carries a learned output-centering bias and post-hoc whitening. This is
/// the serializable snapshot; live training/inference uses <see cref="GemmaLoraRuntime"/>.
/// </summary>
public sealed class GemmaLoraAdapter
{
    internal const int LinearCount = 7; // q, k, v, o, gate, up, down

    private const uint   Magic   = 0x414C5247; // "GRLA" (Gemma Real Lora Adapter)
    private const ushort Version = 1;

    public int Dimension        { get; }
    public int IntermediateSize { get; }
    public int QProjOut         { get; }
    public int KvProjOut        { get; }
    public int NumLayers        { get; }
    public int Rank             { get; }
    public float Alpha          { get; }
    public GemmaLoraTargets Targets { get; }
    public float Scaling => Alpha / Rank;

    private readonly float[][] _a;
    private readonly float[][] _b;

    public float[] OutputBias      { get; internal set; }
    public float[] WhiteningMean   { get; internal set; }
    public float[] WhiteningMatrix { get; internal set; }

    internal GemmaLoraAdapter(int dim, int inter, int qProjOut, int kvProjOut, int numLayers, int rank, float alpha, GemmaLoraTargets targets, float[][] a, float[][] b)
    {
        Dimension = dim; IntermediateSize = inter; QProjOut = qProjOut; KvProjOut = kvProjOut;
        NumLayers = numLayers; Rank = rank; Alpha = alpha; Targets = targets; _a = a; _b = b;
    }

    public long ParameterCount
    {
        get
        {
            long n = 0;
            foreach (var m in _a) if (m != null) n += m.Length;
            foreach (var m in _b) if (m != null) n += m.Length;
            if (OutputBias != null) n += OutputBias.Length;
            return n;
        }
    }

    internal (int outDim, int inDim) LinearShape(int t) => t switch
    {
        0 => (QProjOut, Dimension),          // q
        1 => (KvProjOut, Dimension),         // k
        2 => (KvProjOut, Dimension),         // v
        3 => (Dimension, QProjOut),          // o
        4 => (IntermediateSize, Dimension),  // gate
        5 => (IntermediateSize, Dimension),  // up
        6 => (Dimension, IntermediateSize),  // down
        _ => throw new ArgumentOutOfRangeException(nameof(t)),
    };

    internal static GemmaLoraTargets Flag(int t) => (GemmaLoraTargets)(1 << t);
    internal float[] AData(int layer, int t) => _a[layer * LinearCount + t];
    internal float[] BData(int layer, int t) => _b[layer * LinearCount + t];

    internal static GemmaLoraAdapter CreateInitialized(Gemma3Config cfg, int rank, float? alpha = null, GemmaLoraTargets targets = GemmaLoraTargets.Attention, int seed = 1)
    {
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));
        var a = new float[cfg.NumLayers * LinearCount][];
        var b = new float[cfg.NumLayers * LinearCount][];
        var adapter = new GemmaLoraAdapter(cfg.HiddenSize, cfg.IntermediateSize, cfg.QProjOut, cfg.KvProjOut, cfg.NumLayers, rank, alpha ?? rank, targets, a, b);
        var rng = new Random(seed);
        for (int layer = 0; layer < cfg.NumLayers; layer++)
            for (int t = 0; t < LinearCount; t++)
            {
                if ((targets & Flag(t)) == 0) continue;
                var (outDim, inDim) = adapter.LinearShape(t);
                var am = new float[rank * inDim];
                float std = 1f / MathF.Sqrt(inDim);
                for (int i = 0; i < am.Length; i++) am[i] = (float)NextGaussian(rng) * std;
                a[layer * LinearCount + t] = am;
                b[layer * LinearCount + t] = new float[outDim * rank];
            }
        return adapter;
    }

    internal GemmaLoraRuntime ToRuntime()
    {
        var rt = new GemmaLoraRuntime(NumLayers, Scaling, Targets);
        for (int layer = 0; layer < NumLayers; layer++)
            for (int t = 0; t < LinearCount; t++)
            {
                if ((Targets & Flag(t)) == 0) continue;
                var (outDim, inDim) = LinearShape(t);
                rt.A[layer * LinearCount + t] = new Tensor((float[])AData(layer, t).Clone(), Rank, inDim);
                rt.B[layer * LinearCount + t] = new Tensor((float[])BData(layer, t).Clone(), outDim, Rank);
            }
        if (OutputBias != null) rt.OutBias = new Tensor((float[])OutputBias.Clone(), 1, Dimension);
        return rt;
    }

    internal void CopyFrom(GemmaLoraRuntime rt)
    {
        for (int i = 0; i < _a.Length; i++)
        {
            if (rt.A[i] == null) continue;
            Array.Copy(rt.A[i].Data, _a[i], _a[i].Length);
            Array.Copy(rt.B[i].Data, _b[i], _b[i].Length);
        }
        if (rt.OutBias != null) { OutputBias ??= new float[Dimension]; Array.Copy(rt.OutBias.Data, OutputBias, Dimension); }
    }

    // ----- serialization ---------------------------------------------------------------------------

    public void Save(Stream stream)
    {
        Span<byte> head = stackalloc byte[4 + 2 + 4 * 6 + 4 + 4 + 1 + 1];
        int p = 0;
        BinaryPrimitives.WriteUInt32LittleEndian(head[p..], Magic); p += 4;
        BinaryPrimitives.WriteUInt16LittleEndian(head[p..], Version); p += 2;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], Dimension); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], IntermediateSize); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], QProjOut); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], KvProjOut); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], NumLayers); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], Rank); p += 4;
        BinaryPrimitives.WriteSingleLittleEndian(head[p..], Alpha); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], (int)Targets); p += 4;
        head[p++] = (byte)(OutputBias != null ? 1 : 0);
        head[p++] = (byte)(WhiteningMatrix != null ? 1 : 0);
        stream.Write(head);

        for (int i = 0; i < _a.Length; i++) { if (_a[i] == null) continue; WriteFloats(stream, _a[i]); WriteFloats(stream, _b[i]); }
        if (OutputBias != null) WriteFloats(stream, OutputBias);
        if (WhiteningMatrix != null) { WriteFloats(stream, WhiteningMean); WriteFloats(stream, WhiteningMatrix); }
    }

    public void Save(string path)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        using var fs = File.Create(path);
        Save(fs);
    }

    public static GemmaLoraAdapter Load(Stream stream)
    {
        Span<byte> head = stackalloc byte[4 + 2 + 4 * 6 + 4 + 4 + 1 + 1];
        ReadExactly(stream, head);
        int p = 0;
        if (BinaryPrimitives.ReadUInt32LittleEndian(head[p..]) != Magic) throw new InvalidDataException("Not a Gemma LoRA adapter file (bad magic).");
        p += 4;
        if (BinaryPrimitives.ReadUInt16LittleEndian(head[p..]) != Version) throw new InvalidDataException("Unsupported Gemma LoRA adapter version.");
        p += 2;
        int dim = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        int inter = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        int qo = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        int kv = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        int layers = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        int rank = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        float alpha = BinaryPrimitives.ReadSingleLittleEndian(head[p..]); p += 4;
        var targets = (GemmaLoraTargets)BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        bool hasBias = head[p++] == 1, hasWhiten = head[p++] == 1;

        var a = new float[layers * LinearCount][];
        var b = new float[layers * LinearCount][];
        var adapter = new GemmaLoraAdapter(dim, inter, qo, kv, layers, rank, alpha, targets, a, b);
        for (int layer = 0; layer < layers; layer++)
            for (int t = 0; t < LinearCount; t++)
            {
                if ((targets & Flag(t)) == 0) continue;
                var (outDim, inDim) = adapter.LinearShape(t);
                a[layer * LinearCount + t] = ReadFloats(stream, rank * inDim);
                b[layer * LinearCount + t] = ReadFloats(stream, outDim * rank);
            }
        if (hasBias) adapter.OutputBias = ReadFloats(stream, dim);
        if (hasWhiten) { adapter.WhiteningMean = ReadFloats(stream, dim); adapter.WhiteningMatrix = ReadFloats(stream, dim * dim); }
        return adapter;
    }

    public static GemmaLoraAdapter Load(string path) { using var fs = File.OpenRead(path); return Load(fs); }

    private static void WriteFloats(Stream s, float[] v)
    {
        var bytes = new byte[v.Length * 4];
        for (int i = 0; i < v.Length; i++) BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * 4), v[i]);
        s.Write(bytes);
    }
    private static float[] ReadFloats(Stream s, int count)
    {
        var bytes = new byte[count * 4];
        ReadExactly(s, bytes);
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * 4));
        return v;
    }
    private static void ReadExactly(Stream s, Span<byte> buf)
    {
        int read = 0;
        while (read < buf.Length) { int n = s.Read(buf[read..]); if (n == 0) throw new EndOfStreamException(); read += n; }
    }
    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble(), u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

/// <summary>Live low-rank tensors (with gradients) for a Gemma training/inference run.</summary>
internal sealed class GemmaLoraRuntime
{
    public readonly GemmaLoraTargets Targets;
    public readonly float Scale;
    public readonly Tensor[] A;
    public readonly Tensor[] B;
    public Tensor OutBias;

    public GemmaLoraRuntime(int numLayers, float scale, GemmaLoraTargets targets)
    {
        Targets = targets; Scale = scale;
        A = new Tensor[numLayers * GemmaLoraAdapter.LinearCount];
        B = new Tensor[numLayers * GemmaLoraAdapter.LinearCount];
    }

    public Tensor Aof(int layer, int t) => A[layer * GemmaLoraAdapter.LinearCount + t];
    public Tensor Bof(int layer, int t) => B[layer * GemmaLoraAdapter.LinearCount + t];

    public IEnumerable<Tensor> Parameters()
    {
        for (int i = 0; i < A.Length; i++) if (A[i] != null) { yield return A[i]; yield return B[i]; }
        if (OutBias != null) yield return OutBias;
    }
}
