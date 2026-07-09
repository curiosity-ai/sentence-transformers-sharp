using System.Buffers.Binary;
using SentenceTransformers.Training.Autograd;

namespace SentenceTransformers.Bert.Pure.Model;

/// <summary>Which of a BERT layer's linear projections carry a trainable low-rank adapter.</summary>
[Flags]
public enum LoraTargets
{
    None         = 0,
    Query        = 1 << 0,
    Key          = 1 << 1,
    Value        = 1 << 2,
    AttnOutput   = 1 << 3,
    Intermediate = 1 << 4,
    Output       = 1 << 5,

    /// <summary>All four attention projections (Q, K, V and the attention output dense).</summary>
    Attention    = Query | Key | Value | AttnOutput,

    /// <summary>Both feed-forward projections.</summary>
    Mlp          = Intermediate | Output,

    /// <summary>Every projection in the block.</summary>
    All          = Attention | Mlp,
}

/// <summary>
/// A trained (or freshly initialized) set of real weight-space LoRA parameters for a BERT encoder: for
/// every targeted linear in every layer a pair of low-rank factors <c>A ∈ ℝ^{r×in}</c>, <c>B ∈ ℝ^{out×r}</c>
/// whose scaled product <c>(α/r)·B·A</c> is added to the frozen weight. Optionally also carries a learned
/// output-centering <see cref="OutputBias"/> and a post-hoc <see cref="WhiteningMatrix">whitening</see>
/// transform. This is the serializable snapshot; live training/inference uses <see cref="LoraRuntime"/>.
///
/// <para>Following the LoRA convention <c>B</c> starts at zero and <c>A</c> at small Gaussian noise, so a
/// freshly created adapter reproduces the base encoder exactly and only departs from it as it trains.</para>
/// </summary>
public sealed class LoraAdapter
{
    internal const int LinearCount = 6; // Q, K, V, AttnOutput, Intermediate, Output

    private const uint   Magic   = 0x424C5241; // "ARLB" (Adapter, Real Lora, Bert)
    private const ushort Version = 1;

    public int         Dimension        { get; }
    public int         IntermediateSize { get; }
    public int         NumLayers        { get; }
    public int         Rank             { get; }
    public float       Alpha            { get; }
    public LoraTargets Targets          { get; }

    /// <summary>Effective residual scale <c>α / rank</c> applied to every low-rank delta.</summary>
    public float Scaling => Alpha / Rank;

    // A[layer*LinearCount + t] / B[...] are null when target t is not adapted.
    private readonly float[][] _a;
    private readonly float[][] _b;

    /// <summary>Optional learned per-dimension bias added to the pooled vector before normalization
    /// (mean-centering, which counteracts embedding anisotropy). Null when not used.</summary>
    public float[] OutputBias { get; internal set; }

    /// <summary>Optional post-hoc whitening: <c>u = normalize(W·(z − μ))</c> applied to the pooled vector.
    /// Both null when not used.</summary>
    public float[] WhiteningMean   { get; internal set; }
    public float[] WhiteningMatrix { get; internal set; } // row-major [Dimension, Dimension]

    internal LoraAdapter(int dimension, int intermediateSize, int numLayers, int rank, float alpha, LoraTargets targets, float[][] a, float[][] b)
    {
        Dimension        = dimension;
        IntermediateSize = intermediateSize;
        NumLayers        = numLayers;
        Rank             = rank;
        Alpha            = alpha;
        Targets          = targets;
        _a               = a;
        _b               = b;
    }

    /// <summary>Total number of trainable parameters across all adapted linears (plus output bias, if any).</summary>
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

    // (outDim, inDim) of linear t for this config.
    internal (int outDim, int inDim) LinearShape(int t) => t switch
    {
        0 or 1 or 2 or 3 => (Dimension, Dimension),          // Q, K, V, AttnOutput
        4                => (IntermediateSize, Dimension),   // Intermediate
        5                => (Dimension, IntermediateSize),   // Output
        _                => throw new ArgumentOutOfRangeException(nameof(t)),
    };

    internal static LoraTargets Flag(int t) => (LoraTargets)(1 << t);

    internal float[] AData(int layer, int t) => _a[layer * LinearCount + t];
    internal float[] BData(int layer, int t) => _b[layer * LinearCount + t];

    /// <summary>Creates a fresh adapter (B = 0, A ~ N(0, 1/in)) for the given base <paramref name="config"/>.</summary>
    public static LoraAdapter CreateInitialized(BertConfig config, int rank, float? alpha = null, LoraTargets targets = LoraTargets.Attention, int seed = 1)
    {
        if (rank <= 0) throw new ArgumentOutOfRangeException(nameof(rank));

        var a = new float[config.NumLayers * LinearCount][];
        var b = new float[config.NumLayers * LinearCount][];
        var adapter = new LoraAdapter(config.HiddenSize, config.IntermediateSize, config.NumLayers, rank, alpha ?? rank, targets, a, b);

        var rng = new Random(seed);
        for (int layer = 0; layer < config.NumLayers; layer++)
        {
            for (int t = 0; t < LinearCount; t++)
            {
                if ((targets & Flag(t)) == 0) continue;
                var (outDim, inDim) = adapter.LinearShape(t);
                var am = new float[rank * inDim];
                float std = 1f / MathF.Sqrt(inDim);
                for (int i = 0; i < am.Length; i++) am[i] = (float)NextGaussian(rng) * std;
                a[layer * LinearCount + t] = am;
                b[layer * LinearCount + t] = new float[outDim * rank]; // zeros
            }
        }
        return adapter;
    }

    // ----- serialization ---------------------------------------------------------------------------

    public void Save(Stream stream)
    {
        Span<byte> head = stackalloc byte[4 + 2 + 4 + 4 + 4 + 4 + 4 + 4 + 1 + 1];
        int p = 0;
        BinaryPrimitives.WriteUInt32LittleEndian(head[p..], Magic); p += 4;
        BinaryPrimitives.WriteUInt16LittleEndian(head[p..], Version); p += 2;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], Dimension); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], IntermediateSize); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], NumLayers); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], Rank); p += 4;
        BinaryPrimitives.WriteSingleLittleEndian(head[p..], Alpha); p += 4;
        BinaryPrimitives.WriteInt32LittleEndian(head[p..], (int)Targets); p += 4;
        head[p++] = (byte)(OutputBias != null ? 1 : 0);
        head[p++] = (byte)(WhiteningMatrix != null ? 1 : 0);
        stream.Write(head);

        for (int i = 0; i < _a.Length; i++)
        {
            if (_a[i] == null) continue;
            WriteFloats(stream, _a[i]);
            WriteFloats(stream, _b[i]);
        }
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

    public static LoraAdapter Load(Stream stream)
    {
        Span<byte> head = stackalloc byte[4 + 2 + 4 + 4 + 4 + 4 + 4 + 4 + 1 + 1];
        ReadExactly(stream, head);
        int p = 0;
        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(head[p..]); p += 4;
        if (magic != Magic) throw new InvalidDataException("Not a BERT LoRA adapter file (bad magic).");
        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(head[p..]); p += 2;
        if (version != Version) throw new InvalidDataException($"Unsupported BERT LoRA adapter version {version}.");
        int dim = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        int inter = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        int layers = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        int rank = BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        float alpha = BinaryPrimitives.ReadSingleLittleEndian(head[p..]); p += 4;
        var targets = (LoraTargets)BinaryPrimitives.ReadInt32LittleEndian(head[p..]); p += 4;
        bool hasBias = head[p++] == 1;
        bool hasWhiten = head[p++] == 1;

        var a = new float[layers * LinearCount][];
        var b = new float[layers * LinearCount][];
        var adapter = new LoraAdapter(dim, inter, layers, rank, alpha, targets, a, b);

        for (int layer = 0; layer < layers; layer++)
        {
            for (int t = 0; t < LinearCount; t++)
            {
                if ((targets & Flag(t)) == 0) continue;
                var (outDim, inDim) = adapter.LinearShape(t);
                a[layer * LinearCount + t] = ReadFloats(stream, rank * inDim);
                b[layer * LinearCount + t] = ReadFloats(stream, outDim * rank);
            }
        }
        if (hasBias)   adapter.OutputBias = ReadFloats(stream, dim);
        if (hasWhiten) { adapter.WhiteningMean = ReadFloats(stream, dim); adapter.WhiteningMatrix = ReadFloats(stream, dim * dim); }
        return adapter;
    }

    public static LoraAdapter Load(string path)
    {
        using var fs = File.OpenRead(path);
        return Load(fs);
    }

    // ----- runtime bridging ------------------------------------------------------------------------

    /// <summary>Builds live autograd tensors from this snapshot (for training or graph-based inference).</summary>
    internal LoraRuntime ToRuntime()
    {
        var rt = new LoraRuntime(NumLayers, Scaling, Targets);
        for (int layer = 0; layer < NumLayers; layer++)
        {
            for (int t = 0; t < LinearCount; t++)
            {
                if ((Targets & Flag(t)) == 0) continue;
                var (outDim, inDim) = LinearShape(t);
                rt.A[layer * LinearCount + t] = new Tensor((float[])AData(layer, t).Clone(), Rank, inDim);
                rt.B[layer * LinearCount + t] = new Tensor((float[])BData(layer, t).Clone(), outDim, Rank);
            }
        }
        if (OutputBias != null) rt.OutBias = new Tensor((float[])OutputBias.Clone(), 1, Dimension);
        return rt;
    }

    /// <summary>Copies trained tensor values from <paramref name="rt"/> back into this snapshot.</summary>
    internal void CopyFrom(LoraRuntime rt)
    {
        for (int i = 0; i < _a.Length; i++)
        {
            if (rt.A[i] == null) continue;
            Array.Copy(rt.A[i].Data, _a[i], _a[i].Length);
            Array.Copy(rt.B[i].Data, _b[i], _b[i].Length);
        }
        if (rt.OutBias != null)
        {
            OutputBias ??= new float[Dimension];
            Array.Copy(rt.OutBias.Data, OutputBias, Dimension);
        }
    }

    // ----- io helpers ------------------------------------------------------------------------------

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
        while (read < buf.Length)
        {
            int n = s.Read(buf[read..]);
            if (n == 0) throw new EndOfStreamException("Unexpected end of LoRA adapter stream.");
            read += n;
        }
    }

    private static double NextGaussian(Random rng)
    {
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}

/// <summary>Live low-rank tensors (with gradients) for one training/inference run, indexed the same way
/// as <see cref="LoraAdapter"/>. Frozen base weights live in <see cref="BertModel"/>, not here.</summary>
internal sealed class LoraRuntime
{
    public readonly LoraTargets Targets;
    public readonly float       Scale;
    public readonly Tensor[]    A; // [layer*LinearCount + t], null if untargeted
    public readonly Tensor[]    B;
    public Tensor               OutBias; // [1, dim] or null

    public LoraRuntime(int numLayers, float scale, LoraTargets targets)
    {
        Targets = targets;
        Scale   = scale;
        A = new Tensor[numLayers * LoraAdapter.LinearCount];
        B = new Tensor[numLayers * LoraAdapter.LinearCount];
    }

    public Tensor Aof(int layer, int t) => A[layer * LoraAdapter.LinearCount + t];
    public Tensor Bof(int layer, int t) => B[layer * LoraAdapter.LinearCount + t];

    /// <summary>Enumerates all trainable tensors (for the optimizer and grad clearing).</summary>
    public IEnumerable<Tensor> Parameters()
    {
        for (int i = 0; i < A.Length; i++) if (A[i] != null) { yield return A[i]; yield return B[i]; }
        if (OutBias != null) yield return OutBias;
    }
}
