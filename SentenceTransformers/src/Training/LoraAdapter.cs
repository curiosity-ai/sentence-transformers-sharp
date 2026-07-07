using System.Buffers.Binary;

namespace SentenceTransformers.Training;

/// <summary>
/// A LoRA-style low-rank residual adapter applied to a frozen encoder's pooled embedding.
///
/// <para>
/// Given a base embedding <c>e ∈ ℝ^d</c> produced by any <see cref="ISentenceEncoder"/>, the adapter
/// computes a low-rank residual and re-normalizes:
/// </para>
/// <code>
///   h = A · e                 (A ∈ ℝ^{r×d},  h ∈ ℝ^r)
///   z = e + (α / r) · B · h    (B ∈ ℝ^{d×r},  z ∈ ℝ^d)
///   u = z / ‖z‖               (final L2-normalized embedding)
/// </code>
/// <para>
/// Only the small matrices <c>A</c> and <c>B</c> are trainable — the base encoder stays frozen and is
/// used purely as a black box that turns text into <c>e</c>. Following the LoRA convention <c>B</c> is
/// initialized to zero (so a freshly created adapter reproduces the base encoder exactly, up to
/// re-normalization) and <c>A</c> to small Gaussian noise, giving a safe, well-behaved starting point.
/// The residual formulation keeps the adaptation additive and cheap: <c>O(r·d)</c> per vector,
/// independent of the base model's size, which is why the exact same adapter works for every model in
/// this library.
/// </para>
/// </summary>
public sealed class LoraAdapter
{
    private const uint   Magic   = 0x4C4F5241; // "LORA"
    private const ushort Version = 1;

    /// <summary>Embedding dimension the adapter operates on (must match the base encoder's output).</summary>
    public int Dimension { get; }

    /// <summary>Low-rank bottleneck size <c>r</c>. Smaller = fewer parameters, larger = more capacity.</summary>
    public int Rank { get; }

    /// <summary>LoRA <c>α</c> scaling numerator. The effective residual scale is <c>α / rank</c>.</summary>
    public float Alpha { get; }

    /// <summary>Effective residual scaling factor, <c>α / rank</c>.</summary>
    public float Scaling => Alpha / Rank;

    // Down-projection A (rank × dimension), row-major: A[j, i] = _a[j * Dimension + i].
    private readonly float[] _a;

    // Up-projection B (dimension × rank), row-major: B[k, j] = _b[k * Rank + j].
    private readonly float[] _b;

    /// <summary>Down-projection matrix <c>A</c> (<c>rank × dimension</c>, row-major). Exposed for the trainer/optimizer.</summary>
    public float[] A => _a;

    /// <summary>Up-projection matrix <c>B</c> (<c>dimension × rank</c>, row-major). Exposed for the trainer/optimizer.</summary>
    public float[] B => _b;

    /// <summary>Total number of trainable parameters (<c>rank · dimension · 2</c>).</summary>
    public int ParameterCount => _a.Length + _b.Length;

    private LoraAdapter(int dimension, int rank, float alpha, float[] a, float[] b)
    {
        Dimension = dimension;
        Rank      = rank;
        Alpha     = alpha;
        _a        = a;
        _b        = b;
    }

    /// <summary>
    /// Creates a fresh adapter with <c>B = 0</c> and <c>A</c> drawn from a small Gaussian, so it starts
    /// as a no-op residual and gradually learns a task-specific transform during training.
    /// </summary>
    /// <param name="dimension">Embedding dimension of the base encoder (e.g. 384 for MiniLM).</param>
    /// <param name="rank">Low-rank bottleneck size. Typical values are 8–64.</param>
    /// <param name="alpha">LoRA scaling numerator; the residual scale is <c>alpha / rank</c>. Defaults to <paramref name="rank"/>.</param>
    /// <param name="seed">Seed for the Gaussian initialization of <c>A</c>.</param>
    public static LoraAdapter CreateInitialized(int dimension, int rank, float? alpha = null, int seed = 1)
    {
        if (dimension <= 0) throw new ArgumentOutOfRangeException(nameof(dimension));
        if (rank      <= 0) throw new ArgumentOutOfRangeException(nameof(rank));

        var a   = new float[rank * dimension];
        var b   = new float[dimension * rank];
        var rng = new Random(seed);

        // Kaiming-ish small init for A; standard deviation 1/sqrt(dimension) keeps h at unit-ish scale.
        float std = 1f / MathF.Sqrt(dimension);
        for (int i = 0; i < a.Length; i++)
        {
            a[i] = (float)NextGaussian(rng) * std;
        }
        // B stays zero -> residual is zero at init -> adapter reproduces the base encoder.

        return new LoraAdapter(dimension, rank, alpha ?? rank, a, b);
    }

    /// <summary>
    /// Applies the adapter to a single base embedding, writing the L2-normalized adapted vector into
    /// <paramref name="destination"/>. <paramref name="baseEmbedding"/> and <paramref name="destination"/>
    /// may not overlap.
    /// </summary>
    public void Apply(ReadOnlySpan<float> baseEmbedding, Span<float> destination)
    {
        if (baseEmbedding.Length != Dimension) throw new ArgumentException($"Expected embedding of length {Dimension}, got {baseEmbedding.Length}.", nameof(baseEmbedding));
        if (destination.Length   != Dimension) throw new ArgumentException($"Destination must have length {Dimension}.", nameof(destination));

        Span<float> h = Rank <= 256 ? stackalloc float[Rank] : new float[Rank];
        Forward(baseEmbedding, h, destination, out _);
    }

    /// <summary>Convenience overload returning a freshly allocated adapted vector.</summary>
    public float[] Apply(ReadOnlySpan<float> baseEmbedding)
    {
        var result = new float[Dimension];
        Apply(baseEmbedding, result);
        return result;
    }

    /// <summary>
    /// Core forward pass shared by inference and training. Computes <c>h = A·e</c>, the pre-norm
    /// residual sum <c>z = e + scaling·B·h</c> (written into <paramref name="normalized"/>), and its
    /// norm, then normalizes in place. The trainer needs <paramref name="h"/> and
    /// <paramref name="norm"/> for backprop, so they are surfaced here.
    /// </summary>
    internal void Forward(ReadOnlySpan<float> e, Span<float> h, Span<float> normalized, out float norm, float eps = 1e-12f)
    {
        // h = A e
        for (int j = 0; j < Rank; j++)
        {
            float acc     = 0f;
            int   rowBase = j * Dimension;
            for (int i = 0; i < Dimension; i++)
            {
                acc += _a[rowBase + i] * e[i];
            }
            h[j] = acc;
        }

        // z = e + scaling * B h
        float scaling = Scaling;
        for (int k = 0; k < Dimension; k++)
        {
            float acc     = 0f;
            int   rowBase = k * Rank;
            for (int j = 0; j < Rank; j++)
            {
                acc += _b[rowBase + j] * h[j];
            }
            normalized[k] = e[k] + scaling * acc;
        }

        // u = z / ||z||
        float sumSq = 0f;
        for (int k = 0; k < Dimension; k++)
        {
            sumSq += normalized[k] * normalized[k];
        }
        norm = MathF.Max(MathF.Sqrt(sumSq), eps);

        float inv = 1f / norm;
        for (int k = 0; k < Dimension; k++)
        {
            normalized[k] *= inv;
        }
    }

    /// <summary>Serializes the adapter (metadata + weights) to a stream in a compact binary format.</summary>
    public void Save(Stream stream)
    {
        Span<byte> header = stackalloc byte[4 + 2 + 4 + 4 + 4];
        BinaryPrimitives.WriteUInt32LittleEndian(header[0..],  Magic);
        BinaryPrimitives.WriteUInt16LittleEndian(header[4..],  Version);
        BinaryPrimitives.WriteInt32LittleEndian(header[6..],   Dimension);
        BinaryPrimitives.WriteInt32LittleEndian(header[10..],  Rank);
        BinaryPrimitives.WriteSingleLittleEndian(header[14..], Alpha);
        stream.Write(header);

        WriteFloats(stream, _a);
        WriteFloats(stream, _b);
    }

    /// <summary>Serializes the adapter to a file, creating parent directories as needed.</summary>
    public void Save(string path)
    {
        var dir = Path.GetDirectoryName(Path.GetFullPath(path));
        if (!string.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
        using var fs = File.Create(path);
        Save(fs);
    }

    /// <summary>Deserializes an adapter previously written by <see cref="Save(Stream)"/>.</summary>
    public static LoraAdapter Load(Stream stream)
    {
        Span<byte> header = stackalloc byte[4 + 2 + 4 + 4 + 4];
        ReadExactly(stream, header);

        uint magic = BinaryPrimitives.ReadUInt32LittleEndian(header[0..]);
        if (magic != Magic) throw new InvalidDataException("Not a LoRA adapter file (bad magic).");

        ushort version = BinaryPrimitives.ReadUInt16LittleEndian(header[4..]);
        if (version != Version) throw new InvalidDataException($"Unsupported LoRA adapter version {version}.");

        int   dim   = BinaryPrimitives.ReadInt32LittleEndian(header[6..]);
        int   rank  = BinaryPrimitives.ReadInt32LittleEndian(header[10..]);
        float alpha = BinaryPrimitives.ReadSingleLittleEndian(header[14..]);

        if (dim <= 0 || rank <= 0) throw new InvalidDataException("Corrupt LoRA adapter dimensions.");

        var a = ReadFloats(stream, rank * dim);
        var b = ReadFloats(stream, dim * rank);

        return new LoraAdapter(dim, rank, alpha, a, b);
    }

    /// <summary>Deserializes an adapter from a file.</summary>
    public static LoraAdapter Load(string path)
    {
        using var fs = File.OpenRead(path);
        return Load(fs);
    }

    private static void WriteFloats(Stream stream, float[] values)
    {
        var bytes = new byte[values.Length * sizeof(float)];
        for (int i = 0; i < values.Length; i++)
        {
            BinaryPrimitives.WriteSingleLittleEndian(bytes.AsSpan(i * sizeof(float)), values[i]);
        }
        stream.Write(bytes);
    }

    private static float[] ReadFloats(Stream stream, int count)
    {
        var bytes = new byte[count * sizeof(float)];
        ReadExactly(stream, bytes);
        var values = new float[count];
        for (int i = 0; i < count; i++)
        {
            values[i] = BinaryPrimitives.ReadSingleLittleEndian(bytes.AsSpan(i * sizeof(float)));
        }
        return values;
    }

    private static void ReadExactly(Stream stream, Span<byte> buffer)
    {
        int read = 0;
        while (read < buffer.Length)
        {
            int n = stream.Read(buffer[read..]);
            if (n == 0) throw new EndOfStreamException("Unexpected end of LoRA adapter stream.");
            read += n;
        }
    }

    private static double NextGaussian(Random rng)
    {
        // Box-Muller transform.
        double u1 = 1.0 - rng.NextDouble();
        double u2 = 1.0 - rng.NextDouble();
        return Math.Sqrt(-2.0 * Math.Log(u1)) * Math.Cos(2.0 * Math.PI * u2);
    }
}
