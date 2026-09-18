#nullable enable

using System.Numerics;
using System.Numerics.Tensors;

namespace SentenceTransformers.Stq;

/// <summary>
/// The fixed orthogonal basis change that makes ternary quantization work:
/// <code>
///   R = (1 / sqrt(n)) * H_n * S
/// </code>
/// where <c>H_n</c> is the Sylvester Walsh-Hadamard matrix of order <c>n</c> (a power of two) and
/// <c>S</c> is a fixed diagonal of +/-1 signs. <c>R</c> is orthogonal: <c>R^T R = S (H H / n) S = I</c>,
/// because <c>H_n H_n = n I</c> and <c>S^2 = I</c>.
///
/// <para>A dimension that is not a power of two is covered <b>blockwise</b>: the vector is split into
/// contiguous blocks of <see cref="Block"/> elements and each block gets its own transform, so <c>R</c>
/// is block-diagonal. Harrier's 640-wide hidden state uses five blocks of 128; its 2048-wide
/// MLP activations use two blocks of 1024. Each position has its own sign, so no two blocks share a
/// diagonal.</para>
///
/// <para><b>Why it is the same routine for weights and activations.</b> A linear layer is rewritten as
/// <c>W x = (W R^T)(R x)</c>. The converter stores <c>W R^T</c>; for a single weight row <c>w</c>
/// (a row vector) that is <c>w' = w R^T = w S H / sqrt(n)</c>, whose transpose is
/// <c>(H / sqrt(n)) S w^T</c> - exactly <see cref="Apply"/>, the same operation the runtime performs
/// on the activation. One routine, two uses, so a packing bug cannot silently cancel out.</para>
///
/// <para><see cref="ApplyInverse"/> computes <c>R^T v = S H v / sqrt(n)</c> and recovers an original
/// vector from a stored one. The token embedding table needs it: its rows are stored rotated (which is
/// what buys the quantization headroom on the single largest tensor in the model), and a lookup has no
/// activation to fold the rotation into, so the row is un-rotated after unpacking.</para>
///
/// <para>The signs are stored in the file as a bitmask (one bit per position, set = -1), not derived
/// from a seeded PRNG, so a reader never has to reproduce a generator to load a file.</para>
/// </summary>
public sealed class HadamardRotation
{
    /// <summary>Length of the vectors this rotation applies to.</summary>
    public int Dim { get; }

    /// <summary>Walsh-Hadamard block size: a power of two that divides <see cref="Dim"/>.</summary>
    public int Block { get; }

    /// <summary>Packed +/-1 diagonal, one bit per position, LSB first; a set bit means -1.</summary>
    public byte[] SignBits { get; }

    private readonly float[] _signs; // unpacked, Dim long, each +1 or -1
    private readonly float _norm;    // 1 / sqrt(Block)

    /// <summary>Wraps an existing sign mask (the file-loading path).</summary>
    public HadamardRotation(int dim, int block, byte[] signBits)
    {
        if (dim <= 0)                       throw new ArgumentOutOfRangeException(nameof(dim));
        if (block <= 0 || !IsPowerOfTwo(block)) throw new ArgumentException($"Rotation block {block} must be a power of two.", nameof(block));
        if (dim % block != 0)               throw new ArgumentException($"Rotation block {block} must divide dim {dim}.", nameof(block));
        if (signBits.Length < (dim + 7) / 8) throw new ArgumentException("Sign mask is too short for dim.", nameof(signBits));

        Dim = dim;
        Block = block;
        SignBits = signBits;
        _norm = 1f / MathF.Sqrt(block);
        _signs = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            _signs[i] = (signBits[i >> 3] & (1 << (i & 7))) != 0 ? -1f : 1f;
        }
    }

    /// <summary>
    /// Builds a rotation with a pseudo-random sign diagonal. <paramref name="seed"/> only has to be
    /// reproducible within one conversion run - the resulting mask is written to the file verbatim, so
    /// readers never re-run this generator. Uses SplitMix64 so the choice does not depend on the
    /// runtime's <see cref="Random"/> implementation.
    /// </summary>
    public static HadamardRotation Create(int dim, int block, ulong seed)
    {
        var bits = new byte[(dim + 7) / 8];
        ulong state = seed;
        for (int i = 0; i < dim; i++)
        {
            state += 0x9E3779B97F4A7C15UL;
            ulong z = state;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            z ^= z >> 31;
            if ((z & 1) != 0)
            {
                bits[i >> 3] |= (byte)(1 << (i & 7));
            }
        }
        return new HadamardRotation(dim, block, bits);
    }

    /// <summary>
    /// The largest power of two that divides <paramref name="dim"/>, capped at <paramref name="maxBlock"/>.
    /// 640 -&gt; 128, 1024 -&gt; 1024, 2048 -&gt; 1024 (at the default cap). Returns 1 when <paramref name="dim"/>
    /// is odd, in which case a rotation is pointless and the caller should skip it.
    /// </summary>
    public static int ChooseBlock(int dim, int maxBlock)
    {
        int block = 1;
        while (block * 2 <= maxBlock && dim % (block * 2) == 0)
        {
            block *= 2;
        }
        return block;
    }

    /// <summary>Applies <c>R v</c> in place: signs, then the Walsh-Hadamard butterfly, then 1/sqrt(n).
    /// Used on activations before a ternary matmul, and by the converter on each weight row.</summary>
    public void Apply(Span<float> v)
    {
        if (v.Length != Dim) throw new ArgumentException($"Expected a {Dim}-element vector, got {v.Length}.", nameof(v));

        for (int i = 0; i < Dim; i++)
        {
            v[i] *= _signs[i];
        }
        for (int off = 0; off < Dim; off += Block)
        {
            Fwht(v.Slice(off, Block));
        }
        TensorPrimitives.Multiply(v, _norm, v);
    }

    /// <summary>Applies <c>R^T v</c> in place: the butterfly, then signs, then 1/sqrt(n). Recovers an
    /// original vector from a rotated one (used after a token-embedding row lookup).</summary>
    public void ApplyInverse(Span<float> v)
    {
        if (v.Length != Dim) throw new ArgumentException($"Expected a {Dim}-element vector, got {v.Length}.", nameof(v));

        for (int off = 0; off < Dim; off += Block)
        {
            Fwht(v.Slice(off, Block));
        }
        for (int i = 0; i < Dim; i++)
        {
            v[i] *= _signs[i] * _norm;
        }
    }

    /// <summary>
    /// Unnormalized in-place fast Walsh-Hadamard transform (Sylvester ordering), computing
    /// <c>H_n v</c> in n*log2(n) add/subtract pairs. The caller applies the 1/sqrt(n) normalization.
    /// <paramref name="v"/>'s length must be a power of two.
    /// </summary>
    public static void Fwht(Span<float> v)
    {
        int n = v.Length;
        int w = Vector<float>.Count;
        for (int len = 1; len < n; len <<= 1)
        {
            for (int i = 0; i < n; i += len << 1)
            {
                var lo = v.Slice(i, len);
                var hi = v.Slice(i + len, len);
                int j = 0;
                if (len >= w)
                {
                    // lo' = lo + hi, hi' = lo - hi. Both halves are read before either is written,
                    // so the two loads must happen up front.
                    for (; j <= len - w; j += w)
                    {
                        var a = new Vector<float>(lo.Slice(j, w));
                        var b = new Vector<float>(hi.Slice(j, w));
                        (a + b).CopyTo(lo.Slice(j, w));
                        (a - b).CopyTo(hi.Slice(j, w));
                    }
                }
                for (; j < len; j++)
                {
                    float a = lo[j], b = hi[j];
                    lo[j] = a + b;
                    hi[j] = a - b;
                }
            }
        }
    }

    private static bool IsPowerOfTwo(int x) => x > 0 && (x & (x - 1)) == 0;
}
