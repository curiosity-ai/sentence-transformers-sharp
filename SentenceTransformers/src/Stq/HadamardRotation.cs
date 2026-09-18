#nullable enable

using System.Numerics;
using System.Numerics.Tensors;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

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

    private readonly float[] _signs;     // unpacked, Dim long, each +1 or -1
    private readonly float[] _signsNorm; // _signs scaled by _norm, so neither direction needs a second pass
    private readonly float _norm;        // 1 / sqrt(Block)

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
        _signsNorm = new float[dim];
        for (int i = 0; i < dim; i++)
        {
            _signs[i] = (signBits[i >> 3] & (1 << (i & 7))) != 0 ? -1f : 1f;
            _signsNorm[i] = _signs[i] * _norm;
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
    public void Apply(Span<float> v) => Apply(v, v);

    /// <summary>
    /// Applies <c>R v</c> from <paramref name="src"/> into <paramref name="dst"/>, which may be the
    /// same span. The out-of-place form exists because the caller almost always needs the input
    /// afterwards (it is the residual stream, or a sibling projection's activation) and would
    /// otherwise copy it first - and the sign multiply already writes every element, so it does the
    /// copy for free.
    /// </summary>
    public void Apply(ReadOnlySpan<float> src, Span<float> dst)
    {
        if (src.Length != Dim) throw new ArgumentException($"Expected a {Dim}-element vector, got {src.Length}.", nameof(src));
        if (dst.Length != Dim) throw new ArgumentException($"Expected a {Dim}-element destination, got {dst.Length}.", nameof(dst));

        // The 1/sqrt(n) rides along with the signs: scaling commutes with the (linear) butterfly, so
        // folding it in here costs nothing and saves a whole second pass over the vector.
        TensorPrimitives.Multiply(src, _signsNorm, dst);
        for (int off = 0; off < Dim; off += Block)
        {
            Fwht(dst.Slice(off, Block));
        }
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
        TensorPrimitives.Multiply(v, _signsNorm, v);
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

        // Stages with len < 8 only ever pair elements inside one aligned group of eight, so they are a
        // full 8-point Hadamard on each such group and can be done entirely in registers. Left to the
        // general loop below they were the whole cost of a rotation: at len = 1 a 128-element block
        // means 64 two-element butterflies, each constructing a pair of spans to add and subtract one
        // float. On a 640-wide activation the three sub-vector stages were doing three quarters of the
        // adds with none of the width.
        int start = 1;
        if (n >= 8)
        {
            Hadamard8(v);
            start = 8;
        }

        for (int len = start; len < n; len <<= 1)
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

    // Lane permutations for the three sub-vector stages, and the mask picking which lane of each pair
    // takes the difference rather than the sum.
    private static readonly Vector256<int> _swap1 = Vector256.Create(1, 0, 3, 2, 5, 4, 7, 6);
    private static readonly Vector256<int> _swap2 = Vector256.Create(2, 3, 0, 1, 6, 7, 4, 5);
    private static readonly Vector256<int> _swap4 = Vector256.Create(4, 5, 6, 7, 0, 1, 2, 3);
    private static readonly Vector256<int> _diff1 = Vector256.Create(0, -1, 0, -1, 0, -1, 0, -1);
    private static readonly Vector256<int> _diff2 = Vector256.Create(0, 0, -1, -1, 0, 0, -1, -1);
    private static readonly Vector256<int> _diff4 = Vector256.Create(0, 0, 0, 0, -1, -1, -1, -1);

    /// <summary>Runs the <c>len = 1, 2, 4</c> butterfly stages over every aligned group of eight.</summary>
    private static void Hadamard8(Span<float> v)
    {
        int n = v.Length;
        ref float p = ref MemoryMarshal.GetReference(v);

        if (Vector256.IsHardwareAccelerated)
        {
            for (int i = 0; i + 8 <= n; i += 8)
            {
                var x = Vector256.LoadUnsafe(ref p, (nuint)i);
                x = Butterfly(x, _swap1, _diff1);
                x = Butterfly(x, _swap2, _diff2);
                x = Butterfly(x, _swap4, _diff4);
                x.StoreUnsafe(ref p, (nuint)i);
            }
            return;
        }

        // Same three stages, unrolled and kept in locals: still scalar, but without a span per pair.
        for (int i = 0; i + 8 <= n; i += 8)
        {
            float a0 = Unsafe.Add(ref p, i),     a1 = Unsafe.Add(ref p, i + 1);
            float a2 = Unsafe.Add(ref p, i + 2), a3 = Unsafe.Add(ref p, i + 3);
            float a4 = Unsafe.Add(ref p, i + 4), a5 = Unsafe.Add(ref p, i + 5);
            float a6 = Unsafe.Add(ref p, i + 6), a7 = Unsafe.Add(ref p, i + 7);

            float b0 = a0 + a1, b1 = a0 - a1, b2 = a2 + a3, b3 = a2 - a3;
            float b4 = a4 + a5, b5 = a4 - a5, b6 = a6 + a7, b7 = a6 - a7;

            float c0 = b0 + b2, c1 = b1 + b3, c2 = b0 - b2, c3 = b1 - b3;
            float c4 = b4 + b6, c5 = b5 + b7, c6 = b4 - b6, c7 = b5 - b7;

            Unsafe.Add(ref p, i)     = c0 + c4; Unsafe.Add(ref p, i + 1) = c1 + c5;
            Unsafe.Add(ref p, i + 2) = c2 + c6; Unsafe.Add(ref p, i + 3) = c3 + c7;
            Unsafe.Add(ref p, i + 4) = c0 - c4; Unsafe.Add(ref p, i + 5) = c1 - c5;
            Unsafe.Add(ref p, i + 6) = c2 - c6; Unsafe.Add(ref p, i + 7) = c3 - c7;
        }
    }

    /// <summary>One butterfly stage: every lane pairs with the lane <paramref name="idx"/> names, and
    /// <paramref name="diff"/> says which of the two keeps the difference.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static Vector256<float> Butterfly(Vector256<float> x, Vector256<int> idx, Vector256<int> diff)
    {
        var t = Vector256.Shuffle(x, idx);
        return Vector256.ConditionalSelect(diff.AsSingle(), t - x, x + t);
    }

    private static bool IsPowerOfTwo(int x) => x > 0 && (x & (x - 1)) == 0;
}
