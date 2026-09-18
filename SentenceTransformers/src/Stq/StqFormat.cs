#nullable enable

namespace SentenceTransformers.Stq;

/// <summary>
/// Storage band for a single tensor inside an <c>.stq</c> file. The two ternary bands mirror the
/// packings PrismML ship for the Bonsai models (<c>PQ2_0</c> and <c>PTQ1_0</c>): identical ternary
/// values, identical per-group FP16 scales, differing only in how the trits are packed into bytes.
/// The float bands exist so a converter can leave small, quantization-sensitive tensors (norm
/// vectors, biases) alone in the same container.
/// </summary>
public enum StqBand
{
    /// <summary>Raw float32, 4 bytes/weight. Passthrough - no quantization.</summary>
    F32 = 0,

    /// <summary>Raw IEEE half, 2 bytes/weight. Passthrough - no quantization.</summary>
    F16 = 1,

    /// <summary>Raw bfloat16, 2 bytes/weight. Passthrough - no quantization.</summary>
    BF16 = 2,

    /// <summary>
    /// Ternary, 2 bits per weight: codes packed four to a byte, plus one FP16 scale per group.
    /// 32 + 2 = 34 bytes per 128-weight group = <b>2.125 bits/weight</b>. Cheapest to unpack
    /// (a table lookup per byte). Equivalent to Bonsai's <c>PQ2_0</c>.
    /// </summary>
    TQ2_0 = 3,

    /// <summary>
    /// Ternary, base-3 packed: five trits per byte (3^5 = 243 &lt; 256), plus one FP16 scale per
    /// group. 26 + 2 = 28 bytes per 128-weight group = <b>1.75 bits/weight</b>, within 0.03 bits of
    /// the 1.72 bits/weight information-theoretic floor for this format. 18% less weight traffic
    /// than <see cref="TQ2_0"/> at the cost of a slightly denser unpack. Equivalent to Bonsai's
    /// <c>PTQ1_0</c>.
    /// </summary>
    TQ1_0 = 4,

    /// <summary>
    /// Symmetric 4-bit, two codes per byte, plus one FP16 scale per group: codes in
    /// <c>[-8, +7]</c>. 4.5 bits/weight at group 32, 4.125 at group 128.
    ///
    /// <para>Not a Bonsai band - it exists because ternary turns out to be viable for this model's
    /// embedding table but not for its projections (see <c>QUANTIZATION.md</c> §4), so the useful
    /// checkpoint is a mixed one. Everything else about it matches the ternary bands: the same
    /// group scales, the same optional rotated basis, and the same kernel, since 4-bit codes are
    /// int8 operands just as trits are.</para>
    /// </summary>
    Q4_0 = 5,
}

/// <summary>
/// Constants and size arithmetic for the <c>.stq</c> ("SentenceTransformers Quantized") container.
///
/// <para><b>Why ternary.</b> A ternary weight takes one of three values <c>{-1, 0, +1}</c> scaled by
/// a value shared across a group of neighbouring weights: <c>w_i = s_g * t_i</c>. That carries
/// <c>log2(3) ~= 1.585</c> bits of information per weight; with one FP16 scale per group of 128 the
/// floor is <c>1.585 + 16/128 = 1.71</c> bits/weight, roughly a 9x reduction from FP16. The extra
/// zero state (over 1-bit binary) is what keeps quality intact at that rate.</para>
///
/// <para><b>Why a rotation.</b> Round-to-nearest ternary is hopeless on raw transformer weights:
/// a handful of outliers per group inflate the scale and flatten everything else to zero. Storing
/// the weights in a fixed rotated basis <c>R = H_n S / sqrt(n)</c> (a Walsh-Hadamard matrix times a
/// fixed diagonal of +/-1 signs) spreads every outlier across its whole block, leaving a
/// near-Gaussian distribution that ternarizes cleanly. The identity is exact:
/// <c>W x = (W R^T)(R x)</c>, so the file stores <c>W R^T</c> and the runtime applies <c>R</c> to
/// the activation with a fast Walsh-Hadamard transform - O(n log n), a fraction of a percent of the
/// matmul it precedes. See <see cref="HadamardRotation"/>.</para>
///
/// <para><b>File layout.</b>
/// <code>
///   [0  ..  4)   magic "STQ1"
///   [4  ..  8)   uint32 header length (little-endian)
///   [8  ..  8+n) UTF-8 JSON header
///   pad to a 64-byte boundary
///   data section: every blob referenced by the header, each 64-byte aligned
/// </code>
/// All byte ranges in the header are <c>[begin, end)</c> offsets relative to the start of the data
/// section, exactly as in <c>safetensors</c>. Codes and scales live in <b>separate</b> blobs rather
/// than interleaved blocks (GGUF's choice) so a kernel can stream a row's scales contiguously.</para>
/// </summary>
public static class StqFormat
{
    /// <summary>File magic. Bumped only for a breaking container change.</summary>
    public const string Magic = "STQ1";

    /// <summary>Header schema version. Readers reject anything newer than they understand.
    ///
    /// <para>Version 2 changed the <see cref="StqBand.Q4_0"/> nibble order from adjacent to
    /// half-a-group apart (see <c>StqPacking</c>), so a v1 file's 4-bit tensors decode to garbage
    /// under this build and are rejected rather than silently mis-read. The ternary bands are
    /// unchanged, so v1 files that use only those still load.</para></summary>
    public const int FormatVersion = 2;

    /// <summary>First version whose <see cref="StqBand.Q4_0"/> tensors use the split nibble
    /// layout.</summary>
    public const int SplitNibbleQ4Version = 2;

    /// <summary>Weights per scale group. 128 matches the Bonsai g128 packings and divides every
    /// Harrier input dimension (640, 1024, 2048).</summary>
    public const int DefaultGroupSize = 128;

    /// <summary>Alignment (bytes) of the data section and of every blob within it.</summary>
    public const int Alignment = 64;

    /// <summary>True when <paramref name="band"/> stores ternary codes.</summary>
    public static bool IsTernary(StqBand band) => band is StqBand.TQ2_0 or StqBand.TQ1_0;

    /// <summary>True when <paramref name="band"/> stores packed integer codes with group scales -
    /// the bands that go through the code/scale blob pair and the packed kernel, as opposed to the
    /// raw float bands.</summary>
    public static bool IsPacked(StqBand band) => IsTernary(band) || band is StqBand.Q4_0;

    /// <summary>Packed code bytes for one group of <paramref name="groupSize"/> weights, excluding
    /// the scale. TQ2_0 packs 4 codes/byte; TQ1_0 packs 5 trits/byte in base 3.</summary>
    public static int CodeBytesPerGroup(StqBand band, int groupSize) => band switch
    {
        StqBand.TQ2_0 => (groupSize + 3) / 4,
        StqBand.TQ1_0 => (groupSize + 4) / 5,
        StqBand.Q4_0  => (groupSize + 1) / 2,
        _ => throw new ArgumentOutOfRangeException(nameof(band), band, "Not a packed band."),
    };

    /// <summary>Codes packed into a single byte by <paramref name="band"/>.</summary>
    public static int CodesPerByte(StqBand band) => band switch
    {
        StqBand.TQ2_0 => 4,
        StqBand.TQ1_0 => 5,
        StqBand.Q4_0  => 2,
        _ => throw new ArgumentOutOfRangeException(nameof(band), band, "Not a packed band."),
    };

    /// <summary>
    /// The default group size for every band: 128, matching Bonsai's g128 packings.
    ///
    /// <para>4-bit used to default to 32, on the reasoning that a finer grouping is worth its extra
    /// 0.375 bits/weight. Measured, it is not: the quality difference is negligible, while the packed
    /// kernel reduces and rescales once per group, so quadrupling the group count costs real time -
    /// 8610 ms/iter at 32 against 5158 at 128 on the same weights. Fewer, larger groups it is.</para>
    /// </summary>
    public static int DefaultGroupSizeFor(StqBand band) => DefaultGroupSize;

    /// <summary>Total bits per weight including the FP16 group scale - the number quoted in the
    /// band docs (2.125 for TQ2_0, 1.75 for TQ1_0 at group 128).</summary>
    public static double BitsPerWeight(StqBand band, int groupSize = DefaultGroupSize) => band switch
    {
        StqBand.F32  => 32.0,
        StqBand.F16  => 16.0,
        StqBand.BF16 => 16.0,
        _ => (CodeBytesPerGroup(band, groupSize) + 2) * 8.0 / groupSize,
    };

    /// <summary>Bytes a raw (non-ternary) band uses per element.</summary>
    public static int RawBytesPerElement(StqBand band) => band switch
    {
        StqBand.F32  => 4,
        StqBand.F16  => 2,
        StqBand.BF16 => 2,
        _ => throw new ArgumentOutOfRangeException(nameof(band), band, "Not a raw band."),
    };

    /// <summary>Rounds <paramref name="offset"/> up to the next <see cref="Alignment"/> boundary.</summary>
    public static long AlignUp(long offset) => (offset + Alignment - 1) / Alignment * Alignment;

    /// <summary>Parses the lower-case band name used in the JSON header.</summary>
    public static StqBand ParseBand(string name) => name switch
    {
        "f32"   => StqBand.F32,
        "f16"   => StqBand.F16,
        "bf16"  => StqBand.BF16,
        "tq2_0" => StqBand.TQ2_0,
        "tq1_0" => StqBand.TQ1_0,
        "q4_0"  => StqBand.Q4_0,
        _ => throw new InvalidDataException($"Unknown storage band '{name}'."),
    };

    /// <summary>The lower-case band name used in the JSON header.</summary>
    public static string BandName(StqBand band) => band switch
    {
        StqBand.F32   => "f32",
        StqBand.F16   => "f16",
        StqBand.BF16  => "bf16",
        StqBand.TQ2_0 => "tq2_0",
        StqBand.TQ1_0 => "tq1_0",
        StqBand.Q4_0  => "q4_0",
        _ => throw new ArgumentOutOfRangeException(nameof(band), band, null),
    };
}
