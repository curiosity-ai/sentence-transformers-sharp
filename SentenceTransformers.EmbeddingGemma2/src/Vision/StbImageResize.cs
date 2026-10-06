// Portions of this file are a C# port of stb_image_resize2.h v2.18
// (https://github.com/nothings/stb), by Jeff Roberts (v2) and Jorge L Rodriguez, with Sean Barrett,
// placed in the public domain / dual-licensed under the MIT license. Only the code path used by
// LiteRT-LM's image preprocessor is ported:
//   stbir_resize(..., STBIR_RGB, STBIR_TYPE_UINT8_SRGB, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM)
// reproducing the output of the x86-64 SSE2 build bit for bit (same coefficients, same float
// summation order, same sRGB tables).

using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;

namespace SentenceTransformers.EmbeddingGemma2.Vision;

/// <summary>
/// Bit-exact managed port of the subset of stb_image_resize2 used by LiteRT-LM's
/// <c>stb_image_preprocessor.cc</c>: 3-channel 8-bit sRGB resizing with a Catmull-Rom filter and
/// clamped edges. The filter construction (<c>stbir__calculate_filters</c>), the gather / scatter
/// selection, the vertical-first heuristic and the float summation order of stb's SSE2 kernels are
/// all mirrored so the produced bytes match the native library.
/// </summary>
internal static class StbImageResize
{
    /// <summary><c>stbir__small_float</c>: 2^-120, the threshold under which coefficients are treated as zero.</summary>
    private const float SmallFloat = 1.0f / (1 << 20) / (1 << 20) / (1 << 20) / (1 << 20) / (1 << 20) / (1 << 20);

    /// <summary><c>STBIR__FLOAT_EMPTY_MARKER</c>: marks an unused ring buffer scanline in scatter mode.</summary>
    private const float FloatEmptyMarker = 3.0e+38F;

    /// <summary><c>STBIR_FORCE_GATHER_FILTER_SCANLINES_AMOUNT</c>: vertical downsamples with a filter
    /// footprint of at most this many scanlines still use the gather path.</summary>
    private const int ForceGatherFilterScanlinesAmount = 32;

    /// <summary>Lower bound of the "single weight of one" vertical shortcut (<c>1.0f-0.000001f</c>).</summary>
    private const float OneWeightLow = 1.0f - 0.000001f;

    /// <summary>Upper bound of the "single weight of one" vertical shortcut (<c>1.0f+0.000001f</c>).</summary>
    private const float OneWeightHigh = 1.0f + 0.000001f;

    /// <summary>Number of zeroed floats appended to every scanline buffer, so that the 4-wide horizontal
    /// kernel can read past the last contributing pixel (those lanes always have a zero weight).</summary>
    private const int ScanlinePadding = 16;

    /// <summary><c>stbir__srgb_uchar_to_linear_float</c>: sRGB byte to linear float decode table.</summary>
    private static readonly float[] SrgbUcharToLinearFloat =
    {
        0.000000f, 0.000304f, 0.000607f, 0.000911f, 0.001214f, 0.001518f, 0.001821f, 0.002125f, 0.002428f, 0.002732f, 0.003035f,
        0.003347f, 0.003677f, 0.004025f, 0.004391f, 0.004777f, 0.005182f, 0.005605f, 0.006049f, 0.006512f, 0.006995f, 0.007499f,
        0.008023f, 0.008568f, 0.009134f, 0.009721f, 0.010330f, 0.010960f, 0.011612f, 0.012286f, 0.012983f, 0.013702f, 0.014444f,
        0.015209f, 0.015996f, 0.016807f, 0.017642f, 0.018500f, 0.019382f, 0.020289f, 0.021219f, 0.022174f, 0.023153f, 0.024158f,
        0.025187f, 0.026241f, 0.027321f, 0.028426f, 0.029557f, 0.030713f, 0.031896f, 0.033105f, 0.034340f, 0.035601f, 0.036889f,
        0.038204f, 0.039546f, 0.040915f, 0.042311f, 0.043735f, 0.045186f, 0.046665f, 0.048172f, 0.049707f, 0.051269f, 0.052861f,
        0.054480f, 0.056128f, 0.057805f, 0.059511f, 0.061246f, 0.063010f, 0.064803f, 0.066626f, 0.068478f, 0.070360f, 0.072272f,
        0.074214f, 0.076185f, 0.078187f, 0.080220f, 0.082283f, 0.084376f, 0.086500f, 0.088656f, 0.090842f, 0.093059f, 0.095307f,
        0.097587f, 0.099899f, 0.102242f, 0.104616f, 0.107023f, 0.109462f, 0.111932f, 0.114435f, 0.116971f, 0.119538f, 0.122139f,
        0.124772f, 0.127438f, 0.130136f, 0.132868f, 0.135633f, 0.138432f, 0.141263f, 0.144128f, 0.147027f, 0.149960f, 0.152926f,
        0.155926f, 0.158961f, 0.162029f, 0.165132f, 0.168269f, 0.171441f, 0.174647f, 0.177888f, 0.181164f, 0.184475f, 0.187821f,
        0.191202f, 0.194618f, 0.198069f, 0.201556f, 0.205079f, 0.208637f, 0.212231f, 0.215861f, 0.219526f, 0.223228f, 0.226966f,
        0.230740f, 0.234551f, 0.238398f, 0.242281f, 0.246201f, 0.250158f, 0.254152f, 0.258183f, 0.262251f, 0.266356f, 0.270498f,
        0.274677f, 0.278894f, 0.283149f, 0.287441f, 0.291771f, 0.296138f, 0.300544f, 0.304987f, 0.309469f, 0.313989f, 0.318547f,
        0.323143f, 0.327778f, 0.332452f, 0.337164f, 0.341914f, 0.346704f, 0.351533f, 0.356400f, 0.361307f, 0.366253f, 0.371238f,
        0.376262f, 0.381326f, 0.386430f, 0.391573f, 0.396755f, 0.401978f, 0.407240f, 0.412543f, 0.417885f, 0.423268f, 0.428691f,
        0.434154f, 0.439657f, 0.445201f, 0.450786f, 0.456411f, 0.462077f, 0.467784f, 0.473532f, 0.479320f, 0.485150f, 0.491021f,
        0.496933f, 0.502887f, 0.508881f, 0.514918f, 0.520996f, 0.527115f, 0.533276f, 0.539480f, 0.545725f, 0.552011f, 0.558340f,
        0.564712f, 0.571125f, 0.577581f, 0.584078f, 0.590619f, 0.597202f, 0.603827f, 0.610496f, 0.617207f, 0.623960f, 0.630757f,
        0.637597f, 0.644480f, 0.651406f, 0.658375f, 0.665387f, 0.672443f, 0.679543f, 0.686685f, 0.693872f, 0.701102f, 0.708376f,
        0.715694f, 0.723055f, 0.730461f, 0.737911f, 0.745404f, 0.752942f, 0.760525f, 0.768151f, 0.775822f, 0.783538f, 0.791298f,
        0.799103f, 0.806952f, 0.814847f, 0.822786f, 0.830770f, 0.838799f, 0.846873f, 0.854993f, 0.863157f, 0.871367f, 0.879622f,
        0.887923f, 0.896269f, 0.904661f, 0.913099f, 0.921582f, 0.930111f, 0.938686f, 0.947307f, 0.955974f, 0.964686f, 0.973445f,
        0.982251f, 0.991102f, 1.0f
    };

    /// <summary><c>fp32_to_srgb8_tab4</c> (from https://gist.github.com/rygorous/2203834): packed
    /// bias / scale pairs for the piecewise-linear linear-float to sRGB byte encoder.</summary>
    private static readonly uint[] Fp32ToSrgb8Tab4 =
    {
        0x0073000d, 0x007a000d, 0x0080000d, 0x0087000d, 0x008d000d, 0x0094000d, 0x009a000d, 0x00a1000d,
        0x00a7001a, 0x00b4001a, 0x00c1001a, 0x00ce001a, 0x00da001a, 0x00e7001a, 0x00f4001a, 0x0101001a,
        0x010e0033, 0x01280033, 0x01410033, 0x015b0033, 0x01750033, 0x018f0033, 0x01a80033, 0x01c20033,
        0x01dc0067, 0x020f0067, 0x02430067, 0x02760067, 0x02aa0067, 0x02dd0067, 0x03110067, 0x03440067,
        0x037800ce, 0x03df00ce, 0x044600ce, 0x04ad00ce, 0x051400ce, 0x057b00c5, 0x05dd00bc, 0x063b00b5,
        0x06970158, 0x07420142, 0x07e30130, 0x087b0120, 0x090b0112, 0x09940106, 0x0a1700fc, 0x0a9500f2,
        0x0b0f01cb, 0x0bf401ae, 0x0ccb0195, 0x0d950180, 0x0e56016e, 0x0f0d015e, 0x0fbc0150, 0x10630143,
        0x11070264, 0x1238023e, 0x1357021d, 0x14660201, 0x156601e9, 0x165a01d3, 0x174401c0, 0x182401af,
        0x18fe0331, 0x1a9602fe, 0x1c1502d2, 0x1d7e02ad, 0x1ed4028d, 0x201a0270, 0x21520256, 0x227d0240,
        0x239f0443, 0x25c003fe, 0x27bf03c4, 0x29a10392, 0x2b6a0367, 0x2d1d0341, 0x2ebe031f, 0x304d0300,
        0x31d105b0, 0x34a80555, 0x37520507, 0x39d504c5, 0x3c37048b, 0x3e7c0458, 0x40a8042a, 0x42bd0401,
        0x44c20798, 0x488e071e, 0x4c1c06b6, 0x4f76065d, 0x52a50610, 0x55ac05cc, 0x5892058f, 0x5b590559,
        0x5e0c0a23, 0x631c0980, 0x67db08f6, 0x6c55087f, 0x70940818, 0x74a007bd, 0x787d076c, 0x7c330723,
    };

    /// <summary>The row of <c>stbir__compute_weights</c> for 3 channels (index 2), used by
    /// <c>stbir__should_do_vertical_first</c>. Eight resize classifications of four weights each.</summary>
    private static readonly float[] VerticalFirstWeights3Channels =
    {
        0.00000f, 0.53125f, 0.00000f, 0.03125f,
        0.06250f, 0.96875f, 0.00000f, 0.53125f,
        0.87500f, 0.18750f, 0.00000f, 0.93750f,
        0.00000f, 0.09375f, 1.00000f, 1.00000f,
        0.00000f, 0.53125f, 0.00000f, 0.03125f,
        0.03125f, 0.12500f, 1.00000f, 1.00000f,
        1.00000f, 1.00000f, 0.06250f, 1.00000f,
        0.00000f, 1.00000f, 0.00000f, 0.56250f,
    };

    /// <summary><c>stbir__scale_info</c>: the scale and rational form of one axis.</summary>
    private struct ScaleInfo
    {
        /// <summary>Input size along this axis.</summary>
        public int InputFullSize;

        /// <summary>Output size along this axis.</summary>
        public int OutputSubSize;

        /// <summary>Output / input.</summary>
        public float Scale;

        /// <summary>Input / output (computed in double, then rounded).</summary>
        public float InvScale;

        /// <summary>Starting shift in output pixel space (always 0 for a full-image resize).</summary>
        public float PixelShift;

        /// <summary>Whether the scale is an exact rational, enabling polyphase coefficient reuse.</summary>
        public bool ScaleIsRational;

        /// <summary>Numerator of the rational scale (the polyphase period, in output pixels).</summary>
        public uint ScaleNumerator;

        /// <summary>Denominator of the rational scale (the polyphase step, in input pixels).</summary>
        public uint ScaleDenominator;
    }

    /// <summary><c>stbir__sampler</c>: the filter (contributor ranges and coefficients) for one axis.</summary>
    private sealed class Sampler
    {
        /// <summary>Scale of this axis.</summary>
        public ScaleInfo Scale;

        /// <summary>0 = scatter (vertical only), 1 = gather with scale &gt;= 1, 2 = gather with scale &lt; 1.</summary>
        public int IsGather;
        /// <summary>Input pixels that can affect one output pixel (<c>filter_pixel_width</c>).</summary>
        public int FilterPixelWidth;

        /// <summary>Half of <see cref="FilterPixelWidth"/>: how far the filter overhangs either edge.</summary>
        public int FilterPixelMargin;

        /// <summary>Coefficients stored per contributor (becomes <see cref="Widest"/> after packing).</summary>
        public int CoefficientWidth;

        /// <summary>Output pixels when gathering, input pixels plus margins when scattering.</summary>
        public int NumContributors;

        /// <summary>First contributing pixel per contributor (<c>stbir__contributors.n0</c>).</summary>
        public int[] N0;

        /// <summary>Last contributing pixel per contributor (<c>stbir__contributors.n1</c>).</summary>
        public int[] N1;

        /// <summary>Coefficients, <see cref="CoefficientWidth"/> per contributor.</summary>
        public float[] Coefficients;

        /// <summary>Lowest first pixel over all contributors (<c>stbir__filter_extent_info.lowest</c>).</summary>
        public int Lowest;

        /// <summary>Highest last pixel over all contributors (<c>stbir__filter_extent_info.highest</c>).</summary>
        public int Highest;

        /// <summary>Widest single contributor (<c>stbir__filter_extent_info.widest</c>); selects the horizontal
        /// kernel and sizes the ring buffer.</summary>
        public int Widest;

        /// <summary>Coefficient width of the gather filter built before pivoting to scatter.</summary>
        public int PrescatterCoefficientWidth;

        /// <summary>Contributor count of the gather filter built before pivoting to scatter.</summary>
        public int PrescatterNumContributors;
    }

    /// <summary>
    /// Resizes packed 8-bit sRGB RGB pixels (row-major, 3 bytes/pixel, no row padding) with a Catmull-Rom
    /// filter and clamped edges, reproducing stb_image_resize2's
    /// <c>stbir_resize(..., STBIR_RGB, STBIR_TYPE_UINT8_SRGB, STBIR_EDGE_CLAMP, STBIR_FILTER_CATMULLROM)</c>.
    /// </summary>
    /// <param name="rgb">Source pixels, at least <c>width * height * 3</c> bytes.</param>
    /// <param name="width">Source width in pixels.</param>
    /// <param name="height">Source height in pixels.</param>
    /// <param name="newWidth">Destination width in pixels.</param>
    /// <param name="newHeight">Destination height in pixels.</param>
    /// <returns>The resized image, <c>newWidth * newHeight * 3</c> bytes.</returns>
    public static byte[] ResizeRgb(ReadOnlySpan<byte> rgb, int width, int height, int newWidth, int newHeight)
    {
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "Input image must have a positive width and height.");
        }
        if (newWidth <= 0 || newHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(newWidth), "Output image must have a positive width and height.");
        }
        if (rgb.Length < (long)width * height * 3)
        {
            throw new ArgumentException("Input buffer is smaller than width * height * 3 bytes.", nameof(rgb));
        }

        // stbir__perform_build: region transforms, samplers and conservative horizontal extents.
        var horizontal = CreateSampler(CalculateRegionTransform(newWidth, width), alwaysGather: true);
        var (conservativeN0, conservativeN1) = GetConservativeExtents(horizontal);
        var vertical = CreateSampler(CalculateRegionTransform(newHeight, height), alwaysGather: false);

        // stbir__alloc_internal_mem_and_build_samplers
        bool verticalFirst = ShouldDoVerticalFirst(horizontal.FilterPixelWidth, horizontal.Scale.Scale, horizontal.Scale.OutputSubSize,
                                                   vertical.FilterPixelWidth, vertical.Scale.Scale, vertical.Scale.OutputSubSize, vertical.IsGather);

        // are the two filters identical? (same kernel, support and edge mode always hold here)
        bool copyHorizontal = false;
        Sampler pivotSource = null;
        if (horizontal.Scale.OutputSubSize == vertical.Scale.OutputSubSize)
        {
            float diffScale = horizontal.Scale.Scale - vertical.Scale.Scale;
            float diffShift = horizontal.Scale.PixelShift - vertical.Scale.PixelShift;
            if (diffScale < 0.0f)
            {
                diffScale = -diffScale;
            }
            if (diffShift < 0.0f)
            {
                diffShift = -diffShift;
            }
            if (diffScale <= SmallFloat && diffShift <= SmallFloat)
            {
                if (horizontal.IsGather == vertical.IsGather)
                {
                    copyHorizontal = true;
                }
                else
                {
                    // vertical is scatter, horizontal is gather: pivot the horizontal coefficients
                    pivotSource = horizontal;
                }
            }
        }

        CalculateFilters(horizontal, null);
        horizontal.CoefficientWidth = PackCoefficients(horizontal, conservativeN0, conservativeN1);

        if (copyHorizontal)
        {
            vertical = horizontal;
        }
        else
        {
            CalculateFilters(vertical, pivotSource);
        }

        // info->ring_buffer_num_entries
        int ringEntries = vertical.Widest;
        if (vertical.IsGather == 0 && ringEntries > newHeight)
        {
            ringEntries = newHeight;
        }

        var output = new byte[(long)newWidth * newHeight * 3];
        var job = new ResizeJob(horizontal, vertical, verticalFirst, ringEntries, width, height, newWidth, newHeight);
        unsafe
        {
            fixed (byte* src = rgb)
            fixed (byte* dst = output)
            {
                if (vertical.IsGather != 0)
                {
                    job.VerticalGatherLoop(src, dst);
                }
                else
                {
                    job.VerticalScatterLoop(src, dst);
                }
            }
        }
        return output;
    }

    // ------------------------------------------------------------------------------------------------
    // Scalar helpers

    /// <summary><c>stbir_simd_floorf</c> (SSE2 variant): truncate, then subtract one if that rounded up.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float FloorF(float x)
    {
        float t = (int)x;
        return x < t ? t + -1.0f : t;
    }

    /// <summary><c>stbir_simd_ceilf</c> (SSE2 variant): truncate, then add one if that rounded down.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float CeilF(float x)
    {
        float t = (int)x;
        return t < x ? t + 1.0f : t;
    }

    /// <summary><c>stbir__filter_catmullrom</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float FilterCatmullRom(float x)
    {
        if (x < 0.0f)
        {
            x = -x;
        }
        if (x < 1.0f)
        {
            return 1.0f - x * x * (2.5f - 1.5f * x);
        }
        else if (x < 2.0f)
        {
            return 2.0f - x * (4.0f + x * (0.5f * x - 2.5f));
        }
        return 0.0f;
    }

    /// <summary><c>stbir__support_two</c>: Catmull-Rom has a support of two pixels.</summary>
    private const float CatmullRomSupport = 2.0f;

    /// <summary><c>stbir__edge_clamp_full</c> (via <c>stbir__edge_wrap</c>).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int EdgeClamp(int n, int max)
    {
        if (n < 0)
        {
            return 0;
        }
        if (n >= max)
        {
            return max - 1;
        }
        return n;
    }

    /// <summary><c>stbir__linear_to_srgb_uchar</c>: clamp to [2^-13, 1-eps] (NaN maps to 0), look up the
    /// bias/scale pair from the exponent and top mantissa bits, and interpolate with the next 8 bits.
    /// The SSE2 encoder (<c>stbir__min_max_shift20</c> / <c>stbir__linear_to_srgb_finish</c>) clamps with
    /// min/max instead of early-outs, but yields identical bytes for every input.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static unsafe byte LinearToSrgbUchar(float value, uint* fp32ToSrgb8Tab4)
    {
        const uint AlmostOne = 0x3f7fffff;
        const uint MinVal = (127 - 13) << 23;

        if (!(value > BitConverter.UInt32BitsToSingle(MinVal)))
        {
            return 0;
        }
        if (value > BitConverter.UInt32BitsToSingle(AlmostOne))
        {
            return 255;
        }

        uint bits = BitConverter.SingleToUInt32Bits(value);
        uint tab = fp32ToSrgb8Tab4[(bits - MinVal) >> 20];
        uint bias = (tab >> 16) << 9;
        uint scale = tab & 0xffff;
        uint t = (bits >> 12) & 0xff;
        return (byte)((bias + scale * t) >> 16);
    }

    // ------------------------------------------------------------------------------------------------
    // Scale and sampler setup

    /// <summary><c>stbir__calculate_region_transform</c> for a full input range mapped onto a full output range.</summary>
    private static ScaleInfo CalculateRegionTransform(int outputFullRange, int inputFullRange)
    {
        double inputS0 = 0.0, inputS1 = 1.0;
        double inputS = inputS1 - inputS0;
        double outputRange = outputFullRange;
        double inputRange = inputFullRange;
        double outputS = ((double)outputFullRange) / outputRange;
        double ratio = outputS / inputS;
        double scale = (outputRange / inputRange) * ratio;

        var info = new ScaleInfo
        {
            Scale = (float)scale,
            InvScale = (float)(1.0 / scale),
            // stbir__clip is a no-op for a full output range
            PixelShift = (float)(inputS0 * ratio * outputRange),
            InputFullSize = inputFullRange,
            OutputSubSize = outputFullRange,
        };
        info.ScaleIsRational = DoubleToRational(scale, (scale <= 1.0) ? (uint)outputFullRange : (uint)inputFullRange,
                                                out info.ScaleNumerator, out info.ScaleDenominator, scale >= 1.0);
        return info;
    }

    /// <summary><c>stbir__double_to_rational</c>: continued-fraction search for a rational with less than
    /// one float bit of error, limiting either the numerator or the denominator.</summary>
    private static bool DoubleToRational(double f, uint limit, out uint numer, out uint denom, bool limitDenom)
    {
        double err;
        ulong top, bot;
        ulong numerLast = 0;
        ulong denomLast = 1;
        ulong numerEstimate = 1;
        ulong denomEstimate = 0;

        // scale to past float error range
        top = (ulong)(f * (double)(1 << 25));
        bot = 1 << 25;

        // keep refining, but usually stops in a few loops - usually 5 for bad cases
        for (;;)
        {
            ulong est, temp;

            // hit limit, break out and do best full range estimate
            if ((limitDenom ? denomEstimate : numerEstimate) >= limit)
            {
                break;
            }

            // is the current error less than 1 bit of a float? if so, we're done
            if (denomEstimate != 0)
            {
                err = ((double)numerEstimate / (double)denomEstimate) - f;
                if (err < 0.0)
                {
                    err = -err;
                }
                if (err < (1.0 / (double)(1 << 24)))
                {
                    numer = (uint)numerEstimate;
                    denom = (uint)denomEstimate;
                    return true;
                }
            }

            // no more refinement bits left? break out and do full range estimate
            if (bot == 0)
            {
                break;
            }

            // gcd the estimate bits
            est = top / bot;
            temp = top % bot;
            top = bot;
            bot = temp;

            // move remainders
            temp = est * denomEstimate + denomLast;
            denomLast = denomEstimate;
            denomEstimate = temp;

            // move remainders
            temp = est * numerEstimate + numerLast;
            numerLast = numerEstimate;
            numerEstimate = temp;
        }

        // we didn't find anything good enough for float, use a full range estimate
        if (limitDenom)
        {
            numerEstimate = (ulong)(f * (double)limit + 0.5);
            denomEstimate = limit;
        }
        else
        {
            numerEstimate = limit;
            denomEstimate = (ulong)(((double)limit / f) + 0.5);
        }

        numer = (uint)numerEstimate;
        denom = (uint)denomEstimate;

        err = (denomEstimate != 0) ? (((double)(uint)numerEstimate / (double)(uint)denomEstimate) - f) : 1.0;
        if (err < 0.0)
        {
            err = -err;
        }
        return err < (1.0 / (double)(1 << 24));
    }

    /// <summary><c>stbir__get_filter_pixel_width</c>: input pixels that can affect one output pixel.</summary>
    private static int GetFilterPixelWidth(float scale)
    {
        if (scale >= 1.0f) // scale >= ( 1.0f - stbir__small_float ), which is 1.0f in float
        {
            return (int)CeilF(CatmullRomSupport * 2.0f);
        }
        return (int)CeilF(CatmullRomSupport * 2.0f / scale);
    }

    /// <summary><c>stbir__get_coefficient_width</c>: coefficients stored per contributor.</summary>
    private static int GetCoefficientWidth(float scale, int isGather)
    {
        switch (isGather)
        {
            case 1:
                return (int)CeilF(CatmullRomSupport * 2.0f);
            case 2:
                return (int)CeilF(CatmullRomSupport * 2.0f / scale);
            default:
                return (int)CeilF(CatmullRomSupport * 2.0f);
        }
    }

    /// <summary><c>stbir__set_sampler</c> with an explicit Catmull-Rom filter and clamped edges.</summary>
    private static Sampler CreateSampler(ScaleInfo scaleInfo, bool alwaysGather)
    {
        var samp = new Sampler { Scale = scaleInfo };
        samp.FilterPixelWidth = GetFilterPixelWidth(scaleInfo.Scale);

        // Gather is always better, but in extreme downsamples, you have to have most or all of the data in
        // memory. For horizontal, we always have all the pixels, so we always use gather (always_gather).
        // For vertical, we use gather if scaling up or if the filter footprint is small enough.
        samp.IsGather = 0;
        if (scaleInfo.Scale >= 1.0f)
        {
            samp.IsGather = 1;
        }
        else if (alwaysGather || samp.FilterPixelWidth <= ForceGatherFilterScanlinesAmount)
        {
            samp.IsGather = 2;
        }

        samp.CoefficientWidth = GetCoefficientWidth(scaleInfo.Scale, samp.IsGather);
        samp.FilterPixelMargin = samp.FilterPixelWidth / 2;

        // stbir__get_contributors
        samp.NumContributors = samp.IsGather != 0 ? scaleInfo.OutputSubSize : scaleInfo.InputFullSize + samp.FilterPixelMargin * 2;
        samp.N0 = new int[samp.NumContributors];
        samp.N1 = new int[samp.NumContributors];

        // extra STBIR_INPUT_CALLBACK_PADDING floats of padding (also holds the 8888 sentinel after packing)
        samp.Coefficients = new float[(long)samp.NumContributors * samp.CoefficientWidth + 3];

        if (samp.IsGather == 0)
        {
            samp.PrescatterCoefficientWidth = samp.FilterPixelWidth;
            samp.PrescatterNumContributors = scaleInfo.OutputSubSize;
        }
        return samp;
    }

    /// <summary><c>stbir__calculate_in_pixel_range</c> (non-wrap edge modes).</summary>
    private static void CalculateInPixelRange(out int firstPixel, out int lastPixel, float outPixelCenter, float outFilterRadius,
                                              float invScale, float outShift)
    {
        float outPixelInfluenceLowerbound = outPixelCenter - outFilterRadius;
        float outPixelInfluenceUpperbound = outPixelCenter + outFilterRadius;

        float inPixelInfluenceLowerbound = (outPixelInfluenceLowerbound + outShift) * invScale;
        float inPixelInfluenceUpperbound = (outPixelInfluenceUpperbound + outShift) * invScale;

        int first = (int)FloorF(inPixelInfluenceLowerbound + 0.5f);
        int last = (int)FloorF(inPixelInfluenceUpperbound - 0.5f);
        if (last < first)
        {
            last = first; // point sample mode can span a value *right* at 0.5, and cause these to cross
        }

        firstPixel = first;
        lastPixel = last;
    }

    /// <summary><c>stbir__calculate_out_pixel_range</c>.</summary>
    private static void CalculateOutPixelRange(out int firstPixel, out int lastPixel, float inPixelCenter, float inPixelsRadius,
                                               float scale, float outShift, int outSize)
    {
        float inPixelInfluenceLowerbound = inPixelCenter - inPixelsRadius;
        float inPixelInfluenceUpperbound = inPixelCenter + inPixelsRadius;
        float outPixelInfluenceLowerbound = inPixelInfluenceLowerbound * scale - outShift;
        float outPixelInfluenceUpperbound = inPixelInfluenceUpperbound * scale - outShift;
        int outFirstPixel = (int)FloorF(outPixelInfluenceLowerbound + 0.5f);
        int outLastPixel = (int)FloorF(outPixelInfluenceUpperbound - 0.5f);

        if (outFirstPixel < 0)
        {
            outFirstPixel = 0;
        }
        if (outLastPixel >= outSize)
        {
            outLastPixel = outSize - 1;
        }
        firstPixel = outFirstPixel;
        lastPixel = outLastPixel;
    }

    /// <summary><c>stbir__get_conservative_extents</c> for the (always gathering) horizontal sampler.</summary>
    private static (int N0, int N1) GetConservativeExtents(Sampler samp)
    {
        float scale = samp.Scale.Scale;
        float outShift = samp.Scale.PixelShift;
        int inputFullSize = samp.Scale.InputFullSize;
        float invScale = samp.Scale.InvScale;
        int n0, n1;

        if (samp.IsGather == 1)
        {
            float outFilterRadius = CatmullRomSupport * scale;

            CalculateInPixelRange(out int first, out _, 0.5f, outFilterRadius, invScale, outShift);
            n0 = first;
            CalculateInPixelRange(out _, out int last, ((float)(samp.Scale.OutputSubSize - 1)) + 0.5f, outFilterRadius, invScale, outShift);
            n1 = last;
        }
        else
        {
            // downsample gather, refine
            float inPixelsRadius = CatmullRomSupport * invScale;
            int filterPixelMargin = samp.FilterPixelMargin;
            int outputSubSize = samp.Scale.OutputSubSize;

            // get a conservative area of the input range
            CalculateInPixelRange(out int first, out _, 0, 0, invScale, outShift);
            n0 = first;
            CalculateInPixelRange(out _, out int last, (float)outputSubSize, 0, invScale, outShift);
            n1 = last;

            // now go through the margin to the start of area to find bottom
            int n = n0 + 1;
            int inputEnd = -filterPixelMargin;
            while (n >= inputEnd)
            {
                CalculateOutPixelRange(out int outFirst, out int outLast, ((float)n) + 0.5f, inPixelsRadius, scale, outShift, outputSubSize);
                if (outFirst > outLast)
                {
                    break;
                }
                if (outFirst < outputSubSize || outLast >= 0)
                {
                    n0 = n;
                }
                --n;
            }

            // now go through the end of the area through the margin to find top
            n = n1 - 1;
            inputEnd = n + 1 + filterPixelMargin;
            while (n <= inputEnd)
            {
                CalculateOutPixelRange(out int outFirst, out int outLast, ((float)n) + 0.5f, inPixelsRadius, scale, outShift, outputSubSize);
                if (outFirst > outLast)
                {
                    break;
                }
                if (outFirst < outputSubSize || outLast >= 0)
                {
                    n1 = n;
                }
                ++n;
            }
        }

        // for non-edge-wrap modes, we never read over the edge, so clamp
        if (n0 < 0)
        {
            n0 = 0;
        }
        if (n1 >= inputFullSize)
        {
            n1 = inputFullSize - 1;
        }
        return (n0, n1);
    }

    /// <summary><c>stbir__should_do_vertical_first</c>: cost model that decides whether the vertical pass
    /// runs on decoded input scanlines (vertical first) or on horizontally resampled ones.</summary>
    private static bool ShouldDoVerticalFirst(int horizontalFilterPixelWidth, float horizontalScale, int horizontalOutputSize,
                                              int verticalFilterPixelWidth, float verticalScale, int verticalOutputSize, int isGather)
    {
        int vClassification;

        // categorize the resize into buckets
        if (verticalOutputSize <= 4 || horizontalOutputSize <= 4)
        {
            vClassification = (verticalOutputSize < horizontalOutputSize) ? 6 : 7;
        }
        else if (isGather == 0 && (verticalOutputSize <= 16 || horizontalOutputSize <= 16))
        {
            vClassification = 4;
        }
        else if (verticalScale <= 1.0f)
        {
            vClassification = (isGather != 0) ? 1 : 0;
        }
        else if (verticalScale <= 2.0f)
        {
            vClassification = 2;
        }
        else if (verticalScale <= 3.0f)
        {
            vClassification = 3;
        }
        else
        {
            vClassification = 5; // everything bigger than 3x
        }

        var w = VerticalFirstWeights3Channels.AsSpan(vClassification * 4, 4);

        // float arithmetic (fp-contract off), widened to double only for storage in stb
        float hCost = (float)horizontalFilterPixelWidth * w[0] + horizontalScale * (float)verticalFilterPixelWidth * w[1];
        float vCost = (float)verticalFilterPixelWidth * w[2] + verticalScale * (float)horizontalFilterPixelWidth * w[3];

        return vCost <= hCost;
    }

    // ------------------------------------------------------------------------------------------------
    // Filter coefficient construction (stbir__calculate_filters and helpers)

    /// <summary><c>stbir__calculate_coefficients_for_gather_upsample</c>.</summary>
    private static void CalculateCoefficientsForGatherUpsample(float outFilterRadius, in ScaleInfo scaleInfo, int numContributors,
                                                               int[] cn0, int[] cn1, float[] coefficientGroup, int coefficientWidth)
    {
        float invScale = scaleInfo.InvScale;
        float outShift = scaleInfo.PixelShift;
        int numerator = (int)scaleInfo.ScaleNumerator;
        bool polyphase = scaleInfo.ScaleIsRational && numerator < numContributors;

        // Looping through out pixels
        int end = polyphase ? numerator : numContributors;
        for (int n = 0; n < end; n++)
        {
            int coeffs = n * coefficientWidth;
            float outPixelCenter = (float)n + 0.5f;
            float inCenterOfOut = (outPixelCenter + outShift) * invScale;

            CalculateInPixelRange(out int inFirstPixel, out int inLastPixel, outPixelCenter, outFilterRadius, invScale, outShift);

            // make sure we never generate a range larger than our precalculated coeff width
            if ((inLastPixel - inFirstPixel + 1) > coefficientWidth)
            {
                inLastPixel = inFirstPixel + coefficientWidth - 1;
            }

            int lastNonZero = -1;
            for (int i = 0; i <= inLastPixel - inFirstPixel; i++)
            {
                float inPixelCenter = (float)(i + inFirstPixel) + 0.5f;
                float coeff = FilterCatmullRom(inCenterOfOut - inPixelCenter);

                // kill denormals
                if (coeff < SmallFloat && coeff > -SmallFloat)
                {
                    if (i == 0) // if we're at the front, just eat zero contributors
                    {
                        ++inFirstPixel;
                        i--;
                        continue;
                    }
                    coeff = 0; // make sure is fully zero (should keep denormals away)
                }
                else
                {
                    lastNonZero = i;
                }

                coefficientGroup[coeffs + i] = coeff;
            }

            inLastPixel = lastNonZero + inFirstPixel; // kills trailing zeros
            cn0[n] = inFirstPixel;
            cn1[n] = inLastPixel;
        }
    }

    /// <summary><c>stbir__calculate_coefficients_for_gather_downsample</c>: walks the input pixels and
    /// scatters each one's weights into the output pixels it touches.</summary>
    private static void CalculateCoefficientsForGatherDownsample(int start, int end, float inPixelsRadius, in ScaleInfo scaleInfo,
                                                                 int coefficientWidth, int[] cn0, int[] cn1, float[] coefficientGroup)
    {
        int firstOutInited = -1;
        float scale = scaleInfo.Scale;
        float outShift = scaleInfo.PixelShift;
        int outSize = scaleInfo.OutputSubSize;
        int numerator = (int)scaleInfo.ScaleNumerator;
        bool polyphase = scaleInfo.ScaleIsRational && numerator < outSize;

        // Loop through the input pixels
        for (int inPixel = start; inPixel < end; inPixel++)
        {
            float inPixelCenter = (float)inPixel + 0.5f;
            float outCenterOfIn = inPixelCenter * scale - outShift;

            CalculateOutPixelRange(out int outFirstPixel, out int outLastPixel, inPixelCenter, inPixelsRadius, scale, outShift, outSize);

            if (outFirstPixel > outLastPixel)
            {
                continue;
            }

            // clamp or exit if we are using polyphase filtering, and the limit is up
            if (polyphase)
            {
                // when polyphase, you only have to do coeffs up to the numerator count
                if (outFirstPixel == numerator)
                {
                    break;
                }

                // don't do any extra work, clamp last pixel at numerator too
                if (outLastPixel >= numerator)
                {
                    outLastPixel = numerator - 1;
                }
            }

            for (int i = 0; i <= outLastPixel - outFirstPixel; i++)
            {
                float outPixelCenter = (float)(i + outFirstPixel) + 0.5f;
                float x = outPixelCenter - outCenterOfIn;
                float coeff = FilterCatmullRom(x) * scale;

                // kill the coeff if it's too small (avoid denormals)
                if (coeff < SmallFloat && coeff > -SmallFloat)
                {
                    coeff = 0.0f;
                }

                int o = i + outFirstPixel;
                int coeffs = o * coefficientWidth;

                // is this the first time this output pixel has been seen?  Init it.
                if (o > firstOutInited)
                {
                    firstOutInited = o;
                    cn0[o] = inPixel;
                    cn1[o] = inPixel;
                    coefficientGroup[coeffs] = coeff;
                }
                else
                {
                    // insert on end (always in order)
                    if (coefficientGroup[coeffs] == 0.0f) // if the first coefficent is zero, then zap it for this coeffs
                    {
                        cn0[o] = inPixel;
                    }
                    cn1[o] = inPixel;
                    coefficientGroup[coeffs + inPixel - cn0[o]] = coeff;
                }
            }
        }
    }

    /// <summary><c>stbir__insert_coeff</c>: accumulates a weight for <paramref name="newPixel"/> into a
    /// contributor, growing its range when it still fits in <paramref name="maxWidth"/>.</summary>
    private static void InsertCoeff(int[] cn0, int[] cn1, int contributor, float[] coeffs, int coeffBase, int newPixel, float newCoeff, int maxWidth)
    {
        int n0 = cn0[contributor];
        int n1 = cn1[contributor];
        if (n1 < n0) // this first clause should never happen, but handle in case
        {
            cn0[contributor] = cn1[contributor] = newPixel;
            coeffs[coeffBase] = newCoeff;
        }
        else if (newPixel <= n1) // before the end
        {
            if (newPixel < n0) // before the front?
            {
                if ((n1 - newPixel + 1) <= maxWidth)
                {
                    int o = n0 - newPixel;
                    for (int j = n1 - n0; j >= 0; j--)
                    {
                        coeffs[coeffBase + j + o] = coeffs[coeffBase + j];
                    }
                    for (int j = 1; j < o; j++)
                    {
                        coeffs[coeffBase + j] = 0;
                    }
                    coeffs[coeffBase] = newCoeff;
                    cn0[contributor] = newPixel;
                }
            }
            else
            {
                // add new weight to existing coeff if already there
                coeffs[coeffBase + newPixel - n0] += newCoeff;
            }
        }
        else
        {
            if ((newPixel - n0 + 1) <= maxWidth)
            {
                int e = newPixel - n0;
                for (int j = (n1 - n0) + 1; j < e; j++) // clear in-betweens coeffs if there are any
                {
                    coeffs[coeffBase + j] = 0;
                }
                coeffs[coeffBase + e] = newCoeff;
                cn1[contributor] = newPixel;
            }
        }
    }

    /// <summary><c>stbir__cleanup_gathered_coefficients</c>: renormalizes each contributor (in double), expands
    /// polyphase coefficients, folds out-of-range taps into the edge pixels (clamp), trims trailing zeros and
    /// records the lowest / highest / widest extents.</summary>
    private static void CleanupGatheredCoefficients(in ScaleInfo scaleInfo, int numContributors, int[] cn0, int[] cn1,
                                                    float[] coefficientGroup, int coefficientWidth,
                                                    out int lowestOut, out int highestOut, out int widestOut)
    {
        int inputSize = scaleInfo.InputFullSize;
        int inputLastN1 = inputSize - 1;
        int lowest = 0x7fffffff;
        int highest = -0x7fffffff;
        int widest = -1;
        int numerator = (int)scaleInfo.ScaleNumerator;
        int denominator = (int)scaleInfo.ScaleDenominator;
        bool polyphase = scaleInfo.ScaleIsRational && numerator < numContributors;

        // weight all the coeffs for each sample
        int end = polyphase ? numerator : numContributors;
        for (int n = 0; n < end; n++)
        {
            int coeffs = n * coefficientWidth;
            double totalFilter = 0;

            // add all contribs
            int e = cn1[n] - cn0[n];
            for (int i = 0; i <= e; i++)
            {
                totalFilter += (double)coefficientGroup[coeffs + i];
            }

            // rescale
            if (totalFilter < (double)SmallFloat && totalFilter > -(double)SmallFloat)
            {
                // all coeffs are extremely small, just zero it
                cn1[n] = cn0[n];
                coefficientGroup[coeffs] = 0.0f;
            }
            else
            {
                // if the total isn't 1.0, rescale everything ((1.0f -/+ stbir__small_float) is exactly 1.0f)
                if (totalFilter < 1.0 || totalFilter > 1.0)
                {
                    double filterScale = 1.0 / totalFilter;

                    // scale them all
                    for (int i = 0; i <= e; i++)
                    {
                        coefficientGroup[coeffs + i] = (float)(coefficientGroup[coeffs + i] * filterScale);
                    }
                }
            }
        }

        // if we have a rational for the scale, we can exploit the polyphaseness to not calculate
        // most of the coefficients, so we copy them here (stbir_overlapping_memcpy forward-copies,
        // which replicates the first numerator rows periodically)
        if (polyphase)
        {
            for (int n = numerator; n < numContributors; n++)
            {
                cn0[n] = cn0[n - numerator] + denominator;
                cn1[n] = cn1[n - numerator] + denominator;
            }
            int period = numerator * coefficientWidth;
            int total = numContributors * coefficientWidth;
            for (int j = period; j < total; j++)
            {
                coefficientGroup[j] = coefficientGroup[j - period];
            }
        }

        for (int n = 0; n < numContributors; n++)
        {
            int coeffs = n * coefficientWidth;

            // for clamp, calculate the true inbounds position and just add that to the existing weight

            // right hand side first
            if (cn1[n] > inputLastN1)
            {
                int start = cn0[n];
                int endi = cn1[n];
                cn1[n] = inputLastN1;
                for (int i = inputSize; i <= endi; i++)
                {
                    InsertCoeff(cn0, cn1, n, coefficientGroup, coeffs, EdgeClamp(i, inputSize), coefficientGroup[coeffs + i - start], coefficientWidth);
                }
            }

            // now check left hand edge
            if (cn0[n] < 0)
            {
                int c = coeffs - (cn0[n] + 1);

                // reinsert the coeffs with it clamped (insert accumulates, if the coeffs exist)
                for (int i = -1; i > cn0[n]; i--)
                {
                    InsertCoeff(cn0, cn1, n, coefficientGroup, coeffs, EdgeClamp(i, inputSize), coefficientGroup[c--], coefficientWidth);
                }
                int saveN0 = cn0[n];
                float saveN0Coeff = coefficientGroup[c]; // save it, since we didn't do the final one (i==n0)

                // now slide all the coeffs down (since we have accumulated them in the positive contribs) and reset the first contrib
                cn0[n] = 0;
                for (int i = 0; i <= cn1[n]; i++)
                {
                    coefficientGroup[coeffs + i] = coefficientGroup[coeffs + i - saveN0];
                }

                // now that we have shrunk down the contribs, we insert the first one safely
                InsertCoeff(cn0, cn1, n, coefficientGroup, coeffs, EdgeClamp(saveN0, inputSize), saveN0Coeff, coefficientWidth);
            }

            if (cn0[n] <= cn1[n])
            {
                int diff = cn1[n] - cn0[n] + 1;
                while (diff != 0 && coefficientGroup[coeffs + diff - 1] == 0.0f)
                {
                    --diff;
                }

                cn1[n] = cn0[n] + diff - 1;

                if (cn0[n] <= cn1[n])
                {
                    if (cn0[n] < lowest)
                    {
                        lowest = cn0[n];
                    }
                    if (cn1[n] > highest)
                    {
                        highest = cn1[n];
                    }
                    if (diff > widest)
                    {
                        widest = diff;
                    }
                }

                // re-zero out unused coefficients (if any)
                for (int i = diff; i < coefficientWidth; i++)
                {
                    coefficientGroup[coeffs + i] = 0.0f;
                }
            }
        }

        lowestOut = lowest;
        highestOut = highest;
        widestOut = widest;
    }

    /// <summary><c>stbir__calculate_filters</c>: builds gather coefficients (upsample or downsample), or for a
    /// vertical scatter builds downsample gather coefficients and pivots them into per-input-row weights.</summary>
    private static void CalculateFilters(Sampler samp, Sampler otherAxisForPivot)
    {
        float scale = samp.Scale.Scale;
        float invScale = samp.Scale.InvScale;
        int inputFullSize = samp.Scale.InputFullSize;

        if (samp.IsGather == 1)
        {
            // gather upsample
            float outPixelsRadius = CatmullRomSupport * scale;
            CalculateCoefficientsForGatherUpsample(outPixelsRadius, samp.Scale, samp.NumContributors, samp.N0, samp.N1, samp.Coefficients, samp.CoefficientWidth);
            CleanupGatheredCoefficients(samp.Scale, samp.NumContributors, samp.N0, samp.N1, samp.Coefficients, samp.CoefficientWidth,
                                        out samp.Lowest, out samp.Highest, out samp.Widest);
            return;
        }

        // scatter downsample (only on vertical) or gather downsample
        float inPixelsRadius = CatmullRomSupport * invScale;
        int filterPixelMargin = samp.FilterPixelMargin;
        int inputEnd = inputFullSize + filterPixelMargin;

        int[] gatherN0 = samp.N0;
        int[] gatherN1 = samp.N1;
        float[] gatherCoeffs = samp.Coefficients;
        int gatherCoefficientWidth = samp.CoefficientWidth;
        int gatherNumContributors = samp.NumContributors;

        if (samp.IsGather == 0)
        {
            // if this is a scatter, we do a downsample gather to get the coeffs, and then pivot after
            if (otherAxisForPivot != null)
            {
                // same filter as the horizontal: pivot directly from its (packed) coefficients
                gatherN0 = otherAxisForPivot.N0;
                gatherN1 = otherAxisForPivot.N1;
                gatherCoeffs = otherAxisForPivot.Coefficients;
                gatherCoefficientWidth = otherAxisForPivot.CoefficientWidth;
                gatherNumContributors = otherAxisForPivot.NumContributors;
                samp.Lowest = otherAxisForPivot.Lowest;
                samp.Highest = otherAxisForPivot.Highest;
                samp.Widest = otherAxisForPivot.Widest;
            }
            else
            {
                gatherCoefficientWidth = samp.PrescatterCoefficientWidth;
                gatherNumContributors = samp.PrescatterNumContributors;
                gatherN0 = new int[gatherNumContributors];
                gatherN1 = new int[gatherNumContributors];
                gatherCoeffs = new float[(long)gatherNumContributors * gatherCoefficientWidth + 4];
            }
        }

        if (samp.IsGather != 0 || otherAxisForPivot == null)
        {
            CalculateCoefficientsForGatherDownsample(-filterPixelMargin, inputEnd, inPixelsRadius, samp.Scale, gatherCoefficientWidth,
                                                     gatherN0, gatherN1, gatherCoeffs);
            CleanupGatheredCoefficients(samp.Scale, gatherNumContributors, gatherN0, gatherN1, gatherCoeffs, gatherCoefficientWidth,
                                        out samp.Lowest, out samp.Highest, out samp.Widest);
        }

        if (samp.IsGather != 0)
        {
            return;
        }

        // if this is a scatter (vertical only), then we need to pivot the coeffs
        int highestSet = (-filterPixelMargin) - 1;
        int scatterCoefficientWidth = samp.CoefficientWidth;
        int[] sn0 = samp.N0;
        int[] sn1 = samp.N1;
        float[] scatterCoeffs = samp.Coefficients;
        for (int n = 0; n < gatherNumContributors; n++)
        {
            int gn0 = gatherN0[n], gn1 = gatherN1[n];
            int gCoeffs = n * gatherCoefficientWidth;
            int scatterContributor = gn0 + filterPixelMargin;

            for (int k = gn0; k <= gn1; k++, scatterContributor++)
            {
                float gc = gatherCoeffs[gCoeffs++];
                int scatterBase = scatterContributor * scatterCoefficientWidth;

                // skip zero and denormals - must skip zeros to avoid adding coeffs beyond scatter_coefficient_width
                // (which happens when pivoting from horizontal, which might have dummy zeros)
                if (gc >= SmallFloat || gc <= -SmallFloat)
                {
                    if (k > highestSet || sn0[scatterContributor] > sn1[scatterContributor])
                    {
                        // if we are skipping over several contributors, we need to clear the skipped ones
                        for (int clear = highestSet + filterPixelMargin + 1; clear < scatterContributor; clear++)
                        {
                            sn0[clear] = 0;
                            sn1[clear] = -1;
                        }
                        sn0[scatterContributor] = n;
                        sn1[scatterContributor] = n;
                        scatterCoeffs[scatterBase] = gc;
                        highestSet = k;
                    }
                    else
                    {
                        InsertCoeff(sn0, sn1, scatterContributor, scatterCoeffs, scatterBase, n, gc, scatterCoefficientWidth);
                    }
                }
            }
        }

        // now clear any unset contribs
        for (int clear = highestSet + filterPixelMargin + 1; clear < samp.NumContributors; clear++)
        {
            sn0[clear] = 0;
            sn1[clear] = -1;
        }
    }

    /// <summary><c>stbir__pack_coefficients</c>: compacts the horizontal coefficients to a stride of
    /// <c>widest</c> and, near the right edge, moves contributor starts back (prepending zero weights) so
    /// the fixed-width kernels never read past the scanline. Returns the new coefficient width.</summary>
    private static int PackCoefficients(Sampler samp, int row0, int row1)
    {
        int numContributors = samp.NumContributors;
        int coefficientWidth = samp.CoefficientWidth;
        int widest = samp.Widest;
        float[] coefficients = samp.Coefficients;
        int[] cn0 = samp.N0;
        int[] cn1 = samp.N1;
        int rowEnd = row1 + 1;

        if (coefficientWidth != widest)
        {
            // forward copy, destination never ahead of source
            for (int n = 0; n < numContributors; n++)
            {
                int dst = n * widest;
                int src = n * coefficientWidth;
                for (int i = 0; i < widest; i++)
                {
                    coefficients[dst + i] = coefficients[src + i];
                }
            }
        }

        // some horizontal routines read one float off the end (which is then masked off), so put in a sentinel
        coefficients[widest * numContributors] = 8888.0f;

        // the minimum we might read for unrolled filters widths is 12. So, we need to make sure we never read
        // outside the decode buffer, by possibly moving the sample area back into the scanline, and putting
        // zeros weights first. we start on the right edge and check until we're well past the possible clip
        // area (2*widest).
        int contrib = numContributors - 1;
        int coeffs = widest * (numContributors - 1);

        // go until no chance of clipping (this is usually less than 8 lops)
        while (contrib >= 0 && (cn0[contrib] + widest * 2) >= rowEnd)
        {
            // might we clip??
            if ((cn0[contrib] + widest) > rowEnd)
            {
                int stopRange = widest;

                // if range is larger than 12, it will be handled by generic loops that can terminate on the exact
                // length of this contrib n1, instead of a fixed widest amount - so calculate this
                if (widest > 12)
                {
                    // how far will be read in the n_coeff loop (which depends on the widest count mod4);
                    int mod = widest & 3;
                    stopRange = (((cn1[contrib] - cn0[contrib] + 1) - mod + 3) & ~3) + mod;

                    // the n_coeff loops do a minimum amount of coeffs, so factor that in!
                    if (stopRange < (8 + mod))
                    {
                        stopRange = 8 + mod;
                    }
                }

                // now see if we still clip with the refined range
                if ((cn0[contrib] + stopRange) > rowEnd)
                {
                    int newN0 = rowEnd - stopRange;
                    int num = cn1[contrib] - cn0[contrib] + 1;
                    int backup = cn0[contrib] - newN0;
                    int fromCo = coeffs + num - 1;
                    int toCo = fromCo + backup;

                    // move the coeffs over
                    while (num != 0)
                    {
                        coefficients[toCo--] = coefficients[fromCo--];
                        --num;
                    }

                    // zero new positions
                    while (toCo >= coeffs)
                    {
                        coefficients[toCo--] = 0;
                    }

                    // set new start point
                    cn0[contrib] = newN0;
                }
            }
            --contrib;
            coeffs -= widest;
        }

        return widest;
    }

    // ------------------------------------------------------------------------------------------------
    // Resampling

    /// <summary>Per-call state of one resize (the parts of <c>stbir__info</c> / <c>stbir__per_split_info</c>
    /// used by a single-split resize). Instances are never shared, so <see cref="ResizeRgb"/> is thread-safe.</summary>
    private sealed unsafe class ResizeJob
    {
        private readonly Sampler _horizontal;
        private readonly Sampler _vertical;
        private readonly bool _verticalFirst;
        private readonly int _ringEntries;
        private readonly int _inputWidth;
        private readonly int _inputHeight;
        private readonly int _outputWidth;
        private readonly int _outputHeight;

        /// <summary>Creates the per-call resize state.</summary>
        public ResizeJob(Sampler horizontal, Sampler vertical, bool verticalFirst, int ringEntries,
                         int inputWidth, int inputHeight, int outputWidth, int outputHeight)
        {
            _horizontal = horizontal;
            _vertical = vertical;
            _verticalFirst = verticalFirst;
            _ringEntries = ringEntries;
            _inputWidth = inputWidth;
            _inputHeight = inputHeight;
            _outputWidth = outputWidth;
            _outputHeight = outputHeight;
        }

        /// <summary>Floats per ring buffer scanline: a decoded input scanline when vertical-first, otherwise a
        /// horizontally resampled one.</summary>
        private int RingRowFloats => _verticalFirst ? _inputWidth * 3 : _outputWidth * 3;

        /// <summary><c>stbir__decode_scanline</c> + <c>stbir__decode_uint8_srgb</c>: decodes (edge-clamped)
        /// input row <paramref name="n"/> to linear floats. The whole row is decoded; positions outside stb's
        /// decoded spans are only ever multiplied by zero weights.</summary>
        private void DecodeScanline(byte* input, int n, float* decode)
        {
            int row = EdgeClamp(n, _inputHeight);
            int count = _inputWidth * 3;
            byte* src = input + (long)row * count;
            fixed (float* table = SrgbUcharToLinearFloat)
            {
                for (int i = 0; i < count; i++)
                {
                    decode[i] = table[src[i]];
                }
            }
        }

        /// <summary><c>stbir__encode_scanline</c> + <c>stbir__encode_uint8_srgb</c>.</summary>
        private void EncodeScanline(float* encode, byte* outputRow)
        {
            int count = _outputWidth * 3;
            fixed (uint* table = Fp32ToSrgb8Tab4)
            {
                for (int i = 0; i < count; i++)
                {
                    outputRow[i] = LinearToSrgbUchar(encode[i], table);
                }
            }
        }

        /// <summary><c>stbir__resample_horizontal_gather</c> with the 3-channel SSE2 gather kernels
        /// (<c>stbir__horizontal_gather_3_channels_with_*</c>).
        /// <para>
        /// With up to 3 coefficients the taps are summed left to right. From 4 coefficients on, stb keeps three
        /// accumulators <c>tot0 = [R0 G0 B0 R1]</c>, <c>tot1 = [G1 B1 R2 G2]</c>, <c>tot2 = [B2 R3 G3 B3]</c> over
        /// groups of 4 taps (so tap <c>i</c> lands in lane-group <c>i &amp; 3</c>), and finally combines each
        /// channel as <c>(S0 + S2) + (S1 + S3)</c>. That exact order is reproduced here with
        /// <see cref="Vector128{T}"/> arithmetic (no FMA contraction, as in stb).
        /// </para>
        /// </summary>
        private void HorizontalGather(float* decode, float* output)
        {
            int outputCount = _outputWidth;
            int widest = _horizontal.CoefficientWidth;
            int[] cn0 = _horizontal.N0;
            int[] cn1 = _horizontal.N1;

            fixed (float* coefficients = _horizontal.Coefficients)
            {
                if (widest <= 3)
                {
                    for (int x = 0; x < outputCount; x++)
                    {
                        float* d = decode + cn0[x] * 3;
                        float* c = coefficients + x * widest;
                        float c0 = c[0];
                        float r = d[0] * c0;
                        float g = d[1] * c0;
                        float b = d[2] * c0;
                        if (widest >= 2)
                        {
                            float c1 = c[1];
                            r += d[3] * c1;
                            g += d[4] * c1;
                            b += d[5] * c1;
                            if (widest == 3)
                            {
                                float c2 = c[2];
                                r += d[6] * c2;
                                g += d[7] * c2;
                                b += d[8] * c2;
                            }
                        }
                        float* o = output + x * 3;
                        o[0] = r;
                        o[1] = g;
                        o[2] = b;
                    }
                    return;
                }

                for (int x = 0; x < outputCount; x++)
                {
                    int n0 = cn0[x];
                    // taps past n1 carry zero weights in stb (only add +-0), so only the real taps are summed
                    int count = cn1[x] - n0 + 1;
                    float* d = decode + n0 * 3;
                    float* c = coefficients + x * widest;

                    // stbir__4_coeff_start (always 4 taps; widest >= 4 so taps count..3 are stored zeros)
                    Vector128<float> cs = Vector128.Load(c);
                    Vector128<float> tot0 = Vector128.Load(d) * Shuffle0001(cs);
                    Vector128<float> tot1 = Vector128.Load(d + 4) * Shuffle1122(cs);
                    Vector128<float> tot2 = Vector128.Load(d + 8) * Shuffle2333(cs);

                    // stbir__4_coeff_continue_from_4 / remnants
                    int i = 4;
                    for (; i + 4 <= count; i += 4)
                    {
                        cs = Vector128.Load(c + i);
                        float* di = d + i * 3;
                        tot0 += Vector128.Load(di) * Shuffle0001(cs);
                        tot1 += Vector128.Load(di + 4) * Shuffle1122(cs);
                        tot2 += Vector128.Load(di + 8) * Shuffle2333(cs);
                    }
                    int remaining = count - i;
                    if (remaining > 0)
                    {
                        float* di = d + i * 3;
                        cs = Vector128.Create(c[i], remaining > 1 ? c[i + 1] : 0.0f, remaining > 2 ? c[i + 2] : 0.0f, 0.0f);
                        tot0 += Vector128.Load(di) * Shuffle0001(cs);
                        tot1 += Vector128.Load(di + 4) * Shuffle1122(cs);
                        tot2 += Vector128.Load(di + 8) * Shuffle2333(cs);
                    }

                    // stbir__store_output: [R G B] = (S0 + S2) + (S1 + S3)
                    float* o = output + x * 3;
                    o[0] = (tot0.GetElement(0) + tot1.GetElement(2)) + (tot0.GetElement(3) + tot2.GetElement(1));
                    o[1] = (tot0.GetElement(1) + tot1.GetElement(3)) + (tot1.GetElement(0) + tot2.GetElement(2));
                    o[2] = (tot0.GetElement(2) + tot2.GetElement(0)) + (tot1.GetElement(1) + tot2.GetElement(3));
                }
            }
        }

        /// <summary><c>stbir__simdf_0123to0001</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> Shuffle0001(Vector128<float> v) => Vector128.Shuffle(v, Vector128.Create(0, 0, 0, 1));

        /// <summary><c>stbir__simdf_0123to1122</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> Shuffle1122(Vector128<float> v) => Vector128.Shuffle(v, Vector128.Create(1, 1, 2, 2));

        /// <summary><c>stbir__simdf_0123to2333</c>.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static Vector128<float> Shuffle2333(Vector128<float> v) => Vector128.Shuffle(v, Vector128.Create(2, 3, 3, 3));

        /// <summary><c>stbir__vertical_gather_with_N_coeffs</c> (and the <c>_cont</c> variants): sums the
        /// contributing scanlines strictly in order, <c>o = r0*c0; o += r1*c1; ...</c>, element-wise. A single
        /// scanline with a weight within 1e-6 of one is copied verbatim, as stb does.</summary>
        private static void VerticalGather(float* output, float* coefficients, float** inputs, int total, int width)
        {
            float c0 = coefficients[0];
            if (total == 1 && c0 >= OneWeightLow && c0 <= OneWeightHigh)
            {
                Buffer.MemoryCopy(inputs[0], output, (long)width * sizeof(float), (long)width * sizeof(float));
                return;
            }

            int i = 0;
            for (; i + Vector128<float>.Count <= width; i += Vector128<float>.Count)
            {
                Vector128<float> o = Vector128.Load(inputs[0] + i) * Vector128.Create(c0);
                for (int k = 1; k < total; k++)
                {
                    o += Vector128.Load(inputs[k] + i) * Vector128.Create(coefficients[k]);
                }
                o.Store(output + i);
            }
            for (; i < width; i++)
            {
                float o = inputs[0][i] * c0;
                for (int k = 1; k < total; k++)
                {
                    o += inputs[k][i] * coefficients[k];
                }
                output[i] = o;
            }
        }

        /// <summary><c>stbir__vertical_gather_loop</c> (single split): keeps a ring buffer of the scanlines the
        /// current output row needs, filling it either with decoded scanlines (vertical first) or with
        /// horizontally resampled ones, then blends them vertically and encodes.</summary>
        public void VerticalGatherLoop(byte* input, byte* output)
        {
            int rowFloats = RingRowFloats;
            int stride = rowFloats + ScanlinePadding;
            int entries = _ringEntries;
            int decodeFloats = _inputWidth * 3 + ScanlinePadding;
            int encodeFloats = _outputWidth * 3 + ScanlinePadding;
            int maxRows = _vertical.Widest > 0 ? _vertical.Widest : 1;

            var ringBuffer = new float[(long)entries * stride];
            var decodeBuffer = new float[decodeFloats];
            var encodeBuffer = new float[encodeFloats];
            var rowPointers = new IntPtr[Math.Max(maxRows, entries)];
            int[] vn0 = _vertical.N0;
            int[] vn1 = _vertical.N1;
            int verticalCoefficientWidth = _vertical.CoefficientWidth;
            int outputRowBytes = _outputWidth * 3;

            fixed (float* ring = ringBuffer)
            fixed (float* decode = decodeBuffer)
            fixed (float* encode = encodeBuffer)
            fixed (float* verticalCoefficients = _vertical.Coefficients)
            fixed (IntPtr* rows = rowPointers)
            {
                // initialize the ring buffer for gathering
                int ringBeginIndex = 0;
                int ringFirstScanline = vn0[0];
                int ringLastScanline = ringFirstScanline - 1; // means "empty"

                for (int y = 0; y < _outputHeight; y++)
                {
                    int inFirstScanline = vn0[y];
                    int inLastScanline = vn1[y];

                    // Load in new scanlines
                    while (inLastScanline > ringLastScanline)
                    {
                        // make sure there was room in the ring buffer when we add new scanlines
                        if ((ringLastScanline - ringFirstScanline + 1) == entries)
                        {
                            ringFirstScanline++;
                            ringBeginIndex++;
                        }

                        ++ringLastScanline;
                        float* entry = ring + (long)((ringBeginIndex + (ringLastScanline - ringFirstScanline)) % entries) * stride;
                        if (_verticalFirst)
                        {
                            // Decode the nth scanline from the source image into the ring buffer.
                            DecodeScanline(input, ringLastScanline, entry);
                        }
                        else
                        {
                            // stbir__decode_and_resample_for_vertical_gather_loop
                            DecodeScanline(input, ringLastScanline, decode);
                            HorizontalGather(decode, entry);
                        }
                    }

                    // stbir__resample_vertical_gather
                    int total = inLastScanline - inFirstScanline + 1;
                    for (int k = 0; k < total; k++)
                    {
                        int index = (ringBeginIndex + (inFirstScanline + k - ringFirstScanline)) % entries;
                        rows[k] = (IntPtr)(ring + (long)index * stride);
                    }

                    float* coefficients = verticalCoefficients + (long)y * verticalCoefficientWidth;
                    if (_verticalFirst)
                    {
                        VerticalGather(decode, coefficients, (float**)rows, total, rowFloats);
                        // Now resample the gathered vertical data in the horizontal axis into the encode buffer
                        HorizontalGather(decode, encode);
                    }
                    else
                    {
                        VerticalGather(encode, coefficients, (float**)rows, total, rowFloats);
                    }

                    EncodeScanline(encode, output + (long)y * outputRowBytes);
                }
            }
        }

        /// <summary><c>stbir__vertical_scatter_loop</c> (single split), used for vertical downsamples whose filter
        /// spans more than 32 scanlines: every input row is (optionally horizontally resampled and) accumulated
        /// into the ring buffer rows of the output scanlines it contributes to (<c>o = r*c</c> on the first
        /// contribution, <c>o += r*c</c> after), and finished output rows are evicted and encoded.</summary>
        public void VerticalScatterLoop(byte* input, byte* output)
        {
            int rowFloats = RingRowFloats;
            int stride = rowFloats + ScanlinePadding;
            int entries = _ringEntries;
            int decodeFloats = _inputWidth * 3 + ScanlinePadding;
            int encodeFloats = _outputWidth * 3 + ScanlinePadding;
            int[] vn0 = _vertical.N0;
            int[] vn1 = _vertical.N1;
            int verticalCoefficientWidth = _vertical.CoefficientWidth;
            int margin = _vertical.FilterPixelMargin;
            int startOutputY = 0;
            int endOutputY = _outputHeight;
            int startInputY = -margin;
            int endInputY = _inputHeight + margin;

            var ringBuffer = new float[(long)entries * stride];
            var decodeBuffer = new float[decodeFloats];
            var verticalBufferArray = new float[encodeFloats];

            fixed (float* ring = ringBuffer)
            fixed (float* decode = decodeBuffer)
            fixed (float* verticalBuffer = verticalBufferArray)
            fixed (float* verticalCoefficients = _vertical.Coefficients)
            {
                // the buffer that gets scattered: the decoded scanline (vertical first) or its horizontal resample
                float* scatterBuffer = _verticalFirst ? decode : verticalBuffer;

                // initialize the ring buffer for scattering
                int ringFirstScanline = startOutputY;
                int ringLastScanline = -1;
                int ringBeginIndex = -1;

                // mark all the buffers as empty to start
                for (int e = 0; e < entries; e++)
                {
                    ring[(long)e * stride] = FloatEmptyMarker;
                }

                // do the loop in input space
                for (int y = startInputY; y < endInputY; y++)
                {
                    int contributor = y + margin;
                    int outFirstScanline = vn0[contributor];
                    int outLastScanline = vn1[contributor];

                    if (outLastScanline >= outFirstScanline &&
                        ((outFirstScanline >= startOutputY && outFirstScanline < endOutputY) || (outLastScanline >= startOutputY && outLastScanline < endOutputY)))
                    {
                        float* vc = verticalCoefficients + (long)contributor * verticalCoefficientWidth;

                        // clip the region
                        if (outFirstScanline < startOutputY)
                        {
                            vc += startOutputY - outFirstScanline;
                            outFirstScanline = startOutputY;
                        }
                        if (outLastScanline >= endOutputY)
                        {
                            outLastScanline = endOutputY - 1;
                        }

                        // if very first scanline, init the index
                        if (ringBeginIndex < 0)
                        {
                            ringBeginIndex = outFirstScanline - startOutputY;
                        }

                        // Decode the nth scanline from the source image into the decode buffer.
                        DecodeScanline(input, y, decode);

                        // When horizontal first, we resample horizontally into the vertical buffer before we scatter it out
                        if (!_verticalFirst)
                        {
                            HorizontalGather(decode, verticalBuffer);
                        }

                        // evict from the ringbuffer, if we need are full
                        if ((ringLastScanline - ringFirstScanline + 1) == entries && outLastScanline > ringLastScanline)
                        {
                            EvictFirstScanline(ring, stride, verticalBuffer, output, ref ringBeginIndex, ref ringFirstScanline);
                        }

                        // stbir__resample_vertical_scatter: set (r*c) on empty rows, blend (o + r*c) otherwise
                        for (int k = 0; k <= outLastScanline - outFirstScanline; k++)
                        {
                            int index = (ringBeginIndex + (outFirstScanline + k - ringFirstScanline)) % entries;
                            float* entry = ring + (long)index * stride;
                            float c = vc[k];
                            if (entry[0] == FloatEmptyMarker)
                            {
                                ScatterSet(entry, scatterBuffer, c, rowFloats);
                            }
                            else
                            {
                                ScatterBlend(entry, scatterBuffer, c, rowFloats);
                            }
                        }

                        // update the end of the buffer
                        if (outLastScanline > ringLastScanline)
                        {
                            ringLastScanline = outLastScanline;
                        }
                    }
                }

                // now evict the scanlines that are left over in the ring buffer
                while (ringFirstScanline < endOutputY)
                {
                    EvictFirstScanline(ring, stride, verticalBuffer, output, ref ringBeginIndex, ref ringFirstScanline);
                }
            }
        }

        /// <summary><c>stbir__encode_first_scanline_from_scatter</c> / <c>stbir__horizontal_resample_and_encode_first_scanline_from_scatter</c>:
        /// encodes the oldest ring buffer scanline (resampling it horizontally first when vertical-first), marks it
        /// empty and advances the ring.</summary>
        private void EvictFirstScanline(float* ring, int stride, float* verticalBuffer, byte* output, ref int ringBeginIndex, ref int ringFirstScanline)
        {
            float* entry = ring + (long)ringBeginIndex * stride;
            byte* outputRow = output + (long)ringFirstScanline * _outputWidth * 3;
            if (_verticalFirst)
            {
                HorizontalGather(entry, verticalBuffer);
                EncodeScanline(verticalBuffer, outputRow);
            }
            else
            {
                EncodeScanline(entry, outputRow);
            }

            // mark it as empty
            entry[0] = FloatEmptyMarker;

            // advance the first scanline
            ringFirstScanline++;
            if (++ringBeginIndex == _ringEntries)
            {
                ringBeginIndex = 0;
            }
        }

        /// <summary><c>stbir__vertical_scatter_with_N_coeffs</c> (set): <c>o = r * c</c>.</summary>
        private static void ScatterSet(float* output, float* input, float c, int width)
        {
            var cv = Vector128.Create(c);
            int i = 0;
            for (; i + Vector128<float>.Count <= width; i += Vector128<float>.Count)
            {
                (Vector128.Load(input + i) * cv).Store(output + i);
            }
            for (; i < width; i++)
            {
                output[i] = input[i] * c;
            }
        }

        /// <summary><c>stbir__vertical_scatter_with_N_coeffs_cont</c> (blend): <c>o = o + r * c</c>.</summary>
        private static void ScatterBlend(float* output, float* input, float c, int width)
        {
            var cv = Vector128.Create(c);
            int i = 0;
            for (; i + Vector128<float>.Count <= width; i += Vector128<float>.Count)
            {
                (Vector128.Load(output + i) + Vector128.Load(input + i) * cv).Store(output + i);
            }
            for (; i < width; i++)
            {
                output[i] = output[i] + input[i] * c;
            }
        }
    }
}
