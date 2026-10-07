// Portions of this file are a C# port of stb_image_resize.h v0.97 ("stb_image_resize v1", now in
// stb's deprecated/ folder; https://github.com/nothings/stb), by Jorge L Rodriguez (@VinoBS) with
// Sean Barrett, placed in the public domain / dual-licensed under the MIT license. Only the code path
// used by LiteRT-LM's image preprocessor is ported:
//   stbir_resize(..., STBIR_TYPE_UINT8, 3 channels, no alpha, flags 0, STBIR_EDGE_CLAMP x2,
//                STBIR_FILTER_CATMULLROM x2, STBIR_COLORSPACE_SRGB)
// reproducing the output of the x86-64 build shipped in liblitert-lm.so byte for byte (same
// contributors and coefficients, same float operation order, same sRGB tables, and the same scratch
// memory layout, which v1's filter construction relies on).

using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Intrinsics;

namespace SentenceTransformers.EmbeddingGemma2.Vision;

/// <summary>
/// Bit-exact managed port of the subset of stb_image_resize v1 (<c>stb_image_resize.h</c> v0.97) used by
/// LiteRT-LM's image preprocessor: packed 3-channel 8-bit sRGB resizing with a Catmull-Rom filter in both
/// directions and clamped edges. The filter construction (<c>stbir__calculate_filters</c> and its
/// up/downsampling helpers), the decode / horizontal / vertical passes, the ring-buffer scanline loops and
/// the sRGB encode are mirrored operation by operation so the produced bytes match the native library.
/// </summary>
/// <remarks>
/// All per-call state lives in one zeroed scratch block laid out exactly like stb's <c>tempmem</c>
/// (<c>stbir__resize_allocated</c>): v1's coefficient builders may write one coefficient past a
/// contributor's group, and those spills land in the next region of the block, so keeping the layout keeps
/// the results identical. Nothing is shared between calls, so <see cref="ResizeRgb"/> is thread-safe.
/// </remarks>
internal static class StbImageResize
{
    /// <summary>Number of interleaved channels (packed RGB).</summary>
    private const int Channels = 3;

    /// <summary>Support of <c>stbir__filter_catmullrom</c> (<c>stbir__support_two</c>), in either direction.</summary>
    private const float CatmullRomSupport = 2.0f;

    /// <summary>Extra zeroed bytes after stb's scratch block, so a spill past the last region stays in bounds.</summary>
    private const int ScratchPaddingBytes = 256;

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

    /// <summary>
    /// Resizes packed 8-bit sRGB RGB pixels (row-major, 3 bytes/pixel, no row padding) with a Catmull-Rom
    /// filter and clamped edges, reproducing stb_image_resize v1's
    /// <c>stbir_resize(..., STBIR_TYPE_UINT8, 3, STBIR_ALPHA_CHANNEL_NONE, 0, STBIR_EDGE_CLAMP, STBIR_EDGE_CLAMP,
    /// STBIR_FILTER_CATMULLROM, STBIR_FILTER_CATMULLROM, STBIR_COLORSPACE_SRGB, NULL)</c>.
    /// </summary>
    /// <param name="rgb">Source pixels, at least <c>width * height * 3</c> bytes.</param>
    /// <param name="width">Source width in pixels.</param>
    /// <param name="height">Source height in pixels.</param>
    /// <param name="newWidth">Destination width in pixels.</param>
    /// <param name="newHeight">Destination height in pixels.</param>
    /// <returns>The resized image, <c>newWidth * newHeight * 3</c> bytes.</returns>
    internal static byte[] ResizeRgb(ReadOnlySpan<byte> rgb, int width, int height, int newWidth, int newHeight)
    {
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(width), "Input image must have a positive width and height.");
        }
        if (newWidth <= 0 || newHeight <= 0)
        {
            throw new ArgumentOutOfRangeException(nameof(newWidth), "Output image must have a positive width and height.");
        }
        if (rgb.Length < (long)width * height * Channels)
        {
            throw new ArgumentException("Input buffer is smaller than width * height * 3 bytes.", nameof(rgb));
        }

        var output = new byte[checked(newWidth * newHeight * Channels)];
        var resize = new Resize(rgb, output, width, height, newWidth, newHeight);
        resize.Run();
        return output;
    }

    /// <summary><c>stbir__filter_catmullrom</c> (the unused <c>scale</c> argument is dropped).</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static float FilterCatmullRom(float x)
    {
        x = MathF.Abs(x);

        if (x < 1.0f)
        {
            return 1 - x * x * (2.5f - 1.5f * x);
        }
        else if (x < 2.0f)
        {
            return 2 - x * (4 + x * (0.5f * x - 2.5f));
        }

        return 0.0f;
    }

    /// <summary><c>stbir__use_upsampling</c>.</summary>
    private static bool UseUpsampling(float ratio) => ratio > 1;

    /// <summary><c>stbir__get_filter_pixel_width</c>: the maximum number of input samples that can affect an
    /// output sample.</summary>
    private static int GetFilterPixelWidth(float scale)
    {
        if (UseUpsampling(scale))
        {
            return (int)Math.Ceiling(CatmullRomSupport * 2);
        }
        return (int)Math.Ceiling(CatmullRomSupport * 2 / scale);
    }

    /// <summary><c>stbir__get_filter_pixel_margin</c>: how far buffers are expanded beyond the image edges.</summary>
    private static int GetFilterPixelMargin(float scale) => GetFilterPixelWidth(scale) / 2;

    /// <summary><c>stbir__get_coefficient_width</c> (both branches give <c>ceil(2 * 2)</c> for Catmull-Rom).</summary>
    private static int GetCoefficientWidth(float scale)
    {
        _ = scale;
        return (int)Math.Ceiling(CatmullRomSupport * 2);
    }

    /// <summary><c>stbir__get_contributors</c>: output pixels when upsampling, input pixels plus margins
    /// when downsampling.</summary>
    private static int GetContributors(float scale, int inputSize, int outputSize)
    {
        if (UseUpsampling(scale))
        {
            return outputSize;
        }
        return inputSize + GetFilterPixelMargin(scale) * 2;
    }

    /// <summary><c>stbir__edge_wrap</c> with <c>STBIR_EDGE_CLAMP</c>.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static int EdgeClamp(int n, int max)
    {
        if (n >= 0 && n < max)
        {
            return n;
        }
        if (n < 0)
        {
            return 0;
        }
        return max - 1;
    }

    /// <summary><c>stbir__calculate_sample_range_upsample</c>: which input pixels contribute to output pixel
    /// <paramref name="n"/>. The rounding is done in double, as in C (<c>floor(float + 0.5)</c>).</summary>
    private static void CalculateSampleRangeUpsample(int n, float outFilterRadius, float scaleRatio, float outShift,
                                                     out int inFirstPixel, out int inLastPixel, out float inCenterOfOut)
    {
        float outPixelCenter = (float)n + 0.5f;
        float outPixelInfluenceLowerbound = outPixelCenter - outFilterRadius;
        float outPixelInfluenceUpperbound = outPixelCenter + outFilterRadius;

        float inPixelInfluenceLowerbound = (outPixelInfluenceLowerbound + outShift) / scaleRatio;
        float inPixelInfluenceUpperbound = (outPixelInfluenceUpperbound + outShift) / scaleRatio;

        inCenterOfOut = (outPixelCenter + outShift) / scaleRatio;
        inFirstPixel = (int)Math.Floor(inPixelInfluenceLowerbound + 0.5);
        inLastPixel = (int)Math.Floor(inPixelInfluenceUpperbound - 0.5);
    }

    /// <summary><c>stbir__calculate_sample_range_downsample</c>: which output pixels input pixel
    /// <paramref name="n"/> contributes to.</summary>
    private static void CalculateSampleRangeDownsample(int n, float inPixelsRadius, float scaleRatio, float outShift,
                                                       out int outFirstPixel, out int outLastPixel, out float outCenterOfIn)
    {
        float inPixelCenter = (float)n + 0.5f;
        float inPixelInfluenceLowerbound = inPixelCenter - inPixelsRadius;
        float inPixelInfluenceUpperbound = inPixelCenter + inPixelsRadius;

        float outPixelInfluenceLowerbound = inPixelInfluenceLowerbound * scaleRatio - outShift;
        float outPixelInfluenceUpperbound = inPixelInfluenceUpperbound * scaleRatio - outShift;

        outCenterOfIn = inPixelCenter * scaleRatio - outShift;
        outFirstPixel = (int)Math.Floor(outPixelInfluenceLowerbound + 0.5);
        outLastPixel = (int)Math.Floor(outPixelInfluenceUpperbound - 0.5);
    }

    /// <summary><c>stbir__calculate_coefficients_upsample</c>. <paramref name="contributor"/> starts at an
    /// <c>{ n0, n1 }</c> pair. Like stb, this may write (and sum) one coefficient past the group's width;
    /// that spill lands in the next group (or the next scratch region) and is overwritten later, which is why
    /// both spans run to the end of the scratch block.</summary>
    private static void CalculateCoefficientsUpsample(int inFirstPixel, int inLastPixel, float inCenterOfOut,
                                                      Span<int> contributor, Span<float> coefficientGroup)
    {
        int i;
        float totalFilter = 0;
        float filterScale;

        contributor[0] = inFirstPixel;
        contributor[1] = inLastPixel;

        for (i = 0; i <= inLastPixel - inFirstPixel; i++)
        {
            float inPixelCenter = (float)(i + inFirstPixel) + 0.5f;
            coefficientGroup[i] = FilterCatmullRom(inCenterOfOut - inPixelCenter);

            // If the coefficient is zero, skip it. (Don't do the <0 check here, we want the influence of those outside pixels.)
            if (i == 0 && coefficientGroup[i] == 0)
            {
                contributor[0] = ++inFirstPixel;
                i--;
                continue;
            }

            totalFilter += coefficientGroup[i];
        }

        // Make sure the sum of all coefficients is 1.
        filterScale = 1 / totalFilter;

        for (i = 0; i <= inLastPixel - inFirstPixel; i++)
        {
            coefficientGroup[i] *= filterScale;
        }

        for (i = inLastPixel - inFirstPixel; i >= 0; i--)
        {
            if (coefficientGroup[i] != 0)
            {
                break;
            }

            // This line has no weight. We can skip it.
            contributor[1] = contributor[0] + i - 1;
        }
    }

    /// <summary><c>stbir__calculate_coefficients_downsample</c> (same spill behaviour as the upsample builder).</summary>
    private static void CalculateCoefficientsDownsample(float scaleRatio, int outFirstPixel, int outLastPixel, float outCenterOfIn,
                                                        Span<int> contributor, Span<float> coefficientGroup)
    {
        int i;

        contributor[0] = outFirstPixel;
        contributor[1] = outLastPixel;

        for (i = 0; i <= outLastPixel - outFirstPixel; i++)
        {
            float outPixelCenter = (float)(i + outFirstPixel) + 0.5f;
            float x = outPixelCenter - outCenterOfIn;
            coefficientGroup[i] = FilterCatmullRom(x) * scaleRatio;
        }

        for (i = outLastPixel - outFirstPixel; i >= 0; i--)
        {
            if (coefficientGroup[i] != 0)
            {
                break;
            }

            // This line has no weight. We can skip it.
            contributor[1] = contributor[0] + i - 1;
        }
    }

    /// <summary><c>stbir__normalize_downsample_coefficients</c>: makes every output pixel's weights sum to one,
    /// then drops leading zero / out-of-image coefficients and clamps <c>n1</c> to the output.</summary>
    private static void NormalizeDownsampleCoefficients(Span<int> contributors, Span<float> coefficients, float scaleRatio, int inputSize, int outputSize)
    {
        int numContributors = GetContributors(scaleRatio, inputSize, outputSize);
        int numCoefficients = GetCoefficientWidth(scaleRatio);
        int width = numCoefficients;
        int i, j;
        int skip;

        for (i = 0; i < outputSize; i++)
        {
            float scale;
            float total = 0;

            for (j = 0; j < numContributors; j++)
            {
                int n0 = contributors[2 * j], n1 = contributors[2 * j + 1];
                if (i >= n0 && i <= n1)
                {
                    float coefficient = coefficients[width * j + i - n0];
                    total += coefficient;
                }
                else if (i < n0)
                {
                    break;
                }
            }

            scale = 1 / total;

            for (j = 0; j < numContributors; j++)
            {
                int n0 = contributors[2 * j], n1 = contributors[2 * j + 1];
                if (i >= n0 && i <= n1)
                {
                    coefficients[width * j + i - n0] *= scale;
                }
                else if (i < n0)
                {
                    break;
                }
            }
        }

        // Optimize: Skip zero coefficients and contributions outside of image bounds.
        // Do this after normalizing because normalization depends on the n0/n1 values.
        for (j = 0; j < numContributors; j++)
        {
            int range, max;

            skip = 0;
            while (coefficients[width * j + skip] == 0)
            {
                skip++;
            }

            contributors[2 * j] += skip;

            while (contributors[2 * j] < 0)
            {
                contributors[2 * j]++;
                skip++;
            }

            range = contributors[2 * j + 1] - contributors[2 * j] + 1;
            max = Math.Min(numCoefficients, range);

            for (i = 0; i < max; i++)
            {
                if (i + skip >= width)
                {
                    break;
                }

                coefficients[width * j + i] = coefficients[width * j + i + skip];
            }
        }

        // Using min to avoid writing into invalid pixels.
        for (i = 0; i < numContributors; i++)
        {
            contributors[2 * i + 1] = Math.Min(contributors[2 * i + 1], outputSize - 1);
        }
    }

    /// <summary><c>stbir__calculate_filters</c>: builds the contributor ranges and coefficients of one axis.</summary>
    private static void CalculateFilters(Span<int> contributors, Span<float> coefficients, float scaleRatio, float shift, int inputSize, int outputSize)
    {
        int n;
        int totalContributors = GetContributors(scaleRatio, inputSize, outputSize);
        int coefficientWidth = GetCoefficientWidth(scaleRatio);

        if (UseUpsampling(scaleRatio))
        {
            float outPixelsRadius = CatmullRomSupport * scaleRatio;

            // Looping through out pixels
            for (n = 0; n < totalContributors; n++)
            {
                CalculateSampleRangeUpsample(n, outPixelsRadius, scaleRatio, shift, out int inFirstPixel, out int inLastPixel, out float inCenterOfOut);
                CalculateCoefficientsUpsample(inFirstPixel, inLastPixel, inCenterOfOut, contributors.Slice(2 * n), coefficients.Slice(coefficientWidth * n));
            }
        }
        else
        {
            float inPixelsRadius = CatmullRomSupport / scaleRatio;
            int margin = GetFilterPixelMargin(scaleRatio);

            // Looping through in pixels
            for (n = 0; n < totalContributors; n++)
            {
                int nAdjusted = n - margin;
                CalculateSampleRangeDownsample(nAdjusted, inPixelsRadius, scaleRatio, shift, out int outFirstPixel, out int outLastPixel, out float outCenterOfIn);
                CalculateCoefficientsDownsample(scaleRatio, outFirstPixel, outLastPixel, outCenterOfIn, contributors.Slice(2 * n), coefficients.Slice(coefficientWidth * n));
            }

            NormalizeDownsampleCoefficients(contributors, coefficients, scaleRatio, inputSize, outputSize);
        }
    }

    /// <summary><c>stbir__linear_to_srgb_uchar</c> (IEEE-float version): piecewise-linear table encode.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static byte LinearToSrgbUchar(float value)
    {
        const uint AlmostOneBits = 0x3f7fffff;          // 1-eps
        const uint MinValBits = (127 - 13) << 23;       // 2^-13
        float almostOne = BitConverter.UInt32BitsToSingle(AlmostOneBits);
        float minVal = BitConverter.UInt32BitsToSingle(MinValBits);

        // Clamp to [2^(-13), 1-eps]; these two values map to 0 and 1, respectively.
        // The tests are carefully written so that NaNs map to 0, same as in the reference implementation.
        if (!(value > minVal))
        {
            value = minVal;
        }
        if (value > almostOne)
        {
            value = almostOne;
        }

        // Do the table lookup and unpack bias, scale
        uint u = BitConverter.SingleToUInt32Bits(value);
        uint tab = Fp32ToSrgb8Tab4[(u - MinValBits) >> 20];
        uint bias = (tab >> 16) << 9;
        uint scale = tab & 0xffff;

        // Grab next-highest mantissa bits and perform linear interpolation
        uint t = (u >> 12) & 0xff;
        return (byte)((bias + scale * t) >> 16);
    }

    /// <summary><c>dst[i] += src[i] * coefficient</c> for <paramref name="count"/> floats: the element-wise
    /// multiply-add of stb's vertical passes. Each element keeps its own separate multiply then add (never
    /// fused), so the vectorized form is bit-identical to the scalar C loop.</summary>
    [MethodImpl(MethodImplOptions.AggressiveInlining)]
    private static void MultiplyAccumulate(Span<float> dst, ReadOnlySpan<float> src, float coefficient, int count)
    {
        dst = dst.Slice(0, count);
        src = src.Slice(0, count);
        ref float d = ref MemoryMarshal.GetReference(dst);
        ref float sr = ref MemoryMarshal.GetReference(src);
        int i = 0;
        if (Vector256.IsHardwareAccelerated)
        {
            var c = Vector256.Create(coefficient);
            for (; i <= count - Vector256<float>.Count; i += Vector256<float>.Count)
            {
                var product = Vector256.Multiply(Vector256.LoadUnsafe(ref sr, (nuint)i), c);
                Vector256.Add(Vector256.LoadUnsafe(ref d, (nuint)i), product).StoreUnsafe(ref d, (nuint)i);
            }
        }
        else if (Vector128.IsHardwareAccelerated)
        {
            var c = Vector128.Create(coefficient);
            for (; i <= count - Vector128<float>.Count; i += Vector128<float>.Count)
            {
                var product = Vector128.Multiply(Vector128.LoadUnsafe(ref sr, (nuint)i), c);
                Vector128.Add(Vector128.LoadUnsafe(ref d, (nuint)i), product).StoreUnsafe(ref d, (nuint)i);
            }
        }
        for (; i < count; i++)
        {
            dst[i] += src[i] * coefficient;
        }
    }

    /// <summary>
    /// <c>stbir__info</c> plus the scratch block of <c>stbir__resize_allocated</c> for one call: the
    /// transform (<c>stbir__calculate_transform</c>), the buffer sizes (<c>stbir__calculate_memory</c>) and
    /// the passes that run over them. The scratch block is one zeroed float array (the integer contributor
    /// regions are views of the same memory), and every stb pointer into it is an element offset.
    /// </summary>
    private ref struct Resize
    {
        private readonly int _inputW;
        private readonly int _inputH;
        private readonly int _outputW;
        private readonly int _outputH;

        private readonly float _horizontalScale;
        private readonly float _verticalScale;
        private readonly float _horizontalShift;
        private readonly float _verticalShift;

        private readonly int _horizontalCoefficientWidth;
        private readonly int _verticalCoefficientWidth;
        private readonly int _horizontalFilterPixelMargin;
        private readonly int _verticalFilterPixelMargin;

        private readonly int _ringBufferLength;     // floats per ring buffer entry (output_w * channels)
        private readonly int _ringBufferNumEntries;

        // Region offsets in the scratch block (in 4-byte elements), in tempmem order; -1 for an unused buffer.
        private readonly int _horizontalContributors;
        private readonly int _horizontalCoefficients;
        private readonly int _verticalContributors;
        private readonly int _verticalCoefficients;
        private readonly int _decodeBuffer;
        private readonly int _horizontalBuffer;
        private readonly int _ringBuffer;
        private readonly int _encodeBuffer;

        private readonly Span<float> _scratch;
        private readonly Span<int> _scratchInts;
        private readonly ReadOnlySpan<byte> _input;
        private readonly Span<byte> _output;

        private int _ringBufferFirstScanline;
        private int _ringBufferLastScanline;
        private int _ringBufferBeginIndex;

        /// <summary><c>stbir__setup</c>, <c>stbir__calculate_transform</c> (s0 = t0 = 0, s1 = t1 = 1, no
        /// transform), <c>stbir__calculate_memory</c> and the scratch layout of <c>stbir__resize_allocated</c>.</summary>
        public Resize(ReadOnlySpan<byte> input, Span<byte> output, int inputW, int inputH, int outputW, int outputH)
        {
            _input = input;
            _output = output;
            _inputW = inputW;
            _inputH = inputH;
            _outputW = outputW;
            _outputH = outputH;

            const float s0 = 0, t0 = 0, s1 = 1, t1 = 1;
            _horizontalScale = ((float)outputW / inputW) / (s1 - s0);
            _verticalScale = ((float)outputH / inputH) / (t1 - t0);
            _horizontalShift = s0 * outputW / (s1 - s0);
            _verticalShift = t0 * outputH / (t1 - t0);

            int pixelMargin = GetFilterPixelMargin(_horizontalScale);
            int filterHeight = GetFilterPixelWidth(_verticalScale);
            int horizontalNumContributors = GetContributors(_horizontalScale, inputW, outputW);
            int verticalNumContributors = GetContributors(_verticalScale, inputH, outputH);

            // One extra entry because floating point precision problems sometimes cause an extra to be necessary.
            _ringBufferNumEntries = filterHeight + 1;

            _horizontalCoefficientWidth = GetCoefficientWidth(_horizontalScale);
            _verticalCoefficientWidth = GetCoefficientWidth(_verticalScale);
            _horizontalFilterPixelMargin = pixelMargin;
            _verticalFilterPixelMargin = GetFilterPixelMargin(_verticalScale);
            _ringBufferLength = outputW * Channels;

            // Region sizes in elements (every region holds 4-byte ints or floats).
            long horizontalContributorsSize = (long)horizontalNumContributors * 2;
            long horizontalCoefficientsSize = (long)horizontalNumContributors * _horizontalCoefficientWidth;
            long verticalContributorsSize = (long)verticalNumContributors * 2;
            long verticalCoefficientsSize = (long)verticalNumContributors * _verticalCoefficientWidth;
            long decodeBufferSize = (long)(inputW + pixelMargin * 2) * Channels;
            long horizontalBufferSize = (long)outputW * Channels;
            long ringBufferSize = (long)outputW * Channels * _ringBufferNumEntries;
            long encodeBufferSize = (long)outputW * Channels;

            if (UseUpsampling(_verticalScale))
            {
                // The horizontal buffer is only used when downsampling the height.
                horizontalBufferSize = 0;
            }
            else
            {
                // The encode buffer is only used when upsampling the height.
                encodeBufferSize = 0;
            }

            long p = 0;
            _horizontalContributors = (int)p;
            p += horizontalContributorsSize;
            _horizontalCoefficients = (int)p;
            p += horizontalCoefficientsSize;
            _verticalContributors = (int)p;
            p += verticalContributorsSize;
            _verticalCoefficients = (int)p;
            p += verticalCoefficientsSize;
            _decodeBuffer = (int)p;
            p += decodeBufferSize;
            if (UseUpsampling(_verticalScale))
            {
                _horizontalBuffer = -1;
                _ringBuffer = (int)p;
                p += ringBufferSize;
                _encodeBuffer = (int)p;
            }
            else
            {
                _horizontalBuffer = (int)p;
                p += horizontalBufferSize;
                _ringBuffer = (int)p;
                _encodeBuffer = -1;
            }

            long memoryRequired = horizontalContributorsSize + horizontalCoefficientsSize
                                + verticalContributorsSize + verticalCoefficientsSize
                                + decodeBufferSize + horizontalBufferSize
                                + ringBufferSize + encodeBufferSize;
            _scratch = new float[checked((int)(memoryRequired + ScratchPaddingBytes / sizeof(float)))];
            _scratchInts = MemoryMarshal.Cast<float, int>(_scratch);

            // This signals that the ring buffer is empty
            _ringBufferBeginIndex = -1;
            _ringBufferFirstScanline = 0;
            _ringBufferLastScanline = 0;
        }

        /// <summary><c>stbir__resize_allocated</c> after the layout: builds both filters and runs the up- or
        /// downsampling scanline loop.</summary>
        public void Run()
        {
            // Filter regions run to the end of the block: the builders' spills land in the next region, as in stb.
            CalculateFilters(_scratchInts.Slice(_horizontalContributors), _scratch.Slice(_horizontalCoefficients), _horizontalScale, _horizontalShift, _inputW, _outputW);
            CalculateFilters(_scratchInts.Slice(_verticalContributors), _scratch.Slice(_verticalCoefficients), _verticalScale, _verticalShift, _inputH, _outputH);

            if (UseUpsampling(_verticalScale))
            {
                BufferLoopUpsample();
            }
            else
            {
                BufferLoopDownsample();
            }
        }

        /// <summary><c>stbir__get_decode_buffer</c>: the element offset of decoded pixel 0; the left margin
        /// lies before it.</summary>
        private readonly int DecodeBufferOrigin => _decodeBuffer + _horizontalFilterPixelMargin * Channels;

        /// <summary><c>stbir__decode_scanline</c> for <c>STBIR__DECODE(STBIR_TYPE_UINT8, STBIR_COLORSPACE_SRGB)</c>
        /// with clamped edges (no alpha, so the premultiply step is skipped: <c>STBIR_FLAG_ALPHA_PREMULTIPLIED</c>
        /// is forced on for <c>alpha_channel &lt; 0</c>).</summary>
        private readonly void DecodeScanline(int n)
        {
            var decodeBuffer = _scratch.Slice(_decodeBuffer);   // pixel x lives at (x + margin) * Channels
            var inputData = _input.Slice(EdgeClamp(n, _inputH) * _inputW * Channels, _inputW * Channels);
            var table = SrgbUcharToLinearFloat;
            int margin = _horizontalFilterPixelMargin;
            int maxX = _inputW + margin;
            int x = -margin;

            // Left margin, interior, right margin (stbir__edge_wrap is the identity inside the image).
            for (; x < 0 && x < maxX; x++)
            {
                DecodePixel(decodeBuffer.Slice((x + margin) * Channels), inputData.Slice(EdgeClamp(x, _inputW) * Channels), table);
            }
            int interiorEnd = Math.Min(_inputW, maxX);
            for (; x < interiorEnd; x++)
            {
                DecodePixel(decodeBuffer.Slice((x + margin) * Channels), inputData.Slice(x * Channels), table);
            }
            for (; x < maxX; x++)
            {
                DecodePixel(decodeBuffer.Slice((x + margin) * Channels), inputData.Slice(EdgeClamp(x, _inputW) * Channels), table);
            }
        }

        /// <summary>One pixel of the sRGB decode: <c>stbir__srgb_uchar_to_linear_float[byte]</c> per channel.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static void DecodePixel(Span<float> dst, ReadOnlySpan<byte> src, float[] table)
        {
            dst[0] = table[src[0]];
            dst[1] = table[src[1]];
            dst[2] = table[src[2]];
        }

        /// <summary><c>stbir__get_ring_buffer_entry</c> (to the end of the block, like the C pointer).</summary>
        private readonly Span<float> GetRingBufferEntry(int index) => _scratch.Slice(_ringBuffer + index * _ringBufferLength);

        /// <summary><c>stbir__get_ring_buffer_scanline</c>.</summary>
        private readonly Span<float> GetRingBufferScanline(int getScanline)
        {
            int ringBufferIndex = (_ringBufferBeginIndex + (getScanline - _ringBufferFirstScanline)) % _ringBufferNumEntries;
            return GetRingBufferEntry(ringBufferIndex);
        }

        /// <summary><c>stbir__add_empty_ring_buffer_entry</c>: appends scanline <paramref name="n"/> and zeroes it.</summary>
        private Span<float> AddEmptyRingBufferEntry(int n)
        {
            int ringBufferIndex;

            _ringBufferLastScanline = n;

            if (_ringBufferBeginIndex < 0)
            {
                ringBufferIndex = _ringBufferBeginIndex = 0;
                _ringBufferFirstScanline = n;
            }
            else
            {
                ringBufferIndex = (_ringBufferBeginIndex + (_ringBufferLastScanline - _ringBufferFirstScanline)) % _ringBufferNumEntries;
            }

            var ringBuffer = GetRingBufferEntry(ringBufferIndex);
            ringBuffer.Slice(0, _ringBufferLength).Clear();
            return ringBuffer;
        }

        /// <summary><c>stbir__resample_horizontal_upsample</c> (3-channel case): gathers each output pixel from
        /// its contributing decoded input pixels, in increasing input order.</summary>
        private readonly void ResampleHorizontalUpsample(Span<float> outputBuffer)
        {
            int decodeBuffer = DecodeBufferOrigin;
            var scratch = _scratch;
            var contributors = _scratchInts.Slice(_horizontalContributors);
            var coefficients = _scratch.Slice(_horizontalCoefficients);
            int coefficientWidth = _horizontalCoefficientWidth;

            for (int x = 0; x < _outputW; x++)
            {
                int n0 = contributors[2 * x];
                int n1 = contributors[2 * x + 1];

                var outPixel = outputBuffer.Slice(x * Channels, Channels);
                var coefficientGroup = coefficients.Slice(coefficientWidth * x);
                float o0 = outPixel[0], o1 = outPixel[1], o2 = outPixel[2];

                for (int k = n0; k <= n1; k++)
                {
                    int inPixel = decodeBuffer + k * Channels;
                    float coefficient = coefficientGroup[k - n0];
                    o0 += scratch[inPixel] * coefficient;
                    o1 += scratch[inPixel + 1] * coefficient;
                    o2 += scratch[inPixel + 2] * coefficient;
                }

                outPixel[0] = o0;
                outPixel[1] = o1;
                outPixel[2] = o2;
            }
        }

        /// <summary><c>stbir__resample_horizontal_downsample</c> (3-channel case): scatters each decoded input
        /// pixel (margins included) into the output pixels it contributes to.</summary>
        private readonly void ResampleHorizontalDownsample(Span<float> outputBuffer)
        {
            int decodeBuffer = DecodeBufferOrigin;
            var scratch = _scratch;
            var contributors = _scratchInts.Slice(_horizontalContributors);
            var coefficients = _scratch.Slice(_horizontalCoefficients);
            int coefficientWidth = _horizontalCoefficientWidth;
            int filterPixelMargin = _horizontalFilterPixelMargin;
            int maxX = _inputW + filterPixelMargin * 2;

            for (int x = 0; x < maxX; x++)
            {
                int n0 = contributors[2 * x];
                int n1 = contributors[2 * x + 1];

                int inPixel = decodeBuffer + (x - filterPixelMargin) * Channels;
                float i0 = scratch[inPixel], i1 = scratch[inPixel + 1], i2 = scratch[inPixel + 2];
                var coefficientGroup = coefficients.Slice(coefficientWidth * x);

                for (int k = n0; k <= n1; k++)
                {
                    var outPixel = outputBuffer.Slice(k * Channels, Channels);
                    float coefficient = coefficientGroup[k - n0];
                    outPixel[0] += i0 * coefficient;
                    outPixel[1] += i1 * coefficient;
                    outPixel[2] += i2 * coefficient;
                }
            }
        }

        /// <summary><c>stbir__decode_and_resample_upsample</c>: decodes input row <paramref name="n"/> and resamples
        /// it horizontally into a new ring buffer entry.</summary>
        private void DecodeAndResampleUpsample(int n)
        {
            DecodeScanline(n);

            if (UseUpsampling(_horizontalScale))
            {
                ResampleHorizontalUpsample(AddEmptyRingBufferEntry(n));
            }
            else
            {
                ResampleHorizontalDownsample(AddEmptyRingBufferEntry(n));
            }
        }

        /// <summary><c>stbir__decode_and_resample_downsample</c>: decodes input row <paramref name="n"/> and resamples
        /// it horizontally into the (zeroed) horizontal buffer.</summary>
        private readonly void DecodeAndResampleDownsample(int n)
        {
            DecodeScanline(n);

            var horizontalBuffer = _scratch.Slice(_horizontalBuffer);
            horizontalBuffer.Slice(0, _outputW * Channels).Clear();

            if (UseUpsampling(_horizontalScale))
            {
                ResampleHorizontalUpsample(horizontalBuffer);
            }
            else
            {
                ResampleHorizontalDownsample(horizontalBuffer);
            }
        }

        /// <summary><c>stbir__encode_scanline</c> for <c>STBIR__DECODE(STBIR_TYPE_UINT8, STBIR_COLORSPACE_SRGB)</c>
        /// with no alpha channel: every channel goes through <c>stbir__linear_to_srgb_uchar</c>.</summary>
        private readonly void EncodeScanline(int row, ReadOnlySpan<float> encodeBuffer)
        {
            int count = _outputW * Channels;
            var outputBuffer = _output.Slice(row * count, count);
            encodeBuffer = encodeBuffer.Slice(0, count);
            for (int i = 0; i < count; i++)
            {
                outputBuffer[i] = LinearToSrgbUchar(encodeBuffer[i]);
            }
        }

        /// <summary><c>stbir__resample_vertical_upsample</c>: gathers output row <paramref name="n"/> from its
        /// ring buffer scanlines into the encode buffer, then encodes it.</summary>
        private readonly void ResampleVerticalUpsample(int n)
        {
            int count = _outputW * Channels;
            int contributor = n;
            var coefficientGroup = _scratch.Slice(_verticalCoefficients + _verticalCoefficientWidth * contributor);
            int n0 = _scratchInts[_verticalContributors + 2 * contributor];
            int n1 = _scratchInts[_verticalContributors + 2 * contributor + 1];
            var encodeBuffer = _scratch.Slice(_encodeBuffer);

            encodeBuffer.Slice(0, count).Clear();

            int coefficientCounter = 0;
            for (int k = n0; k <= n1; k++)
            {
                int coefficientIndex = coefficientCounter++;
                var ringBufferEntry = GetRingBufferScanline(k);
                float coefficient = coefficientGroup[coefficientIndex];
                MultiplyAccumulate(encodeBuffer, ringBufferEntry, coefficient, count);
            }

            EncodeScanline(n, encodeBuffer);
        }

        /// <summary><c>stbir__resample_vertical_downsample</c>: scatters the horizontal buffer of input row
        /// <paramref name="n"/> into the ring buffer scanlines it contributes to.</summary>
        private readonly void ResampleVerticalDownsample(int n)
        {
            int count = _outputW * Channels;
            int contributor = n + _verticalFilterPixelMargin;
            var coefficientGroup = _scratch.Slice(_verticalCoefficients + _verticalCoefficientWidth * contributor);
            int n0 = _scratchInts[_verticalContributors + 2 * contributor];
            int n1 = _scratchInts[_verticalContributors + 2 * contributor + 1];
            var horizontalBuffer = _scratch.Slice(_horizontalBuffer);

            for (int k = n0; k <= n1; k++)
            {
                float coefficient = coefficientGroup[k - n0];
                var ringBufferEntry = GetRingBufferScanline(k);
                MultiplyAccumulate(ringBufferEntry, horizontalBuffer, coefficient, count);
            }
        }

        /// <summary><c>stbir__buffer_loop_upsample</c>: for each output row, slides the ring buffer of
        /// horizontally resampled input rows forward and gathers the row.</summary>
        private void BufferLoopUpsample()
        {
            float scaleRatio = _verticalScale;
            float outScanlinesRadius = CatmullRomSupport * scaleRatio;

            for (int y = 0; y < _outputH; y++)
            {
                CalculateSampleRangeUpsample(y, outScanlinesRadius, scaleRatio, _verticalShift, out int inFirstScanline, out int inLastScanline, out _);

                if (_ringBufferBeginIndex >= 0)
                {
                    // Get rid of whatever we don't need anymore.
                    while (inFirstScanline > _ringBufferFirstScanline)
                    {
                        if (_ringBufferFirstScanline == _ringBufferLastScanline)
                        {
                            // We just popped the last scanline off the ring buffer.
                            // Reset it to the empty state.
                            _ringBufferBeginIndex = -1;
                            _ringBufferFirstScanline = 0;
                            _ringBufferLastScanline = 0;
                            break;
                        }
                        else
                        {
                            _ringBufferFirstScanline++;
                            _ringBufferBeginIndex = (_ringBufferBeginIndex + 1) % _ringBufferNumEntries;
                        }
                    }
                }

                // Load in new ones.
                if (_ringBufferBeginIndex < 0)
                {
                    DecodeAndResampleUpsample(inFirstScanline);
                }

                while (inLastScanline > _ringBufferLastScanline)
                {
                    DecodeAndResampleUpsample(_ringBufferLastScanline + 1);
                }

                // Now all buffers should be ready to write a row of vertical sampling.
                ResampleVerticalUpsample(y);
            }
        }

        /// <summary><c>stbir__empty_ring_buffer</c>: encodes and retires every finished output row before
        /// <paramref name="firstNecessaryScanline"/>.</summary>
        private void EmptyRingBuffer(int firstNecessaryScanline)
        {
            if (_ringBufferBeginIndex >= 0)
            {
                // Get rid of whatever we don't need anymore.
                while (firstNecessaryScanline > _ringBufferFirstScanline)
                {
                    if (_ringBufferFirstScanline >= 0 && _ringBufferFirstScanline < _outputH)
                    {
                        EncodeScanline(_ringBufferFirstScanline, GetRingBufferEntry(_ringBufferBeginIndex));
                    }

                    if (_ringBufferFirstScanline == _ringBufferLastScanline)
                    {
                        // We just popped the last scanline off the ring buffer.
                        // Reset it to the empty state.
                        _ringBufferBeginIndex = -1;
                        _ringBufferFirstScanline = 0;
                        _ringBufferLastScanline = 0;
                        break;
                    }
                    else
                    {
                        _ringBufferFirstScanline++;
                        _ringBufferBeginIndex = (_ringBufferBeginIndex + 1) % _ringBufferNumEntries;
                    }
                }
            }
        }

        /// <summary><c>stbir__buffer_loop_downsample</c>: for each input row (margins included), resamples it
        /// horizontally and scatters it into the ring buffer of output rows, emitting rows as they complete.</summary>
        private void BufferLoopDownsample()
        {
            float scaleRatio = _verticalScale;
            int outputH = _outputH;
            float inPixelsRadius = CatmullRomSupport / scaleRatio;
            int pixelMargin = _verticalFilterPixelMargin;
            int maxY = _inputH + pixelMargin;

            for (int y = -pixelMargin; y < maxY; y++)
            {
                CalculateSampleRangeDownsample(y, inPixelsRadius, scaleRatio, _verticalShift, out int outFirstScanline, out int outLastScanline, out _);

                if (outLastScanline < 0 || outFirstScanline >= outputH)
                {
                    continue;
                }

                EmptyRingBuffer(outFirstScanline);

                DecodeAndResampleDownsample(y);

                // Load in new ones.
                if (_ringBufferBeginIndex < 0)
                {
                    AddEmptyRingBufferEntry(outFirstScanline);
                }

                while (outLastScanline > _ringBufferLastScanline)
                {
                    AddEmptyRingBufferEntry(_ringBufferLastScanline + 1);
                }

                // Now the horizontal buffer is ready to write to all ring buffer rows.
                ResampleVerticalDownsample(y);
            }

            EmptyRingBuffer(_outputH);
        }
    }
}
