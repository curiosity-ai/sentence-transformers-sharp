// Image decoding front-end mirroring stb_image.h v2.30 (stbi_load_from_memory) by Sean Barrett and
// contributors (https://github.com/nothings/stb), dual-licensed public domain (unlicense) / MIT.

namespace SentenceTransformers.EmbeddingGemma2.Vision;

/// <summary>
/// Decodes encoded images to the exact 8-bit RGB pixels the reference runtime obtains from
/// <c>stbi_load_from_memory(bytes, len, &amp;w, &amp;h, &amp;c, 3)</c>, so that image embeddings match
/// bit-for-bit. Supports PNG, JPEG (baseline + progressive) and uncompressed BMP; formats are sniffed
/// in stb's order (PNG signature, BMP header, JPEG SOI). EXIF orientation is not applied (stb does not
/// apply it either). All entry points are thread-safe.
/// </summary>
internal static class ImageDecoder
{
    /// <summary>
    /// Decodes <paramref name="data"/> to packed RGB (3 bytes per pixel, row-major, top row first).
    /// Returns false for unsupported formats and for streams stb_image would reject.
    /// </summary>
    public static bool TryDecode(ReadOnlySpan<byte> data, out byte[] rgb, out int width, out int height)
    {
        rgb = null;
        width = 0;
        height = 0;
        try
        {
            // stbi__load_main tests formats with explicit magic numbers first, then JPEG.
            if (PngDecoder.IsPng(data))
            {
                rgb = PngDecoder.Decode(data, out width, out height);
                return true;
            }
            if (BmpDecoder.IsBmp(data))
            {
                rgb = BmpDecoder.Decode(data, out width, out height);
                return true;
            }
            if (JpegDecoder.IsJpeg(data))
            {
                rgb = JpegDecoder.Decode(data, out width, out height);
                return true;
            }
            return false;
        }
        catch (Exception e) when (e is InvalidDataException or IndexOutOfRangeException or ArgumentException or OverflowException)
        {
            // Malformed input. IndexOutOfRange/Argument/Overflow are a safety net for corrupt streams that
            // slip past stb's own validation; the decoders never write outside their buffers.
            rgb = null;
            width = 0;
            height = 0;
            return false;
        }
    }
}
