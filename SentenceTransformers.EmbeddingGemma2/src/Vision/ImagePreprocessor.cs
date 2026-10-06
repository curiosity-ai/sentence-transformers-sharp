namespace SentenceTransformers.EmbeddingGemma2.Vision;

/// <summary>A patchified image, padded to the encoder signature length.</summary>
internal sealed record PatchifiedImage(float[] Patches, int[] Positions, int NumPatches, int ValidPatches, int Height, int Width);

/// <summary>
/// Port of LiteRT-LM's image preprocessing for EmbeddingGemma 2 (<c>stb_image_preprocessor.cc</c> and
/// <c>image_preprocessor_utils.cc</c>): aspect-preserving resize so the patch grid fits
/// <c>tokens · k²</c> patches with both sides multiples of <c>k · patch</c>, sRGB Catmull-Rom resampling
/// (stb_image_resize2), scaling to [0, 1], row-major patchification with (x, y) positions, and padding to
/// the signature length with zero patches at position -1.
/// </summary>
internal static class ImagePreprocessor
{
    /// <summary><c>GetAspectRatioPreservingSize</c> (float32 arithmetic, as in the C++ code). Returns (height, width).</summary>
    public static (int Height, int Width) GetAspectRatioPreservingSize(int width, int height, int maxNumPatches, int patchSize, int poolingKernel)
    {
        if (width <= 0 || height <= 0)
        {
            throw new ArgumentException("Input image has zero width or height.");
        }
        float totalPx = width * height;
        float targetPx = maxNumPatches * (patchSize * patchSize);
        float factor = MathF.Sqrt(targetPx / totalPx);
        float idealHeight = factor * height;
        float idealWidth = factor * width;
        int sideMult = poolingKernel * patchSize;
        int targetHeight = (int)MathF.Floor(idealHeight / sideMult) * sideMult;
        int targetWidth = (int)MathF.Floor(idealWidth / sideMult) * sideMult;
        if (targetHeight == 0 && targetWidth == 0)
        {
            throw new ArgumentException("Attempting to resize to a 0 x 0 image.");
        }
        int maxSideLength = maxNumPatches / (poolingKernel * poolingKernel) * sideMult;
        if (targetHeight == 0)
        {
            targetHeight = sideMult;
            targetWidth = Math.Min((int)MathF.Floor((float)width / height) * sideMult, maxSideLength);
        }
        else if (targetWidth == 0)
        {
            targetWidth = sideMult;
            targetHeight = Math.Min((int)MathF.Floor((float)height / width) * sideMult, maxSideLength);
        }
        if (targetHeight * targetWidth > targetPx)
        {
            throw new ArgumentException("Resizing exceeds max patches.");
        }
        return (targetHeight, targetWidth);
    }

    public static PatchifiedImage Preprocess(EmbeddingGemma2Image image, int tokensPerImage, int patchSize, int poolingKernel)
    {
        int maxPatches = tokensPerImage * poolingKernel * poolingKernel;
        var (th, tw) = GetAspectRatioPreservingSize(image.Width, image.Height, maxPatches, patchSize, poolingKernel);
        var rgb = th == image.Height && tw == image.Width
            ? image.Rgb
            : StbImageResize.ResizeRgb(image.Rgb, image.Width, image.Height, tw, th);
        return Patchify(rgb, th, tw, maxPatches, patchSize);
    }

    /// <summary><c>PatchifyImage</c>: [patches, p·p·3] floats (pixel / 255) and (x, y) positions, padded to
    /// <paramref name="maxPatches"/> with zeros / -1 like the vision executor's input buffers.</summary>
    public static PatchifiedImage Patchify(byte[] rgb, int height, int width, int maxPatches, int patchSize)
    {
        int ph = height / patchSize, pw = width / patchSize;
        int valid = ph * pw;
        if (valid > maxPatches)
        {
            throw new ArgumentException($"Number of patches ({valid}) exceeds max_num_patches ({maxPatches}).");
        }
        int dim = patchSize * patchSize * 3;
        var patches = new float[maxPatches * dim];
        var positions = new int[maxPatches * 2];
        Array.Fill(positions, -1);
        const float Rescale = 1.0f / 255.0f;
        for (int my = 0; my < ph; my++)
        {
            for (int mx = 0; mx < pw; mx++)
            {
                int idx = my * pw + mx;
                positions[2 * idx] = mx;
                positions[2 * idx + 1] = my;
                var dst = patches.AsSpan(idx * dim, dim);
                for (int y = 0; y < patchSize; y++)
                {
                    int src = ((my * patchSize + y) * width + mx * patchSize) * 3;
                    for (int i = 0; i < patchSize * 3; i++)
                    {
                        dst[y * patchSize * 3 + i] = rgb[src + i] * Rescale;
                    }
                }
            }
        }
        return new PatchifiedImage(patches, positions, maxPatches, valid, height, width);
    }
}
