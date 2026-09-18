#nullable enable

namespace SentenceTransformers.Stq;

/// <summary>Per-tensor quantization quality, as reported by the converter.</summary>
/// <param name="RelativeError">Relative Frobenius error <c>||W - W'|| / ||W||</c> against the original
/// weights, measured after un-rotating, so it is comparable across rotated and unrotated tensors.</param>
/// <param name="RowCosine">Mean cosine similarity between each original row and its reconstruction -
/// the number that tracks end-to-end quality more closely than the Frobenius error does.</param>
/// <param name="ZeroFraction">Fraction of trits that came out zero. Around 0.3-0.5 is healthy; near 0
/// or near 1 means the scale search is degenerate for this tensor.</param>
public sealed record StqTensorStats(
    string Name,
    StqBand Band,
    int Rows,
    int InDim,
    int GroupSize,
    bool Rotated,
    double RelativeError,
    double RowCosine,
    double ZeroFraction,
    long StoredBytes);

/// <summary>
/// Turns a float weight matrix into the rotated, group-scaled, packed representation an
/// <c>.stq</c> file stores, and measures how much was lost doing it.
/// </summary>
public static class StqTensorBuilder
{
    /// <summary>The packed result plus its quality report.</summary>
    public sealed record Result(byte[] Codes, ushort[] Scales, StqTensorStats Stats);

    /// <summary>
    /// Quantizes <paramref name="weights"/> (row-major <c>[rows, inDim]</c>).
    ///
    /// <para>Order matters: the rotation is applied to each row <b>first</b> (producing
    /// <c>W R^T</c>), and the groups are formed and quantized in the rotated basis. That is the whole
    /// point - the rotation is what makes each group outlier-free enough to ternarize.</para>
    /// </summary>
    /// <param name="measureError">When false, skips the reconstruction pass (which costs a second
    /// unpack plus an inverse rotation per row) and reports zeros for the error fields.</param>
    public static async Task<Result> BuildAsync(
        string name,
        float[] weights,
        int rows,
        int inDim,
        StqBand band,
        int groupSize,
        HadamardRotation? rotation,
        TernaryMethod method,
        ParallelOptions parallelOptions,
        bool measureError = true)
    {
        if (!StqFormat.IsPacked(band))
        {
            throw new ArgumentException($"{StqFormat.BandName(band)} is not a packed band.", nameof(band));
        }
        if (inDim % groupSize != 0)
        {
            throw new ArgumentException($"Tensor '{name}': input dim {inDim} is not a multiple of the group size {groupSize}.", nameof(groupSize));
        }
        if (rotation is not null && rotation.Dim != inDim)
        {
            throw new ArgumentException($"Tensor '{name}': rotation is {rotation.Dim}-dimensional but the input dim is {inDim}.", nameof(rotation));
        }
        if ((long)rows * inDim != weights.Length)
        {
            throw new ArgumentException($"Tensor '{name}': {weights.Length} weights do not form a [{rows}, {inDim}] matrix.", nameof(weights));
        }

        int groups = inDim / groupSize;
        int groupBytes = StqFormat.CodeBytesPerGroup(band, groupSize);
        int rowBytes = groups * groupBytes;

        var codes = new byte[(long)rows * rowBytes];
        var scales = new ushort[(long)rows * groups];

        // Per-row error accumulators, summed after the parallel pass so the result does not depend on
        // scheduling order.
        var rowSqErr = measureError ? new double[rows] : null;
        var rowSqMag = measureError ? new double[rows] : null;
        var rowCos   = measureError ? new double[rows] : null;
        var rowZeros = new int[rows];

        await ParallelExecution.ForAsync(0, rows, parallelOptions, (r, _) =>
        {
            var rowBuf  = new float[inDim];
            var scratch = new float[inDim];
            var trits   = new sbyte[groupSize];
            weights.AsSpan(r * inDim, inDim).CopyTo(rowBuf);

            rotation?.Apply(rowBuf);

            int zeros = 0;
            for (int g = 0; g < groups; g++)
            {
                var slice = rowBuf.AsSpan(g * groupSize, groupSize);
                float s = StqQuantizer.QuantizeGroup(band, slice, trits, method, scratch.AsSpan(0, groupSize));

                // The scale is stored as FP16, so quantize it here too - otherwise the error we report
                // would be optimistic relative to what the runtime actually reconstructs.
                var half = (Half)s;
                scales[r * groups + g] = BitConverter.HalfToUInt16Bits(half);

                StqPacking.PackGroup(band, trits, codes.AsSpan(r * rowBytes + g * groupBytes, groupBytes));
                for (int i = 0; i < groupSize; i++)
                {
                    if (trits[i] == 0) zeros++;
                }
            }
            rowZeros[r] = zeros;

            if (measureError)
            {
                // Reconstruct exactly as a reader would: unpack, scale by the FP16 value, un-rotate.
                var recon = new float[inDim];
                for (int g = 0; g < groups; g++)
                {
                    float s = (float)BitConverter.UInt16BitsToHalf(scales[r * groups + g]);
                    StqPacking.UnpackGroupScaled(band, codes.AsSpan(r * rowBytes + g * groupBytes, groupBytes),
                                                     recon.AsSpan(g * groupSize, groupSize), groupSize, s);
                }
                rotation?.ApplyInverse(recon);

                var original = weights.AsSpan(r * inDim, inDim);
                double se = 0, mag = 0, dot = 0, rmag = 0;
                for (int i = 0; i < inDim; i++)
                {
                    double o = original[i], q = recon[i], d = o - q;
                    se += d * d;
                    mag += o * o;
                    dot += o * q;
                    rmag += q * q;
                }
                rowSqErr![r] = se;
                rowSqMag![r] = mag;
                rowCos![r] = (mag > 0 && rmag > 0) ? dot / Math.Sqrt(mag * rmag) : 1.0;
            }
            return ValueTask.CompletedTask;
        }).ConfigureAwait(false);

        double totalErr = 0, totalMag = 0, cosSum = 0;
        long zeroTotal = 0;
        for (int r = 0; r < rows; r++)
        {
            if (measureError)
            {
                totalErr += rowSqErr![r];
                totalMag += rowSqMag![r];
                cosSum += rowCos![r];
            }
            zeroTotal += rowZeros[r];
        }

        var stats = new StqTensorStats(
            name, band, rows, inDim, groupSize, rotation is not null,
            RelativeError: measureError && totalMag > 0 ? Math.Sqrt(totalErr / totalMag) : 0.0,
            RowCosine: measureError ? cosSum / rows : 0.0,
            ZeroFraction: (double)zeroTotal / ((long)rows * inDim),
            StoredBytes: codes.LongLength + scales.LongLength * 2);

        return new Result(codes, scales, stats);
    }
}
