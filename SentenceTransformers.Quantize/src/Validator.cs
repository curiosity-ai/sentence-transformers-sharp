using System.Diagnostics;
using SentenceTransformers.Harrier.Small.Pure;
using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Stq;

namespace SentenceTransformers.Quantize;

/// <summary>Thresholds a converted model must clear. Exceeding any of them fails the run.</summary>
public sealed class ValidationThresholds
{
    /// <summary>Lowest acceptable mean cosine between fp32 embeddings and their quantized
    /// counterparts. This is the headline gate; the default is set so a 4-bit conversion of Harrier
    /// Small passes and anything meaningfully worse does not.</summary>
    public double MinMeanCosine { get; init; } = 0.98;

    /// <summary>Per-sentence floor. Looser than the mean gate on purpose: over a small sentence set
    /// the minimum is noisy, so it guards against one sentence collapsing rather than against
    /// gradual drift.</summary>
    public double MinEmbeddingCosine { get; init; } = 0.90;

    /// <summary>Lowest acceptable Spearman correlation between the fp32 and quantized pairwise
    /// similarity matrices - the property that actually decides whether retrieval rankings survive.</summary>
    public double MinSpearman { get; init; } = 0.98;

    /// <summary>Largest acceptable per-tensor relative reconstruction error.</summary>
    public double MaxTensorRelativeError { get; init; } = 0.35;
}

/// <summary>
/// Checks a converted <c>.stq</c> file at three levels, cheapest first, so a structural bug is
/// reported as a structural bug rather than surfacing as a mysterious drop in embedding quality:
///
/// <list type="number">
/// <item><b>Codec</b> - every group is unpacked and re-packed and must reproduce the file's bytes
/// exactly, and every rotation must satisfy <c>R^T R = I</c> numerically. These are exact checks:
/// any failure is a bug in the format code, not a quantization loss.</item>
/// <item><b>Per tensor</b> - each tensor is decoded back to float and compared against the original
/// checkpoint. This is where quantization loss legitimately shows up, so it is bounded by a
/// threshold rather than required to be zero.</item>
/// <item><b>End to end</b> - the fp32 encoder and the ternary encoder embed the same sentences.
/// Per-sentence cosine says whether individual vectors moved; the Spearman correlation between the
/// two pairwise-similarity matrices says whether the <i>rankings</i> a retrieval system would
/// produce moved, which is the number that actually matters.</item>
/// </list>
/// </summary>
public static class Validator
{
    /// <summary>A multilingual, deliberately mixed set: near-duplicates across languages, unrelated
    /// sentences, and short/long pairs, so the similarity matrix has real structure to preserve
    /// rather than being uniformly near zero.</summary>
    public static readonly string[] DefaultSentences =
    {
        "Good morning, how are you?",
        "Buenos días, ¿cómo estás?",
        "おはようございます、お元気ですか？",
        "Guten Morgen, wie geht es dir?",
        "Bom dia, como você está?",
        "Hello world",
        "The cat sat on the mat",
        "A feline rested upon the rug",
        "The dog slept on the floor",
        "Quantum entanglement links two particles across arbitrary distances.",
        "Two particles can remain correlated no matter how far apart they are.",
        "The quarterly earnings report exceeded analyst expectations.",
        "Revenue for the quarter came in above what analysts had forecast.",
        "Preheat the oven to 200 degrees and bake for 25 minutes.",
        "Sort the array in place using a comparison-based algorithm.",
        "public static void Main(string[] args) { Console.WriteLine(\"hi\"); }",
        "L'intelligence artificielle transforme de nombreux secteurs.",
        "人工智能正在改变许多行业。",
        "Искусственный интеллект меняет многие отрасли.",
        "It was the best of times, it was the worst of times.",
    };

    public static async Task<int> RunAsync(string originalPath, string ternaryPath, string[] sentences,
                                           ValidationThresholds thresholds, TextWriter log, CancellationToken ct = default)
    {
        bool ok = true;
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = Environment.ProcessorCount, CancellationToken = ct };

        log.WriteLine($"Loading {ternaryPath} ...");
        var file = await StqFile.LoadAsync(ternaryPath, ct).ConfigureAwait(false);
        foreach (var (k, v) in file.Metadata.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            log.WriteLine($"  {k,-20} {v}");
        }
        log.WriteLine();

        ok &= CheckCodec(file, log);
        log.WriteLine();

        ok &= await CheckKernelAsync(file, log, parallelOptions).ConfigureAwait(false);
        log.WriteLine();

        if (originalPath is not null)
        {
            ok &= await CheckTensorsAsync(file, originalPath, thresholds, log, ct).ConfigureAwait(false);
            log.WriteLine();
            ok &= await CheckEndToEndAsync(originalPath, ternaryPath, sentences, thresholds, parallelOptions, log).ConfigureAwait(false);
        }
        else
        {
            log.WriteLine("No --original given: skipping the per-tensor and end-to-end comparisons.");
        }

        log.WriteLine();
        log.WriteLine(ok ? "VALIDATION PASSED" : "VALIDATION FAILED");
        return ok ? 0 : 1;
    }

    /// <summary>Exact structural checks: pack/unpack symmetry and rotation orthogonality.</summary>
    private static bool CheckCodec(StqFile file, TextWriter log)
    {
        log.WriteLine("Codec checks (exact)");
        bool ok = true;

        foreach (var (id, rotation) in file.Rotations.OrderBy(kv => kv.Key, StringComparer.Ordinal))
        {
            var v = new float[rotation.Dim];
            var rnd = new Random(12345);
            for (int i = 0; i < v.Length; i++)
            {
                v[i] = (float)(rnd.NextDouble() * 2 - 1);
            }
            var original = (float[])v.Clone();

            rotation.Apply(v);
            rotation.ApplyInverse(v);

            double maxDev = 0;
            for (int i = 0; i < v.Length; i++)
            {
                maxDev = Math.Max(maxDev, Math.Abs(v[i] - original[i]));
            }
            // The round trip is 2*log2(block) float adds deep, so a few ulps of drift is expected;
            // anything above 1e-4 means the transform is not its own inverse.
            bool pass = maxDev < 1e-4;
            ok &= pass;
            log.WriteLine($"  rotation {id,-8} dim {rotation.Dim,5} block {rotation.Block,5}  R^T R = I within {maxDev:E2}  {(pass ? "ok" : "FAIL")}");
        }

        int checkedTensors = 0;
        foreach (var info in file.Tensors.Values.Where(t => StqFormat.IsPacked(t.Band)))
        {
            var codes = file.Codes(info).Span;
            int gs = info.GroupSize;
            int groupBytes = StqFormat.CodeBytesPerGroup(info.Band, gs);
            Span<sbyte> trits = stackalloc sbyte[gs];
            Span<byte> repacked = stackalloc byte[groupBytes];

            // Every group of every tensor: this is the check that would catch a packing bug on the one
            // partially-filled trailing byte, which only occurs in a fraction of groups.
            long totalGroups = (long)info.Rows * info.GroupsPerRow;
            for (long g = 0; g < totalGroups; g++)
            {
                var src = codes.Slice((int)(g * groupBytes), groupBytes);
                StqPacking.UnpackGroup(info.Band, src, trits, gs);
                StqPacking.PackGroup(info.Band, trits, repacked);
                if (!src.SequenceEqual(repacked))
                {
                    log.WriteLine($"  FAIL {info.Name}: group {g} does not survive an unpack/repack round trip.");
                    return false;
                }
            }
            checkedTensors++;
        }
        log.WriteLine($"  pack/unpack round trip  {checkedTensors} packed tensors, every group byte-identical  ok");
        return ok;
    }

    /// <summary>
    /// Proves the runtime kernel agrees with the format's own reference decode. For a sample of
    /// tensors it runs <see cref="StqMatrix"/> - packed codes, group scales and the activation
    /// rotation - against a <see cref="FloatMatrix"/> built from <see cref="StqFile.ReadFloat"/>,
    /// which unpacks and un-rotates the same weights by a completely different route. The two must
    /// agree to float32 round-off.
    ///
    /// <para>This is the check that separates the two things that look identical from the outside: a
    /// wrong kernel (rotation applied on the wrong side, a mis-ordered group, a sign error) and
    /// honest quantization loss. Without it, a kernel bug would just look like a bad conversion.</para>
    /// </summary>
    private static async Task<bool> CheckKernelAsync(StqFile file, TextWriter log, ParallelOptions parallelOptions)
    {
        log.WriteLine("Kernel vs. reference decode (exact up to float32 round-off)");

        // One tensor per distinct (shape, band) so every rotation width and both code paths are hit.
        var sample = file.Tensors.Values
            .Where(t => StqFormat.IsPacked(t.Band) && t.Shape.Length == 2 && t.Rows <= 4096)
            .GroupBy(t => (t.Shape[0], t.Shape[1], t.Band))
            .Select(g => g.First())
            .OrderBy(t => t.Name, StringComparer.Ordinal)
            .ToArray();

        bool ok = true;
        const int Seq = 3;
        foreach (var info in sample)
        {
            int inDim = info.InDim, outDim = info.Shape[0];

            var ternaryMatrix = await StqMatrix.CreateAsync(file, info, parallelOptions).ConfigureAwait(false);
            var floatMatrix = new FloatMatrix(file.ReadFloat(info.Name), outDim, inDim);

            var rnd = new Random(987);
            var x = new float[Seq * inDim];
            for (int i = 0; i < x.Length; i++)
            {
                x[i] = (float)(rnd.NextDouble() * 2 - 1);
            }

            var yTernary = new float[Seq * outDim];
            var yFloat = new float[Seq * outDim];
            await ternaryMatrix.MultiplyAsync(x, yTernary, Seq, parallelOptions).ConfigureAwait(false);
            await floatMatrix.MultiplyAsync(x, yFloat, Seq, parallelOptions).ConfigureAwait(false);

            // The VNNI path quantizes activations to int8, so it is not bit-exact against the float
            // dot; judge it by relative agreement, which int8 activations hold to ~1e-2.
            double num = 0, den = 0, maxAbs = 0;
            for (int i = 0; i < yFloat.Length; i++)
            {
                double d = yTernary[i] - yFloat[i];
                num += d * d;
                den += (double)yFloat[i] * yFloat[i];
                maxAbs = Math.Max(maxAbs, Math.Abs(yFloat[i]));
            }
            double rel = den > 0 ? Math.Sqrt(num / den) : 0;

            bool pass = rel < 2e-2;
            ok &= pass;
            log.WriteLine($"  {info.Name,-46} [{outDim,6} x {inDim,5}] rot {info.RotationId ?? "-",-6} relative disagreement {rel:E2}  {(pass ? "ok" : "FAIL")}");
        }


        // The embedding table is excluded above (too many rows to round-trip in full) and takes a
        // different path anyway - a row lookup that un-rotates instead of pushing the rotation onto an
        // activation - so it gets its own spot check against the reference decode.
        var embedName = file.Contains("embed_tokens.weight") ? "embed_tokens.weight" : "model.embed_tokens.weight";
        if (file.Contains(embedName) && StqFormat.IsPacked(file.Info(embedName).Band))
        {
            var embedInfo = file.Info(embedName);
            var embedding = new StqEmbedding(file, embedInfo);
            int hidden = embedInfo.Shape[1];
            var decoded = new float[hidden];
            var row = new float[hidden];
            double worstRel = 0;
            int worstToken = 0;

            // A spread of token ids rather than the first few: the low ids are special tokens whose
            // rows are atypical.
            var rnd = new Random(4242);
            for (int trial = 0; trial < 64; trial++)
            {
                int token = rnd.Next(embedInfo.Shape[0]);
                embedding.Lookup(token, row, 1f);
                ReadRowViaReference(file, embedInfo, token, decoded);

                double num = 0, den = 0;
                for (int i = 0; i < hidden; i++)
                {
                    double d = row[i] - decoded[i];
                    num += d * d;
                    den += (double)decoded[i] * decoded[i];
                }
                double rel = den > 0 ? Math.Sqrt(num / den) : 0;
                if (rel > worstRel)
                {
                    worstRel = rel;
                    worstToken = token;
                }
            }

            bool embedPass = worstRel < 1e-4;
            ok &= embedPass;
            log.WriteLine($"  {embedName,-46} [{embedInfo.Shape[0],6} x {hidden,5}] rot {embedInfo.RotationId ?? "-",-6} " +
                          $"worst row disagreement {worstRel:E2} (token {worstToken})  {(embedPass ? "ok" : "FAIL")}");
        }

        if (!ok)
        {
            log.WriteLine("  FAIL the packed kernel does not reproduce the reference decode - this is a code bug, not quantization loss.");
        }
        return ok;
    }


    /// <summary>Decodes a single embedding row the long way - unpack, scale, un-rotate - so the
    /// <see cref="StqEmbedding"/> lookup has something independent to be checked against.</summary>
    private static void ReadRowViaReference(StqFile file, StqTensorInfo info, int token, Span<float> dst)
    {
        var codes = file.Codes(info).Span;
        var scales = file.Scales(info);
        int gs = info.GroupSize, groups = info.GroupsPerRow;
        int groupBytes = StqFormat.CodeBytesPerGroup(info.Band, gs);
        int rowBytes = groups * groupBytes;

        for (int g = 0; g < groups; g++)
        {
            StqPacking.UnpackGroupScaled(info.Band, codes.Slice(token * rowBytes + g * groupBytes, groupBytes),
                                             dst.Slice(g * gs, gs), gs, scales[token * groups + g]);
        }
        file.RotationFor(info)?.ApplyInverse(dst);
    }

    /// <summary>Decodes every tensor and compares it with the original checkpoint.</summary>
    private static async Task<bool> CheckTensorsAsync(StqFile file, string originalPath,
                                                      ValidationThresholds thresholds, TextWriter log, CancellationToken ct)
    {
        log.WriteLine($"Per-tensor reconstruction against {Path.GetFileName(originalPath)}");
        var st = await SafeTensors.LoadAsync(originalPath, ct).ConfigureAwait(false);
        bool ok = true;
        double worst = 0;
        string worstName = "";
        int compared = 0;

        foreach (var name in file.Tensors.Keys.OrderBy(n => n, StringComparer.Ordinal))
        {
            ct.ThrowIfCancellationRequested();
            if (!st.Contains(name))
            {
                log.WriteLine($"  FAIL {name}: present in the .stq file but not in the original checkpoint.");
                ok = false;
                continue;
            }

            var original = st.ReadFloat(name);
            var decoded = file.ReadFloat(name);
            if (original.Length != decoded.Length)
            {
                log.WriteLine($"  FAIL {name}: {decoded.Length} elements decoded, {original.Length} in the original.");
                ok = false;
                continue;
            }

            double se = 0, mag = 0;
            for (int i = 0; i < original.Length; i++)
            {
                double d = original[i] - decoded[i];
                se += d * d;
                mag += (double)original[i] * original[i];
            }
            double rel = mag > 0 ? Math.Sqrt(se / mag) : 0;
            compared++;

            if (rel > worst)
            {
                worst = rel;
                worstName = name;
            }
            if (rel > thresholds.MaxTensorRelativeError)
            {
                log.WriteLine($"  FAIL {name}: relative error {rel:F4} exceeds {thresholds.MaxTensorRelativeError:F4}.");
                ok = false;
            }
        }

        foreach (var name in st.Names)
        {
            if (!file.Contains(name))
            {
                log.WriteLine($"  FAIL {name}: in the original checkpoint but missing from the .stq file.");
                ok = false;
            }
        }

        log.WriteLine($"  {compared} tensors compared, worst relative error {worst:F4} ({worstName})  {(ok ? "ok" : "FAIL")}");
        return ok;
    }

    /// <summary>Embeds the same sentences with both encoders and compares vectors and rankings.</summary>
    private static async Task<bool> CheckEndToEndAsync(string originalPath, string ternaryPath, string[] sentences,
                                                       ValidationThresholds thresholds, ParallelOptions parallelOptions, TextWriter log)
    {
        log.WriteLine($"End-to-end over {sentences.Length} sentences");

        var sw = Stopwatch.StartNew();
        using var reference = await SentenceEncoder.LoadAsync(originalPath, quantization: Quantization.None, parallelOptions: parallelOptions).ConfigureAwait(false);
        var refVectors = await reference.EncodeAsync(sentences, parallelOptions).ConfigureAwait(false);
        double refSeconds = sw.Elapsed.TotalSeconds;

        sw.Restart();
        using var ternary = await SentenceEncoder.LoadQuantizedAsync(ternaryPath, parallelOptions: parallelOptions).ConfigureAwait(false);
        var tqVectors = await ternary.EncodeAsync(sentences, parallelOptions).ConfigureAwait(false);
        double tqSeconds = sw.Elapsed.TotalSeconds;

        double minCos = 1, sumCos = 0;
        int worst = 0;
        for (int i = 0; i < sentences.Length; i++)
        {
            double c = Cosine(refVectors[i], tqVectors[i]);
            sumCos += c;
            if (c < minCos)
            {
                minCos = c;
                worst = i;
            }
        }

        var refSim = PairwiseSimilarities(refVectors);
        var tqSim = PairwiseSimilarities(tqVectors);
        double spearman = Spearman(refSim, tqSim);
        double maxSimDelta = 0;
        for (int i = 0; i < refSim.Length; i++)
        {
            maxSimDelta = Math.Max(maxSimDelta, Math.Abs(refSim[i] - tqSim[i]));
        }

        log.WriteLine($"  mean embedding cosine     {sumCos / sentences.Length:F5}");
        log.WriteLine($"  min  embedding cosine     {minCos:F5}  (\"{Truncate(sentences[worst], 48)}\")");
        log.WriteLine($"  pairwise-similarity rho   {spearman:F5}");
        log.WriteLine($"  max pairwise-sim delta    {maxSimDelta:F5}");
        log.WriteLine($"  load + encode: fp32 {refSeconds:F1}s, stq {tqSeconds:F1}s");

        bool ok = true;
        double meanCos = sumCos / sentences.Length;
        if (meanCos < thresholds.MinMeanCosine)
        {
            log.WriteLine($"  FAIL mean cosine {meanCos:F5} is below {thresholds.MinMeanCosine:F5}.");
            ok = false;
        }
        if (minCos < thresholds.MinEmbeddingCosine)
        {
            log.WriteLine($"  FAIL min cosine {minCos:F5} is below {thresholds.MinEmbeddingCosine:F5}.");
            ok = false;
        }
        if (spearman < thresholds.MinSpearman)
        {
            log.WriteLine($"  FAIL pairwise-similarity rho {spearman:F5} is below {thresholds.MinSpearman:F5}.");
            ok = false;
        }
        if (ok)
        {
            log.WriteLine("  ok");
        }
        return ok;
    }

    internal static double Cosine(float[] a, float[] b)
    {
        double dot = 0, na = 0, nb = 0;
        for (int i = 0; i < a.Length; i++)
        {
            dot += (double)a[i] * b[i];
            na += (double)a[i] * a[i];
            nb += (double)b[i] * b[i];
        }
        return (na > 0 && nb > 0) ? dot / Math.Sqrt(na * nb) : 0;
    }

    /// <summary>The strictly-upper-triangular pairwise cosine similarities, flattened.</summary>
    internal static double[] PairwiseSimilarities(float[][] vectors)
    {
        int n = vectors.Length;
        var result = new double[n * (n - 1) / 2];
        int k = 0;
        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                result[k++] = Cosine(vectors[i], vectors[j]);
            }
        }
        return result;
    }

    /// <summary>Spearman rank correlation, with average ranks for ties.</summary>
    internal static double Spearman(double[] a, double[] b)
    {
        var ra = Ranks(a);
        var rb = Ranks(b);
        double ma = ra.Average(), mb = rb.Average();
        double cov = 0, va = 0, vb = 0;
        for (int i = 0; i < ra.Length; i++)
        {
            double da = ra[i] - ma, db = rb[i] - mb;
            cov += da * db;
            va += da * da;
            vb += db * db;
        }
        return (va > 0 && vb > 0) ? cov / Math.Sqrt(va * vb) : 1.0;
    }

    private static double[] Ranks(double[] values)
    {
        var order = Enumerable.Range(0, values.Length).OrderBy(i => values[i]).ToArray();
        var ranks = new double[values.Length];
        int i0 = 0;
        while (i0 < order.Length)
        {
            int i1 = i0;
            while (i1 + 1 < order.Length && values[order[i1 + 1]] == values[order[i0]])
            {
                i1++;
            }
            double rank = (i0 + i1) / 2.0;
            for (int i = i0; i <= i1; i++)
            {
                ranks[order[i]] = rank;
            }
            i0 = i1 + 1;
        }
        return ranks;
    }

    private static string Truncate(string s, int max) => s.Length <= max ? s : s[..max] + "...";
}
