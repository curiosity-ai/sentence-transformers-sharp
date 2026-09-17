using System.Diagnostics;
using System.Security.Cryptography;
using SentenceTransformers.Harrier.Small.Pure.Model;
using SentenceTransformers.Ternary;

namespace SentenceTransformers.Quantize;

/// <summary>
/// Converts a safetensors checkpoint into an <c>.stq</c> ternary file.
///
/// <para><b>Which tensors get quantized.</b> The rule is structural, not a hard-coded name list: any
/// rank-2 tensor whose last dimension is a multiple of the group size becomes ternary; everything
/// else (every rank-1 norm vector) is written verbatim as float32. For Harrier Small that quantizes
/// the embedding table and all seven projections per layer - 268.0M of the 268.1M parameters - and
/// leaves the 82 norm vectors alone. Keeping the norms in full precision is the same call PrismML
/// make for Bonsai 2: they are a rounding error in size and the most numerically load-bearing
/// parameters in the model.</para>
///
/// <para><b>Rotations are shared by input dimension.</b> One rotation is created per distinct input
/// width (640, 1024, 2048 for Harrier Small), so every tensor consuming a 640-wide activation is
/// stored under the same basis. That keeps the file small and leaves the door open for a runtime
/// that rotates the hidden state once and feeds q, k and v from it.</para>
/// </summary>
public static class Converter
{
    public static async Task<int> RunAsync(ConversionOptions options, TextWriter log, CancellationToken ct = default)
    {
        var sw = Stopwatch.StartNew();
        var parallelOptions = new ParallelOptions { MaxDegreeOfParallelism = options.MaxDegreeOfParallelism, CancellationToken = ct };

        log.WriteLine($"Reading {options.InputPath} ...");
        var st = await SafeTensors.LoadAsync(options.InputPath, ct).ConfigureAwait(false);
        var sourceInfo = new FileInfo(options.InputPath);

        var writer = new TernaryModelWriter()
            .SetMetadata("producer", "SentenceTransformers.Quantize")
            .SetMetadata("architecture", Gemma3Model.TernaryArchitecture)
            .SetMetadata("source", Path.GetFileName(options.InputPath))
            .SetMetadata("source_bytes", sourceInfo.Length.ToString())
            .SetMetadata("source_sha256", await Sha256Async(options.InputPath, ct).ConfigureAwait(false))
            .SetMetadata("band", TernaryFormat.BandName(options.Band))
            .SetMetadata("embedding_band", TernaryFormat.BandName(options.EmbeddingBand ?? options.Band))
            .SetMetadata("group_size", options.GroupSize.ToString())
            .SetMetadata("method", options.Method.ToString().ToLowerInvariant())
            .SetMetadata("rotation", options.Rotate ? "hadamard" : "none")
            .SetMetadata("rotation_max_block", options.MaxRotationBlock.ToString())
            .SetMetadata("rotation_seed", options.Seed.ToString())
            .SetMetadata("created_utc", DateTime.UtcNow.ToString("O"));

        // One rotation per distinct input width, created lazily and referenced by id.
        var rotations = new Dictionary<int, (string Id, TernaryRotation Rotation)>();
        TernaryRotation? RotationFor(int inDim)
        {
            if (!options.Rotate)
            {
                return null;
            }
            if (rotations.TryGetValue(inDim, out var existing))
            {
                return existing.Rotation;
            }

            int block = TernaryRotation.ChooseBlock(inDim, options.MaxRotationBlock);
            if (block < 2)
            {
                log.WriteLine($"  (no rotation for width {inDim}: no usable power-of-two block)");
                return null;
            }

            // Vary the seed per width so two dimensions never share a sign pattern.
            var rotation = TernaryRotation.Create(inDim, block, options.Seed ^ (ulong)inDim * 0x9E3779B97F4A7C15UL);
            string id = $"h{inDim}";
            rotations[inDim] = (id, rotation);
            writer.AddRotation(id, rotation);
            log.WriteLine($"  rotation {id}: dim {inDim}, Hadamard block {block}");
            return rotation;
        }

        var stats = new List<TernaryTensorStats>();
        long rawBytes = 0, storedBytes = 0, ternaryParams = 0, floatParams = 0;

        // Sorted so the data section's layout is deterministic across runs.
        foreach (var name in st.Names.OrderBy(n => n, StringComparer.Ordinal))
        {
            ct.ThrowIfCancellationRequested();
            var shape = st.Shape(name);
            long count = 1;
            foreach (int d in shape) count *= d;

            bool quantize = shape.Length == 2
                         && shape[1] % options.GroupSize == 0
                         && !options.KeepFloat.Contains(name)
                         && TernaryFormat.IsTernary(options.BandFor(name));

            if (!quantize)
            {
                var data = st.ReadFloat(name);
                writer.AddRaw(name, TernaryBand.F32, shape, data);
                floatParams += count;
                rawBytes += count * 2;      // the source is bf16
                storedBytes += count * 4;
                continue;
            }

            var band = options.BandFor(name);
            int rows = shape[0], inDim = shape[1];
            var rotation = RotationFor(inDim);

            var weights = st.ReadFloat(name);
            var result = await TernaryTensorBuilder.BuildAsync(
                name, weights, rows, inDim, band, options.GroupSize, rotation, options.Method,
                parallelOptions, measureError: !options.SkipErrorReport).ConfigureAwait(false);

            writer.AddTernary(name, band, shape, options.GroupSize, rotation is null ? null : rotations[inDim].Id, result.Codes, result.Scales);

            stats.Add(result.Stats);
            ternaryParams += count;
            rawBytes += count * 2;
            storedBytes += result.Stats.StoredBytes;

            log.WriteLine(options.SkipErrorReport
                ? $"  {name,-46} [{rows,6} x {inDim,5}] {TernaryFormat.BandName(band),-6} {result.Stats.StoredBytes / 1024.0 / 1024.0,7:F2} MB"
                : $"  {name,-46} [{rows,6} x {inDim,5}] {TernaryFormat.BandName(band),-6} {result.Stats.StoredBytes / 1024.0 / 1024.0,7:F2} MB  " +
                  $"relerr {result.Stats.RelativeError,6:F4}  cos {result.Stats.RowCosine,7:F5}  zeros {result.Stats.ZeroFraction,5:P1}");
        }

        log.WriteLine($"Writing {options.OutputPath} ...");
        await writer.WriteAsync(options.OutputPath, ct).ConfigureAwait(false);

        var outInfo = new FileInfo(options.OutputPath);
        log.WriteLine();
        log.WriteLine("Conversion summary");
        log.WriteLine($"  ternary parameters   {ternaryParams:N0} ({TernaryFormat.BandName(options.Band)}, group {options.GroupSize}, {options.Method.ToString().ToLowerInvariant()})");
        log.WriteLine($"  float32 parameters   {floatParams:N0}");
        log.WriteLine($"  source file          {sourceInfo.Length / 1024.0 / 1024.0:F1} MB");
        log.WriteLine($"  output file          {outInfo.Length / 1024.0 / 1024.0:F1} MB  ({(double)sourceInfo.Length / outInfo.Length:F2}x smaller)");
        log.WriteLine($"  effective bits/weight {(double)storedBytes * 8 / (ternaryParams + floatParams):F3} over all parameters");
        if (!options.SkipErrorReport && stats.Count > 0)
        {
            log.WriteLine($"  worst tensor relerr  {stats.Max(s => s.RelativeError):F4} ({stats.MaxBy(s => s.RelativeError)!.Name})");
            log.WriteLine($"  worst tensor cosine  {stats.Min(s => s.RowCosine):F5} ({stats.MinBy(s => s.RowCosine)!.Name})");
            log.WriteLine($"  mean zero fraction   {stats.Average(s => s.ZeroFraction):P1}");
        }
        log.WriteLine($"  elapsed              {sw.Elapsed.TotalSeconds:F1}s");
        _ = rawBytes;
        return 0;
    }

    private static async Task<string> Sha256Async(string path, CancellationToken ct)
    {
        await using var fs = File.OpenRead(path);
        var hash = await SHA256.HashDataAsync(fs, ct).ConfigureAwait(false);
        return Convert.ToHexStringLower(hash);
    }
}
