using SentenceTransformers;

namespace SentenceTransformers.LoraTraining;

/// <summary>
/// Instantiates any of the library's models by short name, so the training CLI can fine-tune every
/// supported encoder through the single model-agnostic LoRA pipeline. Embedded models load instantly;
/// downloaded models fetch their weights on first use.
/// </summary>
internal static class EncoderFactory
{
    public static readonly string[] Names =
    {
        "minilm", "arctic", "qwen3", "harrier-medium", "harrier-small", "harrier-small-pure",
    };

    /// <summary>Creates the encoder identified by <paramref name="name"/> (case-insensitive).</summary>
    public static async Task<ISentenceEncoder> CreateAsync(string name, CancellationToken cancellationToken = default)
    {
        Action<DownloadProgress> progress = p =>
        {
            if (p.TotalBytes is > 0)
            {
                Console.Error.Write($"\r  downloading {name}: {p.Fraction * 100f,5:0.0}%   ");
            }
        };

        switch ((name ?? "").Trim().ToLowerInvariant())
        {
            case "minilm":
                return new SentenceTransformers.MiniLM.SentenceEncoder();

            case "arctic":
            case "arcticxs":
                return new SentenceTransformers.ArcticXs.SentenceEncoder();

            case "qwen3":
            {
                var e = await SentenceTransformers.Qwen3.SentenceEncoder.CreateAsync(reportProgress: progress, cancellationToken: cancellationToken);
                Console.Error.WriteLine();
                return e;
            }

            case "harrier-medium":
            {
                var e = await SentenceTransformers.Harrier.Medium.SentenceEncoder.CreateAsync(reportProgress: progress, cancellationToken: cancellationToken);
                Console.Error.WriteLine();
                return e;
            }

            case "harrier-small":
            {
                var e = await SentenceTransformers.Harrier.Small.SentenceEncoder.CreateAsync(reportProgress: progress, cancellationToken: cancellationToken);
                Console.Error.WriteLine();
                return e;
            }

            case "harrier-small-pure":
            {
                var e = await SentenceTransformers.Harrier.Small.Pure.SentenceEncoder.CreateAsync(
                    reportProgress: progress,
                    parallelOptions: new ParallelOptions { CancellationToken = cancellationToken });
                Console.Error.WriteLine();
                return e;
            }

            default:
                throw new ArgumentException($"Unknown model '{name}'. Known models: {string.Join(", ", Names)}.");
        }
    }
}
