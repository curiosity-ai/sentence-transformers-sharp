using SentenceTransformers.Quantize;
using SentenceTransformers.Ternary;

const string Usage = """
SentenceTransformers.Quantize - convert Harrier checkpoints to ternary .stq files and validate them.

  convert  --input <model.safetensors> --output <model.stq>
           [--band tq1_0|tq2_0]          storage band for the projections   (default tq1_0)
           [--embed-band tq1_0|tq2_0|f32] band for the token embedding table (default: --band)
                                         f32 leaves it unquantized (ablation)
           [--group <n>]                 weights per scale group            (default 128)
           [--method optimal|absmean|twn] group quantization rule           (default optimal)
           [--no-rotate]                 store in the original basis (ablation only)
           [--rotation-block <n>]        max Walsh-Hadamard block           (default 1024)
           [--seed <n>]                  sign-diagonal seed
           [--keep-float <name>]         leave a named tensor in float32 (repeatable)
           [--no-error-report]           skip the reconstruction measurement (faster)
           [--threads <n>]

  validate --ternary <model.stq> [--original <model.safetensors>]
           [--sentences <file>]          one sentence per line (default: a built-in multilingual set)
           [--min-cosine <x>]            default 0.99
           [--min-spearman <x>]          default 0.98
           [--max-tensor-error <x>]      default 0.35

  inspect  --ternary <model.stq>

  compare  --original <model.safetensors> [--ternary <model.stq>] [--sentences <file>]
           Scores the ternary file and the package's existing Int8/Int4 modes against the same
           fp32 baseline, so the size/quality trade-off is visible side by side.

Without --original, validate runs only the exact codec checks.
""";

var argv = args.ToList();
if (argv.Count == 0 || argv[0] is "-h" or "--help" or "help")
{
    Console.WriteLine(Usage);
    return argv.Count == 0 ? 1 : 0;
}

string command = argv[0];
argv.RemoveAt(0);

string? Get(string name)
{
    int i = argv.IndexOf(name);
    if (i < 0) return null;
    if (i + 1 >= argv.Count) throw new ArgumentException($"{name} needs a value.");
    var value = argv[i + 1];
    argv.RemoveRange(i, 2);
    return value;
}

bool Flag(string name) => argv.Remove(name);

List<string> GetAll(string name)
{
    var values = new List<string>();
    for (string? v = Get(name); v is not null; v = Get(name))
    {
        values.Add(v);
    }
    return values;
}

string Require(string name) => Get(name) ?? throw new ArgumentException($"{name} is required.");

try
{
    switch (command)
    {
        case "convert":
        {
            var input = Require("--input");
            var output = Require("--output");
            var band = ParseBand(Get("--band") ?? "tq1_0");
            var embedBand = Get("--embed-band") is { } eb ? TernaryFormat.ParseBand(eb.ToLowerInvariant()) : (TernaryBand?)null;
            var method = Enum.Parse<TernaryMethod>(Get("--method") ?? "optimal", ignoreCase: true);
            int group = int.Parse(Get("--group") ?? TernaryFormat.DefaultGroupSize.ToString());
            int rotationBlock = int.Parse(Get("--rotation-block") ?? "1024");
            ulong seed = ulong.Parse(Get("--seed") ?? "6822930717860817885");
            var keepFloat = GetAll("--keep-float");
            bool noErrorReport = Flag("--no-error-report");
            bool noRotate = Flag("--no-rotate");
            int threads = int.Parse(Get("--threads") ?? Environment.ProcessorCount.ToString());
            Reject(argv);

            var options = new ConversionOptions
            {
                InputPath = input,
                OutputPath = output,
                Band = band,
                EmbeddingBand = embedBand,
                GroupSize = group,
                Method = method,
                Rotate = !noRotate,
                MaxRotationBlock = rotationBlock,
                Seed = seed,
                KeepFloat = new HashSet<string>(keepFloat, StringComparer.Ordinal),
                SkipErrorReport = noErrorReport,
                MaxDegreeOfParallelism = threads,
            };
            return await Converter.RunAsync(options, Console.Out);
        }

        case "validate":
        {
            var ternary = Require("--ternary");
            var original = Get("--original");
            var sentencesFile = Get("--sentences");
            var thresholds = new ValidationThresholds
            {
                MinEmbeddingCosine = double.Parse(Get("--min-cosine") ?? "0.99"),
                MinSpearman = double.Parse(Get("--min-spearman") ?? "0.98"),
                MaxTensorRelativeError = double.Parse(Get("--max-tensor-error") ?? "0.35"),
            };
            Reject(argv);

            var sentences = sentencesFile is null
                ? Validator.DefaultSentences
                : (await File.ReadAllLinesAsync(sentencesFile)).Where(l => l.Trim().Length > 0).ToArray();

            return await Validator.RunAsync(original, ternary, sentences, thresholds, Console.Out);
        }

        case "compare":
        {
            var original = Require("--original");
            var ternary = Get("--ternary");
            var sentencesFile = Get("--sentences");
            Reject(argv);

            var sentences = sentencesFile is null
                ? Validator.DefaultSentences
                : (await File.ReadAllLinesAsync(sentencesFile)).Where(l => l.Trim().Length > 0).ToArray();

            return await Comparer.RunAsync(original, ternary, sentences, Console.Out);
        }

        case "inspect":
        {
            var ternary = Require("--ternary");
            Reject(argv);
            return Inspector.Run(ternary, Console.Out);
        }

        default:
            Console.Error.WriteLine($"Unknown command '{command}'.");
            Console.WriteLine(Usage);
            return 1;
    }
}
catch (ArgumentException ex)
{
    Console.Error.WriteLine($"error: {ex.Message}");
    Console.WriteLine();
    Console.WriteLine(Usage);
    return 1;
}

static TernaryBand ParseBand(string s)
{
    var band = TernaryFormat.ParseBand(s.ToLowerInvariant());
    if (!TernaryFormat.IsTernary(band))
    {
        throw new ArgumentException($"'{s}' is not a ternary band; use tq1_0 or tq2_0.");
    }
    return band;
}

static void Reject(List<string> leftovers)
{
    if (leftovers.Count > 0)
    {
        throw new ArgumentException($"unrecognized argument(s): {string.Join(' ', leftovers)}");
    }
}
