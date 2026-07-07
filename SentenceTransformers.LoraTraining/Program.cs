using System.Globalization;
using SentenceTransformers;
using SentenceTransformers.LoraTraining;
using SentenceTransformers.Training;

// ---------------------------------------------------------------------------------------------------
// SentenceTransformers.LoraTraining — a small CLI to fine-tune any supported model with a LoRA-style
// adapter on related sentence pairs, split into training and validation sets.
//
//   dotnet run -- download [--data ./data]
//   dotnet run -- train    [--model minilm] [--data ./data] [--out ./adapters/minilm.lora] [options]
//   dotnet run -- eval      --model minilm  --adapter ./adapters/minilm.lora [--data ./data] [--split test]
//
// Run `dotnet run -- help` for the full option list.
// ---------------------------------------------------------------------------------------------------

var argList = args.ToList();
string command = argList.Count > 0 && !argList[0].StartsWith('-') ? argList[0].ToLowerInvariant() : "help";

try
{
    switch (command)
    {
        case "download": return await RunDownloadAsync(argList);
        case "train":    return await RunTrainAsync(argList);
        case "eval":     return await RunEvalAsync(argList);
        case "help":
        case "-h":
        case "--help":   PrintHelp(); return 0;
        default:
            Console.Error.WriteLine($"Unknown command '{command}'.\n");
            PrintHelp();
            return 1;
    }
}
catch (Exception ex)
{
    Console.Error.WriteLine($"\nError: {ex.Message}");
    return 1;
}

// ---------------------------------------------------------------------------------------------------

async Task<int> RunDownloadAsync(List<string> a)
{
    string dir = GetOption(a, "--data", "./data");
    Console.WriteLine($"Downloading STS Benchmark into '{Path.GetFullPath(dir)}' ...");
    await StsbDataset.DownloadAsync(dir);
    Console.WriteLine("Done.");
    return 0;
}

async Task<int> RunTrainAsync(List<string> a)
{
    string model       = GetOption(a, "--model", "minilm");
    string dir         = GetOption(a, "--data", "./data");
    string datasetName = GetOption(a, "--dataset", "stsb").ToLowerInvariant();
    string outPath     = GetOption(a, "--out", $"./adapters/{model}-{datasetName}.lora");
    int    maxTrain    = int.Parse(GetOption(a, "--max-train", "0"), CultureInfo.InvariantCulture);

    var options = new LoraTrainingOptions
    {
        Rank                   = int.Parse(GetOption(a, "--rank", "16"), CultureInfo.InvariantCulture),
        Epochs                 = int.Parse(GetOption(a, "--epochs", "20"), CultureInfo.InvariantCulture),
        BatchSize              = int.Parse(GetOption(a, "--batch", "32"), CultureInfo.InvariantCulture),
        LearningRate           = float.Parse(GetOption(a, "--lr", "0.001"), CultureInfo.InvariantCulture),
        Temperature            = float.Parse(GetOption(a, "--temp", "0.05"), CultureInfo.InvariantCulture),
        WeightDecay            = float.Parse(GetOption(a, "--weight-decay", "0.0001"), CultureInfo.InvariantCulture),
        ValidationFraction     = float.Parse(GetOption(a, "--val-frac", "0.1"), CultureInfo.InvariantCulture),
        PositiveScoreThreshold = float.Parse(GetOption(a, "--pos-threshold", "0.6"), CultureInfo.InvariantCulture),
        Seed                   = int.Parse(GetOption(a, "--seed", "42"), CultureInfo.InvariantCulture),
    };

    string alpha = GetOption(a, "--alpha", null);
    if (alpha is not null) options.Alpha = float.Parse(alpha, CultureInfo.InvariantCulture);

    SentencePairDataset dataset;
    if (datasetName == "patent")
    {
        dataset = PatentDataset.Load("train");
    }
    else
    {
        string trainCsv = StsbDataset.TrainPath(dir);
        if (!File.Exists(trainCsv))
        {
            Console.Error.WriteLine($"Training data not found at '{trainCsv}'. Run `download` first (or use --dataset patent).");
            return 1;
        }
        dataset = StsbDataset.Load(trainCsv);
    }

    if (maxTrain > 0 && dataset.Count > maxTrain)
    {
        dataset = new SentencePairDataset(dataset.Pairs.Take(maxTrain));
    }

    Console.WriteLine($"Model:            {model}");
    Console.WriteLine($"Dataset:          {datasetName}");
    Console.WriteLine($"Training pairs:   {dataset.Count} (positive threshold {options.PositiveScoreThreshold:0.00}, val fraction {options.ValidationFraction:0.00})");
    Console.WriteLine($"Adapter:          rank {options.Rank}, alpha {options.Alpha?.ToString(CultureInfo.InvariantCulture) ?? options.Rank.ToString()}");
    Console.WriteLine($"Optimizer:        AdamW lr {options.LearningRate}, weight decay {options.WeightDecay}, {options.Epochs} epochs, batch {options.BatchSize}, temp {options.Temperature}");
    Console.WriteLine();
    Console.WriteLine("Loading base encoder ...");

    using var encoder = await EncoderFactory.CreateAsync(model);

    Console.WriteLine("Embedding dataset and training ...\n");
    Console.WriteLine($"{"epoch",5} {"train_loss",12} {"val_loss",10} {"val_acc",9} {"val_spearman",13}");
    Console.WriteLine(new string('-', 52));

    options.OnEpoch = m =>
    {
        Console.WriteLine($"{m.Epoch,5} {Fmt(m.TrainLoss),12} {Fmt(m.ValidationLoss),10} {Fmt(m.ValidationAccuracy),9} {Fmt(m.ValidationSpearman),13}{(m.IsBest ? "  *best" : "")}");
    };

    var report = await LoraTrainer.TrainAsync(encoder, dataset, options);

    Console.WriteLine();
    Console.WriteLine("Validation summary (base -> tuned):");
    Console.WriteLine($"  retrieval accuracy: {Fmt(report.BaselineAccuracy)} -> {Fmt(report.BestAccuracy)}");
    if (!float.IsNaN(report.BaselineSpearman))
    {
        Console.WriteLine($"  STS spearman:       {Fmt(report.BaselineSpearman)} -> {Fmt(report.BestSpearman)}");
    }

    report.Adapter.Save(outPath);
    Console.WriteLine();
    Console.WriteLine($"Saved adapter ({report.Adapter.ParameterCount:N0} parameters) to '{Path.GetFullPath(outPath)}'.");
    return 0;
}

async Task<int> RunEvalAsync(List<string> a)
{
    string model       = GetOption(a, "--model", "minilm");
    string dir         = GetOption(a, "--data", "./data");
    string adapterPath = GetOption(a, "--adapter", null);
    string split       = GetOption(a, "--split", "test");
    string datasetName = GetOption(a, "--dataset", "stsb").ToLowerInvariant();

    SentencePairDataset data;
    if (datasetName == "patent")
    {
        data = PatentDataset.Load(split);
    }
    else
    {
        string csv = split.ToLowerInvariant() switch
        {
            "train" => StsbDataset.TrainPath(dir),
            "dev"   => StsbDataset.DevPath(dir),
            _       => StsbDataset.TestPath(dir),
        };

        if (!File.Exists(csv))
        {
            Console.Error.WriteLine($"Split '{split}' not found at '{csv}'. Run `download` first (or use --dataset patent).");
            return 1;
        }
        data = StsbDataset.Load(csv);
    }

    Console.WriteLine($"Evaluating '{model}' on {datasetName} {split} split ({data.Count} pairs).\n");

    using var baseEncoder = await EncoderFactory.CreateAsync(model);

    await ReportAsync("base", baseEncoder, data);

    if (adapterPath is not null)
    {
        var adapter = LoraAdapter.Load(adapterPath);
        using var tuned = new AdaptedSentenceEncoder(baseEncoder, adapter, disposeBaseEncoder: false);
        await ReportAsync("tuned", tuned, data);
    }

    return 0;
}

async Task ReportAsync(string label, ISentenceEncoder encoder, SentencePairDataset data)
{
    float spearman  = await EmbeddingEvaluation.SpearmanAsync(encoder, data);
    var   retrieval = await EmbeddingEvaluation.RetrievalAsync(encoder, data, minScore: 0.6f);
    Console.WriteLine($"  [{label,-5}] STS spearman {Fmt(spearman)}   retrieval acc {Fmt(retrieval.Accuracy)}   MRR {Fmt(retrieval.MRR)}  (n={retrieval.Count})");
}

static string Fmt(float v) => float.IsNaN(v) ? "   n/a" : v.ToString("0.0000", CultureInfo.InvariantCulture);

static string GetOption(List<string> a, string name, string def)
{
    int i = a.IndexOf(name);
    if (i >= 0 && i + 1 < a.Count) return a[i + 1];
    return def;
}

void PrintHelp()
{
    Console.WriteLine(
$"""
SentenceTransformers.LoraTraining — LoRA-style adapter fine-tuning for sentence encoders.

USAGE
  dotnet run -- download [--data <dir>]
  dotnet run -- train    [--model <name>] [--data <dir>] [--out <path>] [training options]
  dotnet run -- eval      --model <name> [--adapter <path>] [--data <dir>] [--split train|dev|test]

MODELS (--model)
  {string.Join(", ", EncoderFactory.Names)}

DATASETS (--dataset)
  stsb    English STS Benchmark, downloaded on demand (run `download` first). Default.
  patent  Google Patent Phrase Similarity, embedded in this app — no download needed.

DOWNLOAD
  Downloads the English STS Benchmark (train/dev/test CSVs) into <dir> (default ./data).
  Not needed for --dataset patent.

TRAIN OPTIONS
  --model <name>          Base model to adapt (default minilm).
  --dataset <name>        stsb (default) or patent.
  --data <dir>            Directory holding stsb-en-train.csv (default ./data; stsb only).
  --out <path>            Where to save the trained adapter (default ./adapters/<model>-<dataset>.lora).
  --rank <int>            LoRA rank / bottleneck size (default 16).
  --alpha <float>         LoRA alpha; residual scale is alpha/rank (default = rank).
  --epochs <int>          Training epochs (default 20).
  --batch <int>           Pairs per contrastive batch (default 32).
  --lr <float>            AdamW learning rate (default 0.001).
  --weight-decay <float>  AdamW weight decay (default 0.0001).
  --temp <float>          InfoNCE softmax temperature (default 0.05).
  --val-frac <float>      Fraction of training pairs held out for validation (default 0.1).
  --pos-threshold <float> Min score in [0,1] for a pair to count as a positive (default 0.6).
  --max-train <int>       Cap the number of training pairs (0 = all; useful for quick runs).
  --seed <int>            RNG seed (default 42).

EVAL OPTIONS
  --model <name>          Base model (default minilm).
  --dataset <name>        stsb (default) or patent.
  --adapter <path>        Adapter to evaluate alongside the base encoder (optional).
  --data <dir>            Directory holding the CSVs (default ./data; stsb only).
  --split <name>          Which split to evaluate: train | dev | test (default test).

EXAMPLES
  dotnet run -c Release -- download
  dotnet run -c Release -- train --model minilm --epochs 30 --rank 32
  dotnet run -c Release -- eval  --model minilm --adapter ./adapters/minilm-stsb.lora --split test

  dotnet run -c Release -- train --model minilm --dataset patent --epochs 30 --rank 32
  dotnet run -c Release -- eval  --model minilm --dataset patent --adapter ./adapters/minilm-patent.lora
""");
}
