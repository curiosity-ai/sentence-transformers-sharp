using System.Globalization;
using SentenceTransformers;
using SentenceTransformers.Bert.Pure;
using SentenceTransformers.Bert.Pure.Model;
using SentenceTransformers.Bert.Pure.Training;
using SentenceTransformers.Harrier.Small.Pure.Training;
using SentenceTransformers.LoraTraining;
using SentenceTransformers.Training;

// ---------------------------------------------------------------------------------------------------
// SentenceTransformers.LoraTraining — fine-tune a pure-C# BERT encoder (MiniLM / Arctic-XS) with real
// weight-space LoRA adapters injected into the transformer's linear projections.
//
//   dotnet run -- download [--data ./data]
//   dotnet run -- train    [--model minilm] [--dataset stsb] [--out ./adapters/minilm.lora] [options]
//   dotnet run -- eval      --model minilm  --adapter ./adapters/minilm.lora [--split test]
//
// Weights are read from the fp32 ONNX graphs already embedded in the MiniLM / Arctic packages, so no
// model download is needed (only the STS-B dataset is fetched on demand). Run `help` for all options.
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

bool IsGemma(string model) => model is "harrier-small" or "harrier" or "harrier-small-pure";

async Task<int> RunTrainAsync(List<string> a)
{
    string model       = GetOption(a, "--model", "minilm");
    if (IsGemma(model)) return await RunTrainGemmaAsync(a, model);

    string dir         = GetOption(a, "--data", "./data");
    string datasetName = GetOption(a, "--dataset", "stsb").ToLowerInvariant();
    string outPath     = GetOption(a, "--out", $"./adapters/{model}-{datasetName}.lora");
    int    maxTrain    = int.Parse(GetOption(a, "--max-train", "0"), CultureInfo.InvariantCulture);

    var options = new BertLoraTrainingOptions
    {
        Rank                    = int.Parse(GetOption(a, "--rank", "8"), CultureInfo.InvariantCulture),
        Epochs                  = int.Parse(GetOption(a, "--epochs", "10"), CultureInfo.InvariantCulture),
        BatchSize               = int.Parse(GetOption(a, "--batch", "16"), CultureInfo.InvariantCulture),
        LearningRate            = float.Parse(GetOption(a, "--lr", "0.0005"), CultureInfo.InvariantCulture),
        Temperature             = float.Parse(GetOption(a, "--temp", "0.05"), CultureInfo.InvariantCulture),
        WeightDecay             = float.Parse(GetOption(a, "--weight-decay", "0.0001"), CultureInfo.InvariantCulture),
        ValidationFraction      = float.Parse(GetOption(a, "--val-frac", "0.1"), CultureInfo.InvariantCulture),
        PositiveScoreThreshold  = float.Parse(GetOption(a, "--pos-threshold", "0.6"), CultureInfo.InvariantCulture),
        MaxTokens               = int.Parse(GetOption(a, "--max-tokens", "128"), CultureInfo.InvariantCulture),
        WarmupFraction          = float.Parse(GetOption(a, "--warmup", "0.1"), CultureInfo.InvariantCulture),
        MinedNegativesPerAnchor = int.Parse(GetOption(a, "--mined-negatives", "0"), CultureInfo.InvariantCulture),
        NumSeeds                = int.Parse(GetOption(a, "--seeds", "1"), CultureInfo.InvariantCulture),
        Seed                    = int.Parse(GetOption(a, "--seed", "42"), CultureInfo.InvariantCulture),
        Patience                = int.Parse(GetOption(a, "--patience", "0"), CultureInfo.InvariantCulture),
        LearnableTemperature    = GetFlag(a, "--learnable-temp"),
        UseOutputBias           = GetFlag(a, "--output-bias"),
        ApplyWhitening          = GetFlag(a, "--whitening"),
        QueryPrefix             = GetOption(a, "--query-prefix", null),
        DocumentPrefix          = GetOption(a, "--doc-prefix", null),
    };

    string alpha = GetOption(a, "--alpha", null);
    if (alpha is not null) options.Alpha = float.Parse(alpha, CultureInfo.InvariantCulture);

    options.Objective = GetOption(a, "--objective", "contrastive").ToLowerInvariant() switch
    {
        "regression" or "cosine" or "cosine-regression" => BertTrainingObjective.CosineRegression,
        "cosent"                                         => BertTrainingObjective.CoSent,
        "contrastive"                                    => BertTrainingObjective.Contrastive,
        var other => throw new ArgumentException($"Unknown --objective '{other}'. Use 'contrastive', 'cosent' or 'regression'."),
    };

    options.Targets = GetOption(a, "--targets", "attention").ToLowerInvariant() switch
    {
        "attention" => LoraTargets.Attention,
        "mlp"       => LoraTargets.Mlp,
        "all"       => LoraTargets.All,
        var other   => throw new ArgumentException($"Unknown --targets '{other}'. Use 'attention', 'mlp' or 'all'."),
    };

    string matryoshka = GetOption(a, "--matryoshka", null);
    if (matryoshka is not null)
        options.MatryoshkaDims = matryoshka.Split(',').Select(x => int.Parse(x, CultureInfo.InvariantCulture)).ToArray();

    var dataset = LoadDataset(datasetName, dir, "train");
    if (dataset is null) return 1;
    if (maxTrain > 0 && dataset.Count > maxTrain) dataset = new SentencePairDataset(dataset.Pairs.Take(maxTrain));

    Console.WriteLine($"Model:            {model} (pure C#, real weight-space LoRA)");
    Console.WriteLine($"Dataset:          {datasetName}");
    Console.WriteLine($"Objective:        {options.Objective}");
    Console.WriteLine($"Adapter:          rank {options.Rank}, alpha {options.Alpha?.ToString(CultureInfo.InvariantCulture) ?? options.Rank.ToString()}, targets {options.Targets}");
    Console.WriteLine($"Optimizer:        AdamW lr {options.LearningRate} (warmup {options.WarmupFraction:0.00}, cosine decay), wd {options.WeightDecay}, {options.Epochs} epochs, batch {options.BatchSize}, temp {options.Temperature}");
    Console.WriteLine($"Extras:           mined-neg {options.MinedNegativesPerAnchor}, learnable-temp {options.LearnableTemperature}, output-bias {options.UseOutputBias}, whitening {options.ApplyWhitening}, seeds {options.NumSeeds}");
    Console.WriteLine($"Training pairs:   {dataset.Count}\n");
    Console.WriteLine("Loading base encoder (weights extracted from embedded ONNX) ...");

    using var encoder = EncoderFactory.Create(model);

    Console.WriteLine("Training ...\n");
    Console.WriteLine($"{"seed",4} {"epoch",5} {"train_loss",12} {"val_acc",9} {"val_spearman",13}");
    Console.WriteLine(new string('-', 48));
    options.OnEpoch = m => Console.WriteLine($"{m.Seed,4} {m.Epoch,5} {Fmt(m.TrainLoss),12} {Fmt(m.ValidationAccuracy),9} {Fmt(m.ValidationSpearman),13}{(m.IsBest ? "  *best" : "")}");

    var report = await BertLoraTrainer.TrainAsync(encoder, dataset, options);

    Console.WriteLine();
    Console.WriteLine("Validation summary (base -> tuned):");
    Console.WriteLine($"  retrieval accuracy: {Fmt(report.BaselineAccuracy)} -> {Fmt(report.BestAccuracy)}");
    if (!float.IsNaN(report.BaselineSpearman))
        Console.WriteLine($"  STS spearman:       {Fmt(report.BaselineSpearman)} -> {Fmt(report.BestSpearman)}");

    report.Adapter.Save(outPath);
    Console.WriteLine($"\nSaved adapter ({report.Adapter.ParameterCount:N0} parameters) to '{Path.GetFullPath(outPath)}'.");
    return 0;
}

async Task<int> RunEvalAsync(List<string> a)
{
    string model       = GetOption(a, "--model", "minilm");
    if (IsGemma(model)) return await RunEvalGemmaAsync(a, model);

    string dir         = GetOption(a, "--data", "./data");
    string adapterPath = GetOption(a, "--adapter", null);
    string split       = GetOption(a, "--split", "test");
    string datasetName = GetOption(a, "--dataset", "stsb").ToLowerInvariant();

    var data = LoadDataset(datasetName, dir, split);
    if (data is null) return 1;

    Console.WriteLine($"Evaluating '{model}' on {datasetName} {split} split ({data.Count} pairs).\n");

    using var baseEncoder = EncoderFactory.Create(model);
    await ReportAsync("base", baseEncoder, data);

    if (adapterPath is not null)
    {
        var adapter = LoraAdapter.Load(adapterPath);
        using var tuned = baseEncoder.WithAdapter(adapter);
        await ReportAsync("tuned", tuned, data);
    }
    return 0;
}

async Task<int> RunTrainGemmaAsync(List<string> a, string model)
{
    string dir         = GetOption(a, "--data", "./data");
    string datasetName = GetOption(a, "--dataset", "stsb").ToLowerInvariant();
    string outPath     = GetOption(a, "--out", $"./adapters/{model}-{datasetName}.lora");
    int    maxTrain    = int.Parse(GetOption(a, "--max-train", "0"), CultureInfo.InvariantCulture);

    var options = new GemmaLoraTrainingOptions
    {
        Rank                    = int.Parse(GetOption(a, "--rank", "8"), CultureInfo.InvariantCulture),
        Epochs                  = int.Parse(GetOption(a, "--epochs", "5"), CultureInfo.InvariantCulture),
        BatchSize               = int.Parse(GetOption(a, "--batch", "8"), CultureInfo.InvariantCulture),
        LearningRate            = float.Parse(GetOption(a, "--lr", "0.0005"), CultureInfo.InvariantCulture),
        Temperature             = float.Parse(GetOption(a, "--temp", "0.05"), CultureInfo.InvariantCulture),
        WeightDecay             = float.Parse(GetOption(a, "--weight-decay", "0.0001"), CultureInfo.InvariantCulture),
        ValidationFraction      = float.Parse(GetOption(a, "--val-frac", "0.1"), CultureInfo.InvariantCulture),
        PositiveScoreThreshold  = float.Parse(GetOption(a, "--pos-threshold", "0.6"), CultureInfo.InvariantCulture),
        MaxTokens               = int.Parse(GetOption(a, "--max-tokens", "64"), CultureInfo.InvariantCulture),
        WarmupFraction          = float.Parse(GetOption(a, "--warmup", "0.1"), CultureInfo.InvariantCulture),
        MinedNegativesPerAnchor = int.Parse(GetOption(a, "--mined-negatives", "0"), CultureInfo.InvariantCulture),
        NumSeeds                = int.Parse(GetOption(a, "--seeds", "1"), CultureInfo.InvariantCulture),
        Seed                    = int.Parse(GetOption(a, "--seed", "42"), CultureInfo.InvariantCulture),
        Patience                = int.Parse(GetOption(a, "--patience", "0"), CultureInfo.InvariantCulture),
        LearnableTemperature    = GetFlag(a, "--learnable-temp"),
        UseOutputBias           = GetFlag(a, "--output-bias"),
        ApplyWhitening          = GetFlag(a, "--whitening"),
        QueryPrefix             = GetOption(a, "--query-prefix", null),
        DocumentPrefix          = GetOption(a, "--doc-prefix", null),
    };
    string alpha = GetOption(a, "--alpha", null);
    if (alpha is not null) options.Alpha = float.Parse(alpha, CultureInfo.InvariantCulture);

    options.Objective = GetOption(a, "--objective", "contrastive").ToLowerInvariant() switch
    {
        "regression" or "cosine" or "cosine-regression" => GemmaTrainingObjective.CosineRegression,
        "cosent"                                         => GemmaTrainingObjective.CoSent,
        "contrastive"                                    => GemmaTrainingObjective.Contrastive,
        var other => throw new ArgumentException($"Unknown --objective '{other}'."),
    };
    options.Targets = GetOption(a, "--targets", "attention").ToLowerInvariant() switch
    {
        "attention" => GemmaLoraTargets.Attention,
        "mlp"       => GemmaLoraTargets.Mlp,
        "all"       => GemmaLoraTargets.All,
        var other   => throw new ArgumentException($"Unknown --targets '{other}'."),
    };
    string matryoshka = GetOption(a, "--matryoshka", null);
    if (matryoshka is not null) options.MatryoshkaDims = matryoshka.Split(',').Select(x => int.Parse(x, CultureInfo.InvariantCulture)).ToArray();

    var dataset = LoadDataset(datasetName, dir, "train");
    if (dataset is null) return 1;
    if (maxTrain > 0 && dataset.Count > maxTrain) dataset = new SentencePairDataset(dataset.Pairs.Take(maxTrain));

    Console.WriteLine($"Model:            harrier-small (pure C# Gemma3, real weight-space LoRA)");
    Console.WriteLine($"Dataset:          {datasetName}   Objective: {options.Objective}   Targets: {options.Targets}");
    Console.WriteLine($"Adapter:          rank {options.Rank}, alpha {options.Alpha?.ToString(CultureInfo.InvariantCulture) ?? options.Rank.ToString()}");
    Console.WriteLine($"Training pairs:   {dataset.Count}\n");
    Console.WriteLine("Downloading/loading Harrier Small weights (bf16 safetensors) ...");

    using var encoder = await Gemma3LoraEncoder.CreateAsync(reportProgress: p => Console.Error.Write($"\r  {p.Fraction*100,5:0.0}%   "));
    Console.Error.WriteLine();
    Console.WriteLine($"\n{"seed",4} {"epoch",5} {"train_loss",12} {"val_acc",9} {"val_spearman",13}");
    Console.WriteLine(new string('-', 48));
    options.OnEpoch = m => Console.WriteLine($"{m.Seed,4} {m.Epoch,5} {Fmt(m.TrainLoss),12} {Fmt(m.ValidationAccuracy),9} {Fmt(m.ValidationSpearman),13}{(m.IsBest ? "  *best" : "")}");

    var report = await Gemma3LoraTrainer.TrainAsync(encoder, dataset, options);

    Console.WriteLine("\nValidation summary (base -> tuned):");
    Console.WriteLine($"  retrieval accuracy: {Fmt(report.BaselineAccuracy)} -> {Fmt(report.BestAccuracy)}");
    if (!float.IsNaN(report.BaselineSpearman))
        Console.WriteLine($"  STS spearman:       {Fmt(report.BaselineSpearman)} -> {Fmt(report.BestSpearman)}");
    report.Adapter.Save(outPath);
    Console.WriteLine($"\nSaved adapter ({report.Adapter.ParameterCount:N0} parameters) to '{Path.GetFullPath(outPath)}'.");
    return 0;
}

async Task<int> RunEvalGemmaAsync(List<string> a, string model)
{
    string dir         = GetOption(a, "--data", "./data");
    string adapterPath = GetOption(a, "--adapter", null);
    string split       = GetOption(a, "--split", "test");
    string datasetName = GetOption(a, "--dataset", "stsb").ToLowerInvariant();

    var data = LoadDataset(datasetName, dir, split);
    if (data is null) return 1;

    Console.WriteLine($"Evaluating 'harrier-small' on {datasetName} {split} split ({data.Count} pairs).\n");
    Console.WriteLine("Downloading/loading Harrier Small weights ...");
    using var baseEncoder = await Gemma3LoraEncoder.CreateAsync(reportProgress: p => Console.Error.Write($"\r  {p.Fraction*100,5:0.0}%   "));
    Console.Error.WriteLine();

    await ReportAsync("base", baseEncoder, data);
    if (adapterPath is not null)
    {
        var adapter = GemmaLoraAdapter.Load(adapterPath);
        using var tuned = baseEncoder.WithAdapter(adapter);
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

SentencePairDataset LoadDataset(string datasetName, string dir, string split)
{
    if (datasetName == "patent") return PatentDataset.Load(split);
    string csv = split.ToLowerInvariant() switch
    {
        "train" => StsbDataset.TrainPath(dir),
        "dev"   => StsbDataset.DevPath(dir),
        _       => StsbDataset.TestPath(dir),
    };
    if (!File.Exists(csv))
    {
        Console.Error.WriteLine($"STS-B split '{split}' not found at '{csv}'. Run `download` first (or use --dataset patent).");
        return null;
    }
    return StsbDataset.Load(csv);
}

static string Fmt(float v) => float.IsNaN(v) ? "   n/a" : v.ToString("0.0000", CultureInfo.InvariantCulture);

static string GetOption(List<string> a, string name, string def)
{
    int i = a.IndexOf(name);
    return i >= 0 && i + 1 < a.Count ? a[i + 1] : def;
}

static bool GetFlag(List<string> a, string name) => a.Contains(name);

void PrintHelp()
{
    Console.WriteLine(
$"""
SentenceTransformers.LoraTraining — real weight-space LoRA fine-tuning for pure-C# BERT encoders.

USAGE
  dotnet run -- download [--data <dir>]
  dotnet run -- train    [--model <name>] [--dataset <name>] [--out <path>] [training options]
  dotnet run -- eval      --model <name> [--adapter <path>] [--dataset <name>] [--split train|dev|test]

MODELS (--model)     {string.Join(", ", EncoderFactory.Names)} (weights from embedded fp32 ONNX, no download),
                     harrier-small (pure C# Gemma3; bf16 weights downloaded on first use)
DATASETS (--dataset) stsb (download first), patent (embedded)

TRAIN OPTIONS
  --objective <name>      contrastive (default), cosent, or regression.
                          cosent/regression target graded STS Spearman; contrastive targets retrieval.
  --targets <name>        Which linears get LoRA: attention (default), mlp, or all.
  --rank <int>            LoRA rank (default 8).
  --alpha <float>         LoRA alpha; residual scale is alpha/rank (default = rank).
  --epochs <int>          Training epochs (default 10).
  --batch <int>           Pairs per batch (default 16).
  --lr <float>            AdamW learning rate (default 0.0005).
  --warmup <float>        Warmup fraction of total steps (default 0.1); LR then cosine-decays.
  --weight-decay <float>  AdamW weight decay (default 0.0001).
  --temp <float>          InfoNCE/CoSENT temperature (default 0.05).
  --learnable-temp        Learn the temperature jointly (flag).
  --mined-negatives <int> Hard negatives mined per anchor each epoch (default 0; contrastive only).
  --output-bias           Learn an output-centering bias (de-anisotropy) (flag).
  --whitening             Fit a post-hoc ZCA whitening transform and fold it into the adapter (flag).
  --matryoshka <d1,d2>    Also train truncated sub-dimensions (comma-separated).
  --query-prefix <str>    Instruction prefix prepended to anchors (queries).
  --doc-prefix <str>      Prefix prepended to positives/negatives (documents).
  --seeds <int>           Train N seeds and keep the best by validation metric (default 1).
  --patience <int>        Early-stop a seed after N epochs with no validation improvement (0 = off).
  --max-tokens <int>      Truncate training sequences (default 128).
  --val-frac <float>      Validation fraction (default 0.1).
  --pos-threshold <float> Min score for a pair to count as positive for contrastive (default 0.6).
  --max-train <int>       Cap training pairs (0 = all).
  --seed <int>            Base RNG seed (default 42).

EXAMPLES
  dotnet run -c Release -- download
  dotnet run -c Release -- train --model minilm --objective cosent --rank 8 --epochs 10 --output-bias
  dotnet run -c Release -- eval  --model minilm --adapter ./adapters/minilm-stsb.lora --split test

  dotnet run -c Release -- train --model arctic --dataset patent --objective cosent --rank 8 --whitening
  dotnet run -c Release -- eval  --model arctic --dataset patent --adapter ./adapters/arctic-patent.lora
""");
}
