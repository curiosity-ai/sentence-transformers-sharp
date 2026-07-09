# Fine-tuning with real weight-space LoRA

This library can fine-tune its **pure-C# encoders** to your domain by training a small **LoRA adapter**
— entirely in managed code, with no ONNX Runtime, no PyTorch, and no Python. This document explains what
that means, how it works internally, and how to train and use an adapter.

> TL;DR — pick a trainable model (`minilm`, `arctic`, or `harrier-small`), give the trainer a set of
> related sentence pairs, choose an objective (`CoSent` for graded similarity, `Contrastive` for
> retrieval), and it produces a small `.lora` file you apply with `encoder.WithAdapter(...)`.

---

## What "real weight-space LoRA" means here

LoRA (Low-Rank Adaptation) freezes the base model and learns, for selected weight matrices `W`, a small
low-rank update `ΔW = (α / r)·B·A` where `A ∈ ℝ^{r×in}`, `B ∈ ℝ^{out×r}`, and `r` (the *rank*) is small
(e.g. 8). Only `A` and `B` train; the effective weight at inference is `W + ΔW`.

This is **weight-space** LoRA: the low-rank factors are injected **inside** the transformer's linear
projections (attention Q/K/V/O and the MLP), and the training loss is backpropagated through the *whole*
(frozen) network. That is the real thing — not an output-space transform bolted onto the pooled embedding.

To do that in pure C# we run the transformer's **forward and backward** passes ourselves via a small
tensor autograd engine. Following the LoRA convention, `B` starts at zero and `A` at small Gaussian noise,
so a freshly-initialized adapter reproduces the base model exactly and only departs from it as it trains.

### Which models are trainable

| Model | Package | Architecture | Weights for training |
| --- | --- | --- | --- |
| `minilm` (all-MiniLM-L6-v2) | `SentenceTransformers.Bert.Pure` | BERT encoder | fp32, extracted from the embedded ONNX (no download) |
| `arctic` (snowflake-arctic-embed-xs) | `SentenceTransformers.Bert.Pure` | BERT encoder | fp32, extracted from the embedded ONNX (no download) |
| `harrier-small` (harrier-oss-v1-270m) | `SentenceTransformers.Harrier.Small.Pure` | Gemma3 decoder | bf16 safetensors, downloaded on first use |

The ONNX-only models (Qwen3, Harrier Medium) are inference-only and cannot be trained here.

> **Harrier Small is a ~270M-param decoder**, so every forward+backward is far heavier than the small
> BERTs. Keep the batch size and sequence length modest, and expect training to be slow relative to MiniLM.

---

## How it works internally

```
text ──tokenize──► ids ──►  frozen transformer forward (LoRA-injected linears)  ──► pooled vector z
                                              │                                        │
                                     autograd tape                              L2-normalize → u
                                              │                                        │
                        params (A,B[,bias]) ◄─ backward ◄─ dL/dz ◄─ dL/du ◄──── loss (per batch)
```

1. **Shared autograd engine** — `SentenceTransformers.Training.Autograd` (`Tensor` + `Graph`). The forward
   pass is written once as a sequence of graph ops (matmul, LayerNorm/RMSNorm, GELU, softmax, RoPE,
   attention, pooling, …). Calling `Graph.Backward()` replays the tape to produce exact gradients. Because
   the base weights are plain frozen arrays (not graph nodes), no gradient is computed for them — only the
   tiny LoRA tensors carry grad. That is the LoRA efficiency win.
2. **LoRA injection** — each targeted linear becomes `y = x·Wᵀ + b + (α/r)·(x·Aᵀ)·Bᵀ`. Targets are chosen
   with `Targets` (`Attention`, `Mlp`, or `All`). BERT exposes q/k/v/o + intermediate/output; Gemma3
   exposes q/k/v/o + gate/up/down.
3. **Loss** — computed on the L2-normalized pooled vectors for a batch (see objectives below), then its
   gradient w.r.t. each vector is seeded and back-propagated into the adapter parameters.
4. **Optimizer** — AdamW with linear warmup then cosine decay to zero. Optionally a learned temperature.
5. **Model selection** — after each epoch the adapter is scored on a held-out validation split, and the
   best epoch's adapter is kept. With `NumSeeds > 1` several seeds are trained and the best overall is
   returned (results are seed-sensitive; the frozen base makes extra seeds relatively cheap).

The forward pass is verified to match the reference exactly (bit-for-bit vs. the ONNX runtime for BERT, and
vs. the optimized inference model for Gemma3), and the backward pass is verified by a numerical gradient
check — both live in `SentenceTransformers.Tests`.

### Objectives

Set `Objective` on the options:

- **`Contrastive`** — symmetric InfoNCE / MultipleNegativesRanking. Pulls each anchor toward its positive
  and away from negatives. Best for **retrieval / nearest-neighbour** separation. Uses only pairs at or
  above `PositiveScoreThreshold` (all pairs when unscored).
- **`CoSent`** — the CoSENT pairwise-ranking loss over graded pairs. It optimizes the *ordering* that STS
  Spearman measures and generally beats plain cosine-MSE. Best when you care about **graded similarity**.
- **`CosineRegression`** — mean-squared error between adapted cosine and the gold `[0,1]` score.

### Negative samples (contrastive only)

For the `Contrastive` objective, negatives come from three sources plus a safety mask:

- **In-batch negatives** — every *other* pair's positive (and anchor, in the reverse direction) in the
  batch is a negative. Always on; a larger `BatchSize` gives more of them.
- **Explicit hard negatives** — set `SentencePair.Negative`; used when `UseExplicitNegatives` (default on).
- **Mined hard negatives** — set `MinedNegativesPerAnchor > 0` to mine each anchor's nearest *foreign*
  positives every `MineEveryEpochs`, skipping any candidate with cosine above `MinedNegativeMaxCosine`
  (default 0.95) to avoid false negatives.
- **False-negative masking** — `MaskFalseNegatives` (default on) drops in-batch candidates whose text
  duplicates the row's own anchor/positive.

`CoSent` and `CosineRegression` don't use negatives (they rank/fit scored pairs directly).

### Stop conditions

Training is a fixed schedule with best-checkpointing and optional early stopping:

- **Epoch budget** — each seed runs up to `Epochs`.
- **Early stopping** — set `Patience > 0` to stop a seed after that many epochs with no improvement in the
  validation metric (Spearman when pairs are scored, else retrieval accuracy). `0` disables it. The best
  adapter seen is always the one returned, so early stopping only saves time; it never changes which
  adapter you get for a plateau.
- **Best checkpoint** — the returned adapter is the best epoch across all seeds, not the last.
- **Cancellation** — pass a `CancellationToken`; it is observed each epoch and batch.

### Progress reporting

Two independent hooks:

- **`OnEpoch`** — an `Action<…EpochMetrics>` callback invoked once per epoch with `(seed, epoch, trainLoss,
  validationAccuracy, validationSpearman, isBest)`.
- **`Logger`** — an optional `Microsoft.Extensions.Logging.ILogger`. When set, the trainer logs the
  baseline metrics, per-epoch progress, hard-negative mining (debug), early stopping, whitening, and a
  final base→tuned summary.

### Optional quality extras

- **Warmup + cosine LR** (`WarmupFraction`, `CosineDecay`).
- **Learned temperature** (`LearnableTemperature`).
- **Output-centering bias** (`UseOutputBias`) — a learned per-dimension bias added before normalization,
  which counters embedding anisotropy.
- **Post-hoc ZCA whitening** (`ApplyWhitening`) — fits a whitening transform from the tuned training
  embeddings and folds it into the adapter; another cheap anisotropy fix for cosine STS.
- **Matryoshka** (`MatryoshkaDims`) — also train truncated leading-prefix sub-vectors so shortened
  embeddings stay good.
- **Asymmetric prefixes** (`QueryPrefix`, `DocumentPrefix`) — instruction-style prefixes applied to
  anchors vs. documents (some models expect these).

---

## Train one — library API

### MiniLM / Arctic (BERT)

```csharp
using SentenceTransformers.Bert.Pure;
using SentenceTransformers.Bert.Pure.Model;
using SentenceTransformers.Bert.Pure.Training;
using SentenceTransformers.Training;

// Weights are read from the embedded ONNX — no download.
using var encoder = await SentenceEncoder.CreateMiniLMAsync();   // or CreateArcticXsAsync()

var data = new SentencePairDataset(new[]
{
    new SentencePair("how do I reset my password", "Use the ‘Forgot password’ link on the sign-in page.", 0.95f),
    new SentencePair("cancel my subscription",      "Go to Billing → Manage plan → Cancel.",              0.95f),
    new SentencePair("track my order",              "the museum opens at 9am",                             0.05f),
    // optional explicit hard negative:
    new SentencePair("reset password",              "Use the ‘Forgot password’ link.", 0.95f, Negative: "delete my account permanently"),
    // … a few hundred to a few thousand related pairs …
});

var report = await BertLoraTrainer.TrainAsync(encoder, data, new BertLoraTrainingOptions
{
    Objective = BertTrainingObjective.CoSent,   // graded similarity → CoSENT
    Rank      = 8,
    Targets   = LoraTargets.Attention,
    Epochs    = 20,
    Patience  = 3,                              // early stop after 3 epochs w/o improvement
    UseOutputBias = true,
    // Logger = myLoggerFactory.CreateLogger("lora"),
    OnEpoch = m => Console.WriteLine($"seed {m.Seed} epoch {m.Epoch}: spearman {m.ValidationSpearman:0.000}"),
});

Console.WriteLine($"spearman {report.BaselineSpearman:0.000} -> {report.BestSpearman:0.000}");
report.Adapter.Save("support-faq.lora");

// Use it — a drop-in ISentenceEncoder with the adapter folded into the transformer:
using var tuned = encoder.WithAdapter(report.Adapter);
float[] v = await tuned.EncodeAsync("I can't log in");
```

### Harrier Small (Gemma3)

```csharp
using SentenceTransformers.Harrier.Small.Pure.Training;
using SentenceTransformers.Training;

using var encoder = await Gemma3LoraEncoder.CreateAsync();   // downloads bf16 weights on first use

var report = await Gemma3LoraTrainer.TrainAsync(encoder, data, new GemmaLoraTrainingOptions
{
    Objective = GemmaTrainingObjective.CoSent,
    Rank      = 8,
    Targets   = GemmaLoraTargets.Attention,
    Epochs    = 8,
    Patience  = 2,
    BatchSize = 8,        // keep small — 270M-param decoder
    MaxTokens = 64,
});

report.Adapter.Save("harrier.lora");
using var tuned = encoder.WithAdapter(report.Adapter);
```

### Applying a saved adapter later

```csharp
// BERT
using var enc = await SentenceEncoder.CreateMiniLMAsync(adapter: LoraAdapter.Load("support-faq.lora"));

// Gemma
using var genc = await Gemma3LoraEncoder.CreateAsync(adapter: GemmaLoraAdapter.Load("harrier.lora"));
```

`SentencePairDataset` is just a list of `SentencePair(Anchor, Positive, Score?, Negative?)`. `Score` is an
optional graded similarity in `[0,1]` (required for `CoSent` / `CosineRegression`, and used to filter
positives for `Contrastive`); `Negative` is an optional explicit hard negative.

---

## Train one — command line

The `SentenceTransformers.LoraTraining` console app wraps all of the above and ships two example datasets:
`stsb` (English STS Benchmark, downloaded on demand) and `patent` (Google Patent Phrase Similarity,
embedded, no download).

```bash
cd SentenceTransformers.LoraTraining

# One-time dataset download (model weights are embedded / auto-downloaded):
dotnet run -c Release -- download

# MiniLM on STS-B with CoSENT, early stopping and an output-centering bias:
dotnet run -c Release -- train --model minilm --objective cosent --rank 8 --epochs 20 --patience 3 --output-bias
dotnet run -c Release -- eval  --model minilm --adapter ./adapters/minilm-stsb.lora --split test

# Arctic on the domain-specific patent set, with whitening:
dotnet run -c Release -- train --model arctic --dataset patent --objective cosent --rank 8 --whitening
dotnet run -c Release -- eval  --model arctic --dataset patent --adapter ./adapters/arctic-patent.lora

# Harrier Small (downloads bf16 weights on first use; heavier, so smaller batch):
dotnet run -c Release -- train --model harrier-small --dataset patent --objective cosent --rank 8 --batch 8 --max-tokens 64 --patience 2
```

Run `dotnet run -- help` for the full flag list.

### Options reference

| Option (CLI `--flag` / API property) | Default | Meaning |
| --- | --- | --- |
| `--objective` / `Objective` | `contrastive` | `contrastive`, `cosent`, or `regression`. |
| `--targets` / `Targets` | `attention` | Which linears get adapters: `attention`, `mlp`, `all`. |
| `--rank` / `Rank` | 8 | LoRA rank (capacity). |
| `--alpha` / `Alpha` | = rank | LoRA scale numerator; residual scale is `α/rank`. |
| `--epochs` / `Epochs` | 10 | Max epochs per seed. |
| `--patience` / `Patience` | 0 (off) | Early-stop after N epochs without validation improvement. |
| `--batch` / `BatchSize` | 16 (Gemma 8) | Pairs per batch (more = more in-batch negatives). |
| `--lr` / `LearningRate` | 5e-4 | AdamW learning rate. |
| `--warmup` / `WarmupFraction` | 0.1 | Warmup fraction of total steps; LR then cosine-decays. |
| `--weight-decay` / `WeightDecay` | 1e-4 | AdamW weight decay. |
| `--temp` / `Temperature` | 0.05 | InfoNCE / CoSENT temperature. |
| `--learnable-temp` / `LearnableTemperature` | off | Learn the temperature jointly. |
| `--mined-negatives` / `MinedNegativesPerAnchor` | 0 | Hard negatives mined per anchor each epoch (contrastive). |
| `--output-bias` / `UseOutputBias` | off | Learn an output-centering bias. |
| `--whitening` / `ApplyWhitening` | off | Fit a post-hoc ZCA whitening transform into the adapter. |
| `--matryoshka` / `MatryoshkaDims` | none | Also train truncated sub-dimensions (comma-separated). |
| `--query-prefix` / `QueryPrefix` | none | Instruction prefix for anchors. |
| `--doc-prefix` / `DocumentPrefix` | none | Prefix for positives/negatives. |
| `--seeds` / `NumSeeds` | 1 | Train N seeds; keep the best by validation. |
| `--seed` / `Seed` | 42 | Base RNG seed. |
| `--max-tokens` / `MaxTokens` | 128 (Gemma 64) | Truncate training sequences. |
| `--val-frac` / `ValidationFraction` | 0.1 | Held-out validation fraction. |
| `--pos-threshold` / `PositiveScoreThreshold` | 0.6 | Min score to count as a positive (contrastive). |
| (API only) `Logger` | none | Optional `ILogger` for progress. |

---

## Practical tips

- **Match the objective to the metric.** If you care about graded similarity / STS Spearman, use `CoSent`.
  If you care about retrieving the right item, use `Contrastive` with a decent `BatchSize` (and hard
  negatives if you have them).
- **Start small on rank.** Rank 8 on the attention projections is a strong default; raise the rank or add
  `Targets = All` only if you have enough data to avoid overfitting.
- **Regularize for distribution shift.** Fewer epochs (or `Patience`), higher weight decay, and small rank
  help the gains carry to unseen data rather than just the training distribution.
- **Anisotropy fixes are cheap.** `UseOutputBias` and/or `ApplyWhitening` often help cosine-based STS.
- **Harrier is expensive.** Keep `BatchSize`/`MaxTokens` small, use `Patience`, and prefer MiniLM/Arctic
  unless you specifically need Harrier's multilingual quality.

---

## Where the code lives

- `SentenceTransformers/src/Training/Autograd.cs` — the shared `Tensor`/`Graph` autograd engine.
- `SentenceTransformers/src/Training/LoraLosses.cs` — InfoNCE, CoSENT, and regression losses.
- `SentenceTransformers/src/Training/Whitening.cs` — the ZCA whitening fitter.
- `SentenceTransformers.Bert.Pure/` — pure BERT model, `LoraAdapter`, `BertLoraTrainer`, encoder.
- `SentenceTransformers.Harrier.Small.Pure/src/Training/` — Gemma3 model, `GemmaLoraAdapter`,
  `Gemma3LoraTrainer`, `Gemma3LoraEncoder`.
- `SentenceTransformers.LoraTraining/` — the training CLI and example datasets.
- `SentenceTransformers.Tests/` — forward-parity, gradient-check, save/load, early-stop and
  training-improvement tests.
