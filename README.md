# sentence-transformers-sharp

Fast, dependency-light **sentence embeddings for .NET**. This library wraps a set of
[ONNX](https://onnx.ai/) embedding models behind a single, simple `ISentenceEncoder` interface so you
can turn text into vectors — for semantic search, clustering, retrieval-augmented generation (RAG),
deduplication, recommendations and similarity scoring — entirely in-process, with no Python runtime
and no external API calls.

It is built and maintained by [Curiosity](https://curiosity.ai) and powers the AI search / vector
indexing features of [Curiosity Workspace](https://curiosity.ai).

```csharp
using SentenceTransformers.MiniLM;

using var encoder = new SentenceEncoder();

float[][] vectors = await encoder.EncodeAsync(new[]
{
    "The quick brown fox jumps over the lazy dog",
    "A fast auburn fox leaps above a sleepy hound",
});

// vectors[0] and vectors[1] are L2-normalized float[384] embeddings.
```

## Why this library

- **No Python, no servers.** Inference runs locally via the [ONNX Runtime](https://onnxruntime.ai/);
  the embedded models ship inside the NuGet package.
- **One interface, many models.** Swap models by changing a single `using` — every encoder implements
  [`ISentenceEncoder`](SentenceTransformers/src/ISentenceEncoder.cs).
- **Tokenizer-aware chunking built in.** Long documents are split on token boundaries (never exceeding
  the model's context window) and encoded in one call, with optional overlap, progress reporting and
  offset alignment back into the source text.
- **Normalized vectors.** All models return L2-normalized embeddings, so cosine similarity is just a
  dot product.

## Models

| Package | Model | Dimensions | Max tokens | Languages | Weights |
| --- | --- | --- | --- | --- | --- |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.svg?label=SentenceTransformers)](https://www.nuget.org/packages/SentenceTransformers/) [![Downloads](https://img.shields.io/nuget/dt/SentenceTransformers.svg?label=)](https://www.nuget.org/packages/SentenceTransformers/) | Core interfaces & chunking helpers | — | — | — | — |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.MiniLM.svg?label=SentenceTransformers.MiniLM)](https://www.nuget.org/packages/SentenceTransformers.MiniLM/) [![Downloads](https://img.shields.io/nuget/dt/SentenceTransformers.MiniLM.svg?label=)](https://www.nuget.org/packages/SentenceTransformers.MiniLM/) | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | 256 | English | Embedded |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.ArcticXs.svg?label=SentenceTransformers.ArcticXs)](https://www.nuget.org/packages/SentenceTransformers.ArcticXs/) [![Downloads](https://img.shields.io/nuget/dt/SentenceTransformers.ArcticXs.svg?label=)](https://www.nuget.org/packages/SentenceTransformers.ArcticXs/) | [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs) | 384 | 512 | English | Embedded |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.Qwen3.svg?label=SentenceTransformers.Qwen3)](https://www.nuget.org/packages/SentenceTransformers.Qwen3/) [![Downloads](https://img.shields.io/nuget/dt/SentenceTransformers.Qwen3.svg?label=)](https://www.nuget.org/packages/SentenceTransformers.Qwen3/) | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 1024 | 32768 | Multilingual | Downloaded on first use |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.Harrier.Medium.svg?label=SentenceTransformers.Harrier.Medium)](https://www.nuget.org/packages/SentenceTransformers.Harrier.Medium/) [![Downloads](https://img.shields.io/nuget/dt/SentenceTransformers.Harrier.Medium.svg?label=)](https://www.nuget.org/packages/SentenceTransformers.Harrier.Medium/) | [harrier-oss-v1-0.6b](https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX) | 1024 | 32768 | Multilingual | Downloaded on first use |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.Harrier.Small.svg?label=SentenceTransformers.Harrier.Small)](https://www.nuget.org/packages/SentenceTransformers.Harrier.Small/) [![Downloads](https://img.shields.io/nuget/dt/SentenceTransformers.Harrier.Small.svg?label=)](https://www.nuget.org/packages/SentenceTransformers.Harrier.Small/) | [harrier-oss-v1-270m](https://huggingface.co/onnx-community/harrier-oss-v1-270m-ONNX) | 640 | 32768 | Multilingual | Downloaded on first use |

- **Embedded** models bundle the ONNX weights inside the NuGet package, so the encoder is ready
  immediately after construction.
- **Downloaded** models are larger; their weights are fetched once on first use and cached on disk
  (under the system temp folder by default — see [Choosing where weights are stored](#choosing-where-weights-are-stored)).

Pick **MiniLM** for the smallest/fastest footprint, **Arctic XS** for a strong English default,
**Harrier Small** when you need multilingual coverage without paying for the larger 0.6b weights,
and **Qwen3** or **Harrier Medium** when you want the highest-quality embeddings (1024 dim) and
the full 32k-token context window.

## Installation

Install only the model package(s) you need (the core `SentenceTransformers` package is pulled in as a
dependency):

```bash
dotnet add package SentenceTransformers.MiniLM
dotnet add package SentenceTransformers.ArcticXs
dotnet add package SentenceTransformers.Qwen3
dotnet add package SentenceTransformers.Harrier.Medium
dotnet add package SentenceTransformers.Harrier.Small
```

Targets **.NET 10**.

## Usage

### Embedded models (MiniLM, Arctic XS)

Embedded models are ready to use as soon as you construct them:

```csharp
using SentenceTransformers.ArcticXs;

using var encoder = new SentenceEncoder();

float[][] vectors = await encoder.EncodeAsync(new[]
{
    "How do I reset my password?",
    "I forgot my login credentials.",
});
```

### Downloaded models (Qwen3, Harrier Medium, Harrier Small)

Larger models download their ONNX weights on first use. Create them with the async `CreateAsync`
factory — the download is cached, so subsequent runs are instant:

```csharp
using SentenceTransformers.Qwen3;

// Downloads the model to a temp folder on first use, then loads it.
using var encoder = await SentenceEncoder.CreateAsync();

float[][] vectors = await encoder.EncodeAsync(new[] { "Hello world" });
// vectors[0] is a float[1024]
```

Harrier Medium is multilingual:

```csharp
using SentenceTransformers.Harrier.Medium;

using var encoder = await SentenceEncoder.CreateAsync();

float[][] vectors = await encoder.EncodeAsync(new[]
{
    "Good morning",   // English
    "Buenos días",    // Spanish
    "おはよう",          // Japanese
});
```

Harrier Small is the same multilingual family at ~270M parameters (640-dim embeddings),
suitable when you want multilingual coverage without paying for the 0.6b weights:

```csharp
using SentenceTransformers.Harrier.Small;

using var encoder = await SentenceEncoder.CreateAsync();

float[][] vectors = await encoder.EncodeAsync(new[]
{
    "Good morning",
    "Buenos días",
    "おはよう",
});
// vectors[0] is a float[640]
```

### Comparing two texts (cosine similarity)

Because every model returns L2-normalized vectors, cosine similarity is simply the dot product:

```csharp
static float CosineSimilarity(float[] a, float[] b)
{
    float dot = 0f;
    for (int i = 0; i < a.Length; i++)
    {
        dot += a[i] * b[i];
    }
    return dot; // vectors are unit-length, so dot product == cosine similarity
}

using var encoder = new SentenceTransformers.MiniLM.SentenceEncoder();
var v = await encoder.EncodeAsync(new[] { "cat", "kitten", "spaceship" });

Console.WriteLine(CosineSimilarity(v[0], v[1])); // cat vs kitten   -> high
Console.WriteLine(CosineSimilarity(v[0], v[2])); // cat vs spaceship -> low
```

### Embedding long documents (chunking)

`EncodeAsync` expects each input to fit within the model's context window
(`encoder.MaxChunkLength` tokens). For longer text, use the built-in chunking helpers, which split on
token boundaries and embed each chunk:

```csharp
using var encoder = await SentenceTransformers.Qwen3.SentenceEncoder.CreateAsync();

EncodedChunk[] chunks = await encoder.ChunkAndEncodeAsync(
    longDocument,
    chunkLength:  512,   // tokens per chunk (clamped to MaxChunkLength)
    chunkOverlap: 64,    // tokens of overlap between consecutive chunks
    reportProgress: p => Console.WriteLine($"{p:P0}"));

foreach (var chunk in chunks)
{
    // chunk.Text   -> the chunk's source text
    // chunk.Vector -> its embedding
    Index(chunk.Text, chunk.Vector);
}
```

Need to map results back to their position in the original text (e.g. to highlight a passage)? Use
`ChunkAndEncodeAlignedAsync`, which additionally returns character offsets (`Start`, `LastStart`,
`ApproximateEnd`) into the source. There are also tagged variants
(`ChunkAndEncodeTaggedAsync` / `…AlignedAsync`) for carrying per-chunk metadata such as page numbers
through the chunking pipeline.

### Choosing where weights are stored

For the downloaded models you can control the cache location and the source URL:

```csharp
using var encoder = await SentenceTransformers.Qwen3.SentenceEncoder.CreateAsync(
    downloadToPath: "/var/models/qwen3.onnx");

// Or point at your own mirror:
using var harrier = await SentenceTransformers.Harrier.Medium.SentenceEncoder.CreateAsync(
    modelUrl:     "https://my-mirror.example.com/harrier/model_quantized.onnx",
    modelDataUrl: "https://my-mirror.example.com/harrier/model_quantized.onnx_data");
```

You can also pass a custom `Microsoft.ML.OnnxRuntime.SessionOptions` to any constructor / `CreateAsync`
to tune threading or enable hardware execution providers.

### Choosing a Harrier quantization

Both Harrier packages ship multiple quantization formats — pick the one that fits your CPU / GPU
memory budget. URLs for every variant are exposed as constants on `SentenceEncoder.Quantizations`:

| Variant     | Constant                                       | Harrier Medium (0.6b) weights | Harrier Small (270m) weights |
| ---         | ---                                            | ---:                          | ---:                          |
| Full (fp32) | `Quantizations.FullModelUrl`                   | 2.09 GB (+306 MB)             | 1.11 GB                       |
| FP16        | `Quantizations.Fp16ModelUrl`                   | 1.20 GB                       | 553 MB                        |
| Q4          | `Quantizations.Q4ModelUrl`                     | 399 MB                        | 205 MB                        |
| Q4 + FP16   | `Quantizations.Q4Fp16ModelUrl` *(default)*     | 353 MB                        | 172 MB                        |
| Quantized   | `Quantizations.QuantizedModelUrl`              | 706 MB                        | 344 MB                        |

`Q4F16` is the default — it's the smallest variant on disk and keeps multilingual retrieval quality
close to the unquantized reference. Pick `Quantized` when you need a pure float32 output and
broader ONNX-Runtime execution-provider compatibility; `FP16` for the most precision per byte;
`Full` for the unquantized reference. Each entry has a matching `…ModelDataUrl` constant for the
external weights file (and the Harrier Medium `Full` variant additionally has `FullModelDataUrl2`
because its fp32 weights are split into two files).

```csharp
using SentenceTransformers.Harrier.Small;

// Use the unquantized reference instead of the default Q4F16:
using var encoder = await SentenceEncoder.CreateAsync(
    modelUrl:     SentenceEncoder.Quantizations.FullModelUrl,
    modelDataUrl: SentenceEncoder.Quantizations.FullModelDataUrl);
```

## Performance

Encoding throughput on a 4-core Intel® Xeon® @ 2.80 GHz, .NET 10 / ONNX Runtime 1.24.2 CPU
execution provider, batch size 8, average of 5 iterations after a warmup. The full numbers and
the harness used to produce them live in
[`SentenceTransformers.Benchmark/BENCHMARK-RESULT.md`](SentenceTransformers.Benchmark/BENCHMARK-RESULT.md);
re-run with `dotnet run --project SentenceTransformers.Benchmark -c Release`.

| Model                 | Dim  | ms/iter (batch=8) | Embeddings / hour | Relative to MiniLM |
| ---                   | ---: | ---:              | ---:              | ---:               |
| MiniLM-L6-v2          |  384 |   267             |        107,874    | 1.0×               |
| ArcticXs              |  384 |   350             |         82,257    | 1.3× slower        |
| Harrier-Small-270m    |  640 | 2,441             |         11,797    | 9× slower          |
| Qwen3-0.6B            | 1024 | 3,944             |          7,301    | 15× slower         |
| Harrier-Medium-0.6B   | 1024 | 10,098            |          2,852    | 38× slower         |

The smaller / English-only models are an order of magnitude faster on CPU and are the right
default for interactive query embedding. The multilingual 0.6b-class models trade speed for
quality and language coverage — pin them to a background worker and feed them large batches when
you can. A GPU / DirectML / CoreML execution provider helps the 0.6b-class models the most.

## How it works

Each model package contains:

- a tokenizer (WordPiece for the BERT-family embedded models, BPE via Hugging Face tokenizers for the
  Qwen3 / Harrier Medium / Harrier Small models),
- the ONNX graph (embedded, or downloaded on first use), and
- a thin `SentenceEncoder` that tokenizes, runs ONNX Runtime inference, pools the token outputs and
  L2-normalizes the result.

The shared `SentenceTransformers` package provides the [`ISentenceEncoder`](SentenceTransformers/src/ISentenceEncoder.cs)
contract and the default-implemented chunking helpers, so the model packages only implement
`EncodeAsync` and expose their `MaxChunkLength` / `Tokenizer`.

## Contributing & building

```bash
dotnet build SentenceTransformers.sln -c Release
dotnet test  SentenceTransformers.sln
```

NuGet packages are produced and published by the Azure DevOps pipeline in
[`.devops/azure-pipelines.yml`](.devops/azure-pipelines.yml) on pushes to `main`.

## License

[MIT](https://opensource.org/licenses/MIT). The BERT tokenizers are derived from
[BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers) (MIT, © 2021 Othneil Drew). Each wrapped
model is distributed under its own upstream license — see the linked Hugging Face model pages.
