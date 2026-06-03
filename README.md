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
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.svg?label=SentenceTransformers)](https://www.nuget.org/packages/SentenceTransformers/) | Core interfaces & chunking helpers | — | — | — | — |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.MiniLM.svg?label=SentenceTransformers.MiniLM)](https://www.nuget.org/packages/SentenceTransformers.MiniLM/) | [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) | 384 | 256 | English | Embedded |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.ArcticXs.svg?label=SentenceTransformers.ArcticXs)](https://www.nuget.org/packages/SentenceTransformers.ArcticXs/) | [snowflake-arctic-embed-xs](https://huggingface.co/Snowflake/snowflake-arctic-embed-xs) | 384 | 512 | English | Embedded |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.Qwen3.svg?label=SentenceTransformers.Qwen3)](https://www.nuget.org/packages/SentenceTransformers.Qwen3/) | [Qwen3-Embedding-0.6B](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) | 1024 | 32768 | Multilingual | Downloaded on first use |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.Harrier.svg?label=SentenceTransformers.Harrier)](https://www.nuget.org/packages/SentenceTransformers.Harrier/) | [harrier-oss-v1-0.6b](https://huggingface.co/onnx-community/harrier-oss-v1-0.6b-ONNX) | 1024 | 32768 | Multilingual | Downloaded on first use |

- **Embedded** models bundle the ONNX weights inside the NuGet package, so the encoder is ready
  immediately after construction.
- **Downloaded** models are larger; their weights are fetched once on first use and cached on disk
  (under the system temp folder by default — see [Choosing where weights are stored](#choosing-where-weights-are-stored)).

Pick **MiniLM** for the smallest/fastest footprint, **Arctic XS** for a strong English default,
and **Qwen3** or **Harrier** when you need a larger context window or multilingual coverage.

## Installation

Install only the model package(s) you need (the core `SentenceTransformers` package is pulled in as a
dependency):

```bash
dotnet add package SentenceTransformers.MiniLM
dotnet add package SentenceTransformers.ArcticXs
dotnet add package SentenceTransformers.Qwen3
dotnet add package SentenceTransformers.Harrier
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

### Downloaded models (Qwen3, Harrier)

Larger models download their ONNX weights on first use. Create them with the async `CreateAsync`
factory — the download is cached, so subsequent runs are instant:

```csharp
using SentenceTransformers.Qwen3;

// Downloads the model to a temp folder on first use, then loads it.
using var encoder = await SentenceEncoder.CreateAsync();

float[][] vectors = await encoder.EncodeAsync(new[] { "Hello world" });
// vectors[0] is a float[1024]
```

Harrier is multilingual:

```csharp
using SentenceTransformers.Harrier;

using var encoder = await SentenceEncoder.CreateAsync();

float[][] vectors = await encoder.EncodeAsync(new[]
{
    "Good morning",   // English
    "Buenos días",    // Spanish
    "おはよう",          // Japanese
});
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
using var harrier = await SentenceTransformers.Harrier.SentenceEncoder.CreateAsync(
    modelUrl:     "https://my-mirror.example.com/harrier/model_quantized.onnx",
    modelDataUrl: "https://my-mirror.example.com/harrier/model_quantized.onnx_data");
```

You can also pass a custom `Microsoft.ML.OnnxRuntime.SessionOptions` to any constructor / `CreateAsync`
to tune threading or enable hardware execution providers.

## How it works

Each model package contains:

- a tokenizer (WordPiece for the BERT-family embedded models, BPE via Hugging Face tokenizers for the
  Qwen3 / Harrier models),
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
