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
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.Harrier.Small.Pure.svg?label=SentenceTransformers.Harrier.Small.Pure)](https://www.nuget.org/packages/SentenceTransformers.Harrier.Small.Pure/) [![Downloads](https://img.shields.io/nuget/dt/SentenceTransformers.Harrier.Small.Pure.svg?label=)](https://www.nuget.org/packages/SentenceTransformers.Harrier.Small.Pure/) | [harrier-oss-v1-270m](https://huggingface.co/microsoft/harrier-oss-v1-270m) (**pure C#, no ONNX**) | 640 | 32768 | Multilingual | Downloaded on first use |
| [![NuGet](https://img.shields.io/nuget/v/SentenceTransformers.EmbeddingGemma2.svg?label=SentenceTransformers.EmbeddingGemma2)](https://www.nuget.org/packages/SentenceTransformers.EmbeddingGemma2/) [![Downloads](https://img.shields.io/nuget/dt/SentenceTransformers.EmbeddingGemma2.svg?label=)](https://www.nuget.org/packages/SentenceTransformers.EmbeddingGemma2/) | EmbeddingGemma 2: [text-270m](https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm), [text-vision-440m](https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm), [740m](https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm) (**pure C#**; text, images, audio) | 768 (MRL: 512/256/128) | 1024 | Multilingual | Downloaded on first use |

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
dotnet add package SentenceTransformers.EmbeddingGemma2
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

### Harrier Small, pure C# — no ONNX, no native dependencies

`SentenceTransformers.Harrier.Small.Pure` is a 100% managed reimplementation of Harrier Small. It runs
the Gemma3 forward pass and the Gemma BPE tokenizer **entirely in C#** (on top of
`System.Numerics.Tensors`), with **no ONNX Runtime and no native tokenizer** — so there is not a single
`.so`/`.dll`/`.dylib` to ship. That makes it trim/AOT-friendly and portable to anywhere .NET runs,
including Blazor WebAssembly and mobile. The API mirrors the ONNX package:

```csharp
using SentenceTransformers.Harrier.Small.Pure;

// Downloads the original bfloat16 safetensors weights (~540 MB) on first use, then loads them.
using var encoder = await SentenceEncoder.CreateAsync();

// Queries take a task instruction prefix; documents are encoded as-is.
float[][] queryVectors = await encoder.EncodeQueriesAsync(
    new[] { "how much protein should a female eat" },
    SentenceEncoder.Prompts.WebSearchQuery);

float[][] docVectors = await encoder.EncodeAsync(new[] { "…a passage about dietary protein…" });
// vectors are L2-normalized float[640]
```

It produces the same embeddings as the reference: pure **fp32** reproduces the query/document
similarity matrix published on the [model card](https://huggingface.co/microsoft/harrier-oss-v1-270m)
to within **0.01** — actually closer to the reference than the shipped ONNX `Q4F16` build.

**Choosing a quantization.** The transformer weights can be loaded at reduced precision to cut both
memory and inference time. Pass a `Quantization` to `CreateAsync` (or the constructor):

```csharp
using SentenceTransformers.Harrier.Small.Pure;
using SentenceTransformers.Harrier.Small.Pure.Model;

// fp32 (default, most faithful), Int8 (recommended — fastest & ~40% less memory), or Int4 (smallest).
using var encoder = await SentenceEncoder.CreateAsync(quantization: Quantization.Int8);
```

The `Int8`/`Int4` paths run as true int8 GEMMs and pick the best instruction set available at runtime:
`vpdpbsud` on 512-bit registers (`AvxVnniInt8.V512`, 64 int8 MACs/instruction), `vpdpbusd`/`vpdpbsud`
on 256-bit (`AvxVnni`/`AvxVnniInt8`), or a widen + `vpmaddwd` sequence on AVX-512 / AVX2 CPUs. On an
AVX-512 host, also set `DOTNET_PreferredVectorBitWidth=512` to let the JIT emit 512-bit vectors.

**Benchmark — pure C# vs ONNX** (harrier-oss-v1-270m, .NET 10, 4-core Xeon, single-text encode):

| Variant | Native deps | Max err vs model card¹ | Short text | ~512-token text | Resident weights² |
| --- | --- | ---: | ---: | ---: | ---: |
| ONNX `Q4F16` (`SentenceTransformers.Harrier.Small`) | ONNX Runtime | 2.30 | **~40 ms** | **~0.7 s** | **~172 MB** (native) |
| Pure **fp32** | none | **0.01** | 226 ms | 4.4 s | ~740 MB |
| Pure **int8** | none | 0.97 | 86 ms | 1.6 s | ~440 MB |
| Pure **int4** | none | 1.42 | 260 ms | 3.8 s | ~390 MB |

¹ Largest absolute deviation (on a 0–100 cosine×100 scale) from the published query/document score
matrix; lower is more faithful — every pure variant tracks the reference more closely than the ONNX
`Q4F16` weights. ² Approximate model-weight memory; all pure variants share the same bfloat16
token-embedding table (~335 MB), which is the floor. Numbers are **hardware-dependent** — the run above
is an AVX-512 server CPU where the int8 dot uses widen + `vpmaddwd`; CPUs with the int8-VNNI
instructions are substantially faster (see below).

**How to read this.** `Int8` is the recommended pure setting — ~2.5× faster than fp32, ~40 % smaller, and
still more faithful than ONNX `Q4F16`.

How close it gets to ONNX depends on the CPU's int8 instruction set. ONNX Runtime's MLAS uses hand-tuned
assembly with **4-bit weights** and **int8-VNNI** (`vpdpbusd`/`vpdpbsud`). The pure build emits int8-VNNI
too when the runtime exposes it: `AvxVnni` (256-bit, Alder Lake and newer client CPUs) or
`AvxVnniInt8.V512` (512-bit, AVX10.2 / Granite Rapids-class CPUs) — on those it is within ~1.5–2× of ONNX
and can approach parity with the 512-bit path. The one gap is "classic" AVX-512 servers that report the
`avx512_vnni` CPUID flag but not `AvxVnni`/`AvxVnniInt8`/AVX10: .NET 10 has no standalone `Avx512Vnni`
intrinsic, so there the pure build must fall back to widen + `vpmaddwd` (~6× the instructions) and lands
~2.5× off ONNX. Either way the pure build's case is **zero native dependencies** (trim/AOT/WASM/mobile,
one managed package) and **higher fidelity**, at a CPU-inference cost within a small multiple of ONNX.

### EmbeddingGemma 2 — multimodal (text, images, audio), pure C#

`SentenceTransformers.EmbeddingGemma2` runs Google's **EmbeddingGemma 2** family from the official
LiteRT-LM bundles (`.litertlm`), in a single package with **no native dependencies** — no LiteRT /
TFLite runtime, no native tokenizer, no image or audio codec libraries:

| `EmbeddingGemma2Model` | Bundle | Download | Inputs |
| --- | --- | ---: | --- |
| `Text270M` (default) | [embeddinggemma-2-text-270m](https://huggingface.co/litert-community/embeddinggemma-2-text-270m-litert-lm) | 165 MB | text |
| `TextVision440M` | [embeddinggemma-2-text-vision-440m](https://huggingface.co/litert-community/embeddinggemma-2-text-vision-440m-litert-lm) | 388 MB | text, images |
| `Multimodal740M` | [embeddinggemma-2-740m](https://huggingface.co/litert-community/embeddinggemma-2-740m-litert-lm) | 485 MB | text, images, audio |

All three share the same text tower, so text embeddings are identical across bundles, and images and
audio land in the same 768-d space as text.

```csharp
using SentenceTransformers.EmbeddingGemma2;

// Downloads the bundle once (cached under the temp folder), then loads it.
using var encoder = await SentenceEncoder.CreateAsync(EmbeddingGemma2Model.Multimodal740M);

// Retrieval: task prompt for queries, "title: … | text: …" for documents.
float[][] queries = await encoder.EncodeQueriesAsync(new[] { "What causes the northern lights?" });
float[][] docs = await encoder.EncodeDocumentsAsync(
    new[] { "Auroras are caused by charged particles from the sun." }, titles: new[] { "Aurora" });

// Images (PNG / JPEG / BMP, or raw RGB/RGBA/BGRA pixels) and audio (WAV, PCM16 or float samples).
float[] photo = await encoder.EncodeImageAsync(EmbeddingGemma2Image.FromFile("photo.jpg"));
float[] clip = await encoder.EncodeAudioAsync(EmbeddingGemma2Audio.FromFile("speech.wav"));

// Interleaved content → one embedding (strings convert implicitly).
float[] mixed = await encoder.EncodeContentAsync(new EmbeddingGemma2Content[]
{
    "A photo of", EmbeddingGemma2Image.FromFile("photo.jpg"), "taken at night",
});

// Matryoshka: shorter vectors (re-normalized) for cheaper storage.
encoder.OutputDimension = 256;
```

Other task prompts are in `SentenceEncoder.Prompts` (question answering, fact checking, code retrieval,
classification, clustering, sentence similarity). Inputs longer than 1024 tokens are truncated by default.
Set `OverflowStrategy` to `ChunkAndAverage` or `Error` to change that; the long-document chunking helpers
(`ChunkAndEncodeAsync`, …) work as with every other encoder. Use `VisionTokensPerImage = 70` to halve
the image cost (the default is 140 soft tokens per image).

**Fidelity.** The package is a port of the `litert-lm` runtime's embedding engine, not an approximation.
It reproduces the embeddings of the LiteRT-LM 0.18 CPU engine (XNNPACK's AVX-512 path) **bit for bit**
(maximum absolute difference 0) for text, images, interleaved text + image, and audio:
- **Text:** the SentencePiece tokenizer is token-for-token identical to the reference.
- **Images:** decoding (PNG / JPEG / BMP) and the sRGB Catmull-Rom resize reproduce `stb_image` /
  `stb_image_resize` v0.97 (the version the engine links) byte for byte.
- **Audio:** the log-mel front-end (framing, Hann window, single-precision KISS FFT, HTK mel filterbank,
  correctly rounded `logf`) matches the engine's to the bit, and long clips use the engine's streaming
  chunk schedule.
- **Model graphs:** the int4 / int2 / int8 weights are executed with the same dynamic (`qd8`) and static
  int8 activation quantization, graph rewrites (e.g. `x·x` → squared reductions, dequantize +
  `BATCH_MATMUL` → f32 × int8 GEMM) and accumulation orders as the XNNPACK kernels the engine runs. The
  transcendental functions (RMSNorm's reciprocal square root, GELU, softmax `exp`, sigmoid, tanh, RoPE
  `sin`/`cos`) are ports of XNNPACK's AVX-512 polynomial kernels rather than calls into the .NET math
  library.

The results do not depend on the host CPU. The 16-lane AVX-512 reductions are emulated with 128-bit
vectors, and the `vrsqrt14` estimate is reproduced from a table measured on AVX-512 hardware where the
instruction is missing. On x64 this is verified with AVX-512, AVX2, SSE-only and scalar code paths, so
the same input gives the same embedding on any machine. (The engine itself only gives these values on
AVX-512 CPUs, because its AVX2 kernels round differently.) ARM64 runs the same 128-bit, fused
multiply-add code paths but has not been verified. CPUs without FMA instructions (pre-2013 x64) fall back
to software fused multiply-add, which is exact but several times slower.

**Performance** (4-vCPU Ice Lake AVX-512 VM, steady state after warm-up, best of three alternating runs; the
LiteRT-LM 0.18 engine measured on the same machine through its Python API):

| Workload | Pure C#, 1 thread | Pure C#, 4 threads | LiteRT-LM, 1 thread | LiteRT-LM, 4 threads |
| --- | ---: | ---: | ---: | ---: |
| Short query (15 tokens) | 55 ms | 20 ms | 133 ms | 59 ms |
| Batch of 32 sentences | 0.90 s | 0.30 s | 4.3 s | 1.9 s |
| One 1003-token document | 2.8 s | 0.93 s | 2.4 s | 0.67 s |
| One image (640×480, 140 tokens, 440M) | 5.0 s | 1.5 s | 1.5 s | 1.3 s |
| 1 s of audio (740M) | 360 ms | 198 ms | 468 ms | 258 ms |
| 3.3 s of audio (740M) | 830 ms | 407 ms | 941 ms | 458 ms |

How the managed port gets there:
- **Integer GEMMs:** INT4/INT2 weights stay packed in memory (half and a quarter of a byte per weight) in
  32-channel panels and run as XNNPACK-style broadcast kernels (`vpmaddubsw`, every output channel in its
  own lane, AVX-512BW or AVX2). 8-bit weights use the same kernels on their ±64 part plus an exact sparse
  correction for the rare larger values. Other CPUs use row kernels (`AvxVnni`, `vpmaddwd`, ARM `sdot`).
- **Float GEMMs:** a 12 × 32 AVX-512 FMA tile; attention packs `Kᵀ` and `V` once per head.
- **Scheduling:** a spinning worker pool keeps the many sub-millisecond operations parallel, and the
  audio graph executor recycles its activation buffers between ops and streaming chunks.

The remaining gap is in compute-bound int8 work on few threads (long documents, and above all the 8-bit
vision tower). The engine uses the 512-bit `vpdpbusd` (AVX-512 VNNI) instruction there, which .NET 10 only
exposes on CPUs that also report AVX-VNNI or AVX10.

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

## Fine-tuning for your use case (real weight-space LoRA)

> 📖 **See [LORA.md](LORA.md)** for the full guide: how it works internally, every option, negative-sample
> handling, stop conditions, and library + CLI training walkthroughs.

You can specialize the **pure-C# encoders** — MiniLM, Arctic XS and Harrier Small — for a specific
domain (support tickets, legal clauses, product descriptions, a particular language pair) by training a
**real weight-space LoRA adapter** from a set of *related pairs* (a query and a relevant passage, two
paraphrases, a question and its duplicate…). This is proper LoRA: small low-rank factors are injected
*inside* the transformer's attention/MLP projections (`W + (α/r)·B·A`), and the loss is backpropagated
through the whole (frozen) network. The forward **and backward** passes run entirely in pure C# — no ONNX
Runtime, and no PyTorch/autodiff dependency. A shared tensor autograd engine (`SentenceTransformers.Training.Autograd`)
powers both model families.

> **MiniLM / Arctic XS** (`SentenceTransformers.Bert.Pure`, the `BertModel` architecture) read their
> full-precision weights directly from the fp32 ONNX graphs already embedded in the MiniLM / Arctic
> packages, so training is fully self-contained — no model download.
>
> **Harrier Small** (`SentenceTransformers.Harrier.Small.Pure`, a Gemma3 decoder) trains through the same
> autograd engine and LoRA objectives via `Gemma3LoraTrainer` / `Gemma3LoraEncoder`; its bf16 weights are
> downloaded on first use, and because it is a ~270M-param decoder each step is much heavier than the small
> BERTs (keep the batch / sequence length modest). LoRA there targets q/k/v/o and the gate/up/down MLP
> projections.
>
> The remaining ONNX-only models (Qwen3, Harrier Medium) are inference-only and are not trainable here.

```csharp
using SentenceTransformers.Bert.Pure;
using SentenceTransformers.Bert.Pure.Model;
using SentenceTransformers.Bert.Pure.Training;
using SentenceTransformers.Training;

// Pure-C# MiniLM (weights fetched once from HuggingFace, or use SentenceEncoder.LoadFromOnnx to reuse
// the weights embedded in the SentenceTransformers.MiniLM package with no download at all).
using var baseEncoder = await SentenceEncoder.CreateMiniLMAsync();

var dataset = new SentencePairDataset(new[]
{
    new SentencePair("how do I reset my password", "Use the ‘Forgot password’ link on the sign-in page.", 0.95f),
    new SentencePair("cancel my subscription",      "Go to Billing → Manage plan → Cancel.",              0.95f),
    // … a few hundred to a few thousand related (optionally scored) pairs …
});

var report = await BertLoraTrainer.TrainAsync(baseEncoder, dataset, new BertLoraTrainingOptions
{
    Objective = BertTrainingObjective.CoSent,     // pairwise ranking loss — directly targets STS Spearman
    Rank      = 8,                                // LoRA rank
    Targets   = LoraTargets.Attention,            // inject into Q, K, V and the attention output
    Epochs    = 10,
});

report.Adapter.Save("support-faq.lora");

// The tuned model is a drop-in ISentenceEncoder (the adapter is folded into the transformer at runtime):
using var tuned = baseEncoder.WithAdapter(report.Adapter);
float[][] vectors = await tuned.EncodeAsync(new[] { "I can't log in" });
```

**How it works.** A tiny tensor-level autograd engine records the BERT forward pass; the exact gradients
for the injected LoRA factors are obtained by replaying the tape backward (verified against finite
differences in the test-suite). Only the low-rank factors — and optionally a learned output-centering bias
— are trainable; every base weight stays frozen, so the gradients for the frozen network are skipped (the
LoRA efficiency win). The held-out validation set is scored with STS Spearman and top-1 retrieval accuracy
and the best adapter is kept.

**Objectives** (`BertLoraTrainingOptions.Objective`):

- `Contrastive` *(default)* — symmetric InfoNCE (MultipleNegativesRanking) with in-batch **and hard
  negatives** (explicit triplets via `SentencePair.Negative`, plus optional mined negatives), with
  false-negative masking. Best for **retrieval / nearest-neighbour** separation.
- `CoSent` — the CoSENT pairwise-ranking loss over graded pairs. It optimizes the *ordering* that Spearman
  measures and consistently beats plain cosine-MSE on STS. Best when you care about **graded similarity**.
- `CosineRegression` — mean-squared error between adapted cosine and the gold `[0,1]` score.

**Extras** (all optional): warmup + cosine LR schedule, a learned temperature (`LearnableTemperature`),
multi-seed model selection (`NumSeeds`), a learned output-centering bias (`UseOutputBias`) and post-hoc
ZCA whitening (`ApplyWhitening`) to counter embedding anisotropy, Matryoshka sub-dimension losses
(`MatryoshkaDims`), asymmetric instruction prefixes (`QueryPrefix` / `DocumentPrefix`), and a choice of
which projections carry adapters (`Targets`: `Attention`, `Mlp` or `All`).

### Training CLI + example dataset

The `SentenceTransformers.LoraTraining` project is a ready-to-run console app that fine-tunes MiniLM,
Arctic XS or Harrier Small against one of two bundled example datasets (`--dataset`):

- **`stsb`** — the English [STS Benchmark](https://github.com/PhilipMay/stsb-multi-mt), a broad
  general-English similarity set, downloaded on demand.
- **`patent`** — the [Google Patent Phrase Similarity](https://www.kaggle.com/datasets/google/google-patent-phrase-similarity-dataset)
  dataset (CC BY 4.0), embedded directly in the app (no download). Terse, domain-specific technical
  phrases where a general encoder has real headroom.

```bash
cd SentenceTransformers.LoraTraining

# General-English STS Benchmark (needs a one-time dataset download; model weights are embedded):
dotnet run -c Release -- download
dotnet run -c Release -- train --model minilm --objective cosent --rank 8 --epochs 10 --output-bias
dotnet run -c Release -- eval  --model minilm --adapter ./adapters/minilm-stsb.lora --split test

# Domain-specific patent phrases (embedded, no download) — graded scores:
dotnet run -c Release -- train --model arctic --dataset patent --objective cosent --rank 8 --whitening
dotnet run -c Release -- eval  --model arctic --dataset patent --adapter ./adapters/arctic-patent.lora --split test

# Harrier Small (Gemma3, pure C#) — bf16 weights downloaded on first use; heavier, so keep batch small:
dotnet run -c Release -- train --model harrier-small --dataset patent --objective cosent --rank 8 --batch 8 --max-tokens 64
```

`--model` accepts `minilm`, `arctic` or `harrier-small`. Run `dotnet run -- help` for the full option list (objective,
targets, rank, α, learning rate, warmup, temperature, mined negatives, learnable temperature, output bias,
whitening, Matryoshka, query/doc prefixes, multi-seed, …). Training prints per-epoch validation retrieval
accuracy and STS Spearman and a base-vs-tuned summary at the end.

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

The `SentenceTransformers.Harrier.Small.Pure` package is the exception: instead of ONNX Runtime it
ships its own pure-managed implementation of the model. It reads the original
[safetensors](https://github.com/huggingface/safetensors) weights directly, runs the Gemma3
decoder forward pass (RMSNorm, grouped-query attention with Q/K-norm and RoPE, GeGLU MLP, last-token
pooling) on [`TensorPrimitives`](https://learn.microsoft.com/dotnet/api/system.numerics.tensors.tensorprimitives),
and tokenizes with a from-scratch Gemma byte-level BPE tokenizer — so it depends only on the .NET base
class library and `System.Numerics.Tensors`.

`SentenceTransformers.EmbeddingGemma2` is also pure managed code. It reads the `.litertlm` container
and the TFLite flatbuffers inside it, with weights taken directly from the int4 / int2 / int8 tensors.
- **Text tower:** a dedicated SIMD implementation of the Gemma encoder (per-layer embeddings,
  local/global attention with RoPE, GeGLU, mean pooling and projection) on a packed int8 GEMM and an
  FMA SGEMM.
- **Vision and audio towers:** executed op by op from their graphs by a small TFLite graph executor,
  which routes the heavy `FULLY_CONNECTED` / `BATCH_MATMUL` ops through the same kernels.
- **Inputs:** the SentencePiece BPE tokenizer, the image decoders and resizer, and the audio front-end
  are C# ports of the libraries the reference runtime uses.

## Contributing & building

```bash
dotnet build SentenceTransformers.sln -c Release
dotnet test  SentenceTransformers.sln
```

The EmbeddingGemma 2 parity tests compare against fixtures generated with Google's `litert-lm` Python
runtime by [`scripts/generate_embeddinggemma2_reference.py`](scripts/generate_embeddinggemma2_reference.py).
The tests that need model weights are opt-in:
- Point `EMBEDDINGGEMMA2_MODELS_DIR` at the `.litertlm` files, or set `EMBEDDINGGEMMA2_DOWNLOAD=1` to
  fetch them.
- For the per-layer comparisons, also set `EMBEDDINGGEMMA2_REFERENCE_DIR` to the script's dump
  directory.

NuGet packages are produced and published by the Azure DevOps pipeline in
[`.devops/azure-pipelines.yml`](.devops/azure-pipelines.yml) on pushes to `main`.

## License

[MIT](https://opensource.org/licenses/MIT). The BERT tokenizers are derived from
[BERTTokenizers](https://github.com/NMZivkovic/BertTokenizers) (MIT, © 2021 Othneil Drew). The
EmbeddingGemma 2 image decoders and resizer are ports of [stb_image / stb_image_resize](https://github.com/nothings/stb)
(public domain / MIT, Sean Barrett), its FFT is a port of [KISS FFT](https://github.com/mborgerding/kissfft)
(BSD-3-Clause, Mark Borgerding), and its activation kernels follow
[XNNPACK](https://github.com/google/XNNPACK) (BSD-3-Clause, Google). Each wrapped
model is distributed under its own upstream license — see the linked Hugging Face model pages. The
[Google Patent Phrase Similarity](https://www.kaggle.com/datasets/google/google-patent-phrase-similarity-dataset)
dataset bundled with the `SentenceTransformers.LoraTraining` example is © Google, licensed
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
