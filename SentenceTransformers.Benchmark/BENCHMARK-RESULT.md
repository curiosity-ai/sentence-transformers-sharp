# Embedding Benchmark Results

This document captures local benchmark measurements for the five sentence-embedding encoders
shipped by this repo:

| Model                  | Dim  | Variant on disk                  |
| ---                    | ---: | ---                              |
| **MiniLM-L6-v2**       |  384 | Embedded (no download)           |
| **ArcticXs**           |  384 | Embedded (no download)           |
| **Harrier-Small-270m** |  640 | Downloaded — Q4F16 (~172 MB)     |
| **Harrier-Medium-0.6B**| 1024 | Downloaded — Q4F16 (~353 MB)     |
| **Qwen3-0.6B**         | 1024 | Downloaded — uint8 (~700 MB)     |

All runs used the same benchmark harness (`SentenceTransformers.Benchmark`) with:

* **Batch size:** 8
* **Warmup iterations:** 1
* **Measured iterations:** 5

Each input is a 3-paragraph, 5-sentence-per-paragraph synthetic document built from a fixed
vocabulary by `MakeParagraph` in `Program.cs` — average input length is ~600 UTF-8 bytes per item.

## Environment

* CPU: Intel® Xeon® @ 2.80 GHz, 4 cores
* OS: Linux 6.18.5 x86_64
* Runtime: .NET 10 (SDK 10.0.108), Release configuration
* ONNX Runtime: 1.24.2 (CPU execution provider, default session options)
* sentence-transformers-sharp: 26.6.1581

---

## Raw results

### MiniLM-L6-v2

* **Batch:** 8
* **Dim:** 384
* **Time:** 1.3 s for 5 iterations
* **Avg:** 267.0 ms/iter (~33 ms per embedding at batch=8)
* **Input throughput:** 198.4 UTF-8 MB / hour
* **Output throughput:** 158.0 MB / hour
* **Embeddings / hour:** 107,874

### ArcticXs

* **Batch:** 8
* **Dim:** 384
* **Time:** 1.8 s for 5 iterations
* **Avg:** 350.1 ms/iter (~44 ms per embedding at batch=8)
* **Input throughput:** 151.3 UTF-8 MB / hour
* **Output throughput:** 120.5 MB / hour
* **Embeddings / hour:** 82,257

### Harrier-Small-270m (Q4F16 default)

* **Batch:** 8
* **Dim:** 640
* **Time:** 12.2 s for 5 iterations
* **Avg:** 2,441.4 ms/iter (~305 ms per embedding at batch=8)
* **Input throughput:** 21.7 UTF-8 MB / hour
* **Output throughput:** 28.8 MB / hour
* **Embeddings / hour:** 11,797

### Harrier-Medium-0.6B (Q4F16 default)

* **Batch:** 8
* **Dim:** 1024
* **Time:** 50.5 s for 5 iterations
* **Avg:** 10,097.6 ms/iter (~1.26 s per embedding at batch=8)
* **Input throughput:** 5.2 UTF-8 MB / hour
* **Output throughput:** 11.1 MB / hour
* **Embeddings / hour:** 2,852

### Qwen3-0.6B (uint8 quantized)

* **Batch:** 8
* **Dim:** 1024
* **Time:** 19.7 s for 5 iterations
* **Avg:** 3,944.4 ms/iter (~493 ms per embedding at batch=8)
* **Input throughput:** 13.4 UTF-8 MB / hour
* **Output throughput:** 28.5 MB / hour
* **Embeddings / hour:** 7,301

---

## Summary table

| Model                 | Dim  | Batch | ms/iter  | Embeddings/hour | In MB/hour | Out MB/hour |
| ---                   | ---: | ---:  | ---:     | ---:            | ---:       | ---:        |
| MiniLM-L6-v2          | 384  | 8     | 267.0    | 107,874         | 198.4      | 158.0       |
| ArcticXs              | 384  | 8     | 350.1    |  82,257         | 151.3      | 120.5       |
| Harrier-Small-270m    | 640  | 8     | 2,441.4  |  11,797         |  21.7      |  28.8       |
| Harrier-Medium-0.6B   | 1024 | 8     | 10,097.6 |   2,852         |   5.2      |  11.1       |
| Qwen3-0.6B            | 1024 | 8     | 3,944.4  |   7,301         |  13.4      |  28.5       |

---

## Quick analysis

### Speed

* **MiniLM-L6-v2** is the fastest at **~108k embeddings/hour**.
* **ArcticXs** sits at **~82k embeddings/hour** (≈ **1.3×** slower than MiniLM, double the context window).
* **Harrier-Small-270m** drops to **~12k embeddings/hour** — about **9×** slower than MiniLM, but
  with multilingual coverage across 94 languages and a 32k-token context.
* **Qwen3-0.6B** lands at **~7k embeddings/hour**.
* **Harrier-Medium-0.6B** is the slowest at **~2.9k embeddings/hour** — about **37×** slower than
  MiniLM. The Q4F16 quantization keeps the weights small on disk (~353 MB) but inference is more
  arithmetic-heavy than Qwen3's uint8 graph on a CPU execution provider.

### What the dimension cost looks like in practice

* MiniLM / ArcticXs → **384-d** (1,536 bytes/embedding as float32)
* Harrier-Small      → **640-d** (2,560 bytes/embedding as float32)
* Harrier-Medium / Qwen3 → **1024-d** (4,096 bytes/embedding as float32)

Higher-dim vectors are slower to compare and use more memory in HNSW / similarity indexes — for
corpora over a few million chunks the 384-d models keep the index footprint manageable.

### Recommendations from these numbers

* **Interactive search where queries embed at request time** → MiniLM or ArcticXs. Both finish a
  batch of 8 in under half a second on a 4-core Xeon.
* **Multilingual retrieval, default choice** → Harrier-Small. Order of magnitude slower than
  MiniLM, but the only multilingual option in this list that's also small enough to pre-bundle
  inside a Docker image (default Q4F16 weights at ~172 MB).
* **Highest-quality multilingual, batch / offline workloads** → Harrier-Medium or Qwen3. Pin to a
  background worker and feed it large batches.

---

## Notes / next steps

* Throughput scales sub-linearly with batch size — repeat with **batch=16/32** if your workload
  routinely encodes larger batches; the per-iteration overhead is amortised better there.
* Numbers above are **CPU-only** with default ONNX Runtime session options. A GPU / DirectML /
  CoreML execution provider would help the 0.6b-class models the most.
* Re-run after upstream model updates (e.g. a new Harrier quantization landing on the curiosity.ai
  mirror) to track drift.

Reproduce with:

```bash
dotnet run --project SentenceTransformers.Benchmark -c Release
```
