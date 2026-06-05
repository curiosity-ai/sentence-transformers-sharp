# Embedding Benchmark Results

Vector-generation speed across **all five** shipped encoders, reported as **tokens/s** and as a
**ratio vs MiniLM** (the smallest/fastest baseline = `1.00x`):

* **MiniLM-L6-v2** (384-d, embedded)
* **ArcticXs** (384-d, embedded)
* **Qwen3-Embedding-0.6B** (1024-d, downloaded, ONNX dynamic-uint8)
* **Harrier Medium / harrier-oss-v1-0.6b** (1024-d, downloaded, Q4F16 default)
* **Harrier Small / harrier-oss-v1-270m** (640-d, downloaded, Q4F16 default)

All runs used the same harness ([`Program.cs`](Program.cs)):

* **Batch size:** 8 (corpus of 4 synthetic ~3-paragraph documents, repeated to fill the batch)
* **Warmup:** 1 iteration
* **Iterations:** 5 (timed)
* **Tokens** are counted from each model's *own* tokenizer (sum of the attention mask), so the
  metric stays comparable even though the families tokenize differently (WordPiece vs byte-level BPE).

### Environment

* **CPU:** Intel Xeon @ 2.80 GHz, 4 cores
* **RAM:** 15 GiB
* **OS:** Linux 6.18.5
* **Runtime:** .NET 10.0.108, CPU execution provider only (no GPU)
* **ONNX Runtime:** Microsoft.ML.OnnxRuntime 1.24.2

> ⚠️ These are **CPU-only** numbers on a modest 4-core VM. They show *relative* throughput between
> the models, not the best achievable speed. Expect large gains from a GPU/accelerated execution
> provider, larger batches, and a beefier CPU.

---

## Summary table (tokens/s, ratio vs MiniLM)

| Model          | Dim  | Batch | ms/iter | Tokens/s | vs MiniLM |
|----------------|------|-------|---------|----------|-----------|
| MiniLM-L6-v2   | 384  | 8     | 215.2   | 9,514.9  | 1.00x     |
| ArcticXs       | 384  | 8     | 274.2   | 9,942.7  | 1.04x     |
| Qwen3-0.6B     | 1024 | 8     | 3,176.2 | 685.1    | 0.07x     |
| Harrier-Medium | 1024 | 8     | 7,340.9 | 296.4    | 0.03x     |
| Harrier-Small  | 640  | 8     | 1,945.5 | 1,036.2  | 0.11x     |

---

## Quick analysis

* **MiniLM-L6-v2** and **ArcticXs** are in a class of their own at **~9.5k–9.9k tokens/s**. They are
  the small embedded BERT-family models (384-d) and are the right default for interactive /
  real-time query embedding. ArcticXs is marginally faster per token here (1.04x) despite a slightly
  higher ms/iter, because its tokenizer emits more tokens for the same text.
* **Harrier-Small** (270m, 640-d) runs at **~1.0k tokens/s — roughly 0.11x MiniLM (~9x slower)**.
  A reasonable middle ground when you need multilingual coverage without the 0.6b weights.
* **Qwen3-0.6B** (1024-d) lands at **~685 tokens/s ≈ 0.07x MiniLM (~14x slower)**.
* **Harrier-Medium** (0.6b, 1024-d) is the slowest at **~296 tokens/s ≈ 0.03x MiniLM (~32x slower)**
  on CPU. Its default Q4F16 graph leans on the `GatherBlockQuantized` / quantized matmul contrib
  ops, which are not as CPU-optimized as the embedded BERT models — so on CPU it pays a steep price
  for the higher-quality multilingual 1024-d embeddings.

### Recommendation

* **Interactive search / real-time query embedding:** MiniLM or ArcticXs.
* **Multilingual, quality-sensitive, mostly offline indexing:** Harrier Small (lighter) or
  Qwen3 / Harrier Medium (highest quality 1024-d) — ideally with batching and/or GPU acceleration.

---

## Reproducing

```bash
dotnet run --project SentenceTransformers.Benchmark -c Release
```

The downloaded models (Qwen3, Harrier Medium, Harrier Small) fetch their ONNX weights on first run
and cache them under the system temp folder; subsequent runs skip the download.

### Note: tokenizer collision when referencing multiple BPE models

Qwen3, Harrier Medium and Harrier Small each ship a `Resources/tokenizer.json` that copies to the
**same** output path. A project that references more than one of them gets a single, arbitrarily
"winning" `tokenizer.json` in its output directory, so the other BPE encoders silently load the wrong
vocabulary and fail at inference with an out-of-range token id (e.g.
`indices element out of data bounds, idx=236761 ...` in a `Gather` / `GatherBlockQuantized` node).

The benchmark works around this by deleting the colliding
`Resources/tokenizer.json` at startup, which makes each encoder fall back to its own *embedded*
tokenizer (extracted to a per-package temp path). The same workaround is used in
[`SentenceTransformers.Test/HarrierVariantsTest.cs`](../SentenceTransformers.Test/HarrierVariantsTest.cs).

---

## Notes / next steps

* Re-run with **larger batches** (16/32) — the 1024-d models usually amortize much better and the
  ratios should narrow.
* Capture **GPU / accelerated EP** numbers; the gap between the small BERT models and the 0.6b
  transformer encoders is expected to shrink dramatically off CPU.
* The per-model `Tokens/s` already normalizes away tokenizer differences; `ms/iter` is included for
  raw latency context at this batch size.
