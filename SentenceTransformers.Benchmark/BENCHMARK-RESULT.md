# Embedding Benchmark Results

This document captures local benchmark measurements for three sentence embedding encoders:

* **MiniLM-L6-v2** (384-d)
* **ArcticXs** (384-d)
* **Qwen3-Embedding-0.6B (ONNX uint8)** (1024-d)

All runs used the same benchmark harness and:

* **Batch size:** 8
* **Iterations:** 50

---

## Raw results

### MiniLM-L6-v2

* **Batch:** 8
* **Dim:** 384
* **Time:** 4043.9 ms for 50 iterations
* **Avg:** 80.88 ms/iter
* **Input throughput (UTF-8 bytes/s):** 190,756
* **Output throughput (bytes/s):** 151,932
* **Embeddings/s:** 98.91
* **WorkingSet Δ (bytes):** 136,495,104
* **GC heap Δ (bytes):** 6,094,944

### ArcticXs

* **Batch:** 8
* **Dim:** 384
* **Time:** 5726.9 ms for 50 iterations
* **Avg:** 114.54 ms/iter
* **Input throughput (UTF-8 bytes/s):** 134,699
* **Output throughput (bytes/s):** 107,284
* **Embeddings/s:** 69.85
* **WorkingSet Δ (bytes):** 71,172,096
* **GC heap Δ (bytes):** 3,040,536

### Qwen3-0.6B (ONNX uint8)

* **Batch:** 8
* **Dim:** 1024
* **Time:** 67,886.9 ms for 50 iterations
* **Avg:** 1,357.74 ms/iter
* **Input throughput (UTF-8 bytes/s):** 11,363
* **Output throughput (bytes/s):** 24,134
* **Embeddings/s:** 5.89
* **WorkingSet Δ (bytes):** 152,567,808
* **GC heap Δ (bytes):** 7,689,352

---

## Summary table

| Model        | Dim  | Batch | ms/iter  | Emb/s | In MB/s  | Out MB/s | WS Δ (bytes) | GC Δ (bytes) |
|--------------|------|-------|----------|-------|----------|----------|--------------|--------------|
| MiniLM-L6-v2 | 384  | 8     | 80.88    | 98.91 | 0.190756 | 0.151932 | 136,495,104  | 6,094,944    |
| ArcticXs     | 384  | 8     | 114.54   | 69.85 | 0.134699 | 0.107284 | 71,172,096   | 3,040,536    |
| Qwen3-0.6B   | 1024 | 8     | 1,357.74 | 5.89  | 0.011363 | 0.024134 | 152,567,808  | 7,689,352    |

---

## Quick analysis

### 1) Speed / throughput

* **MiniLM-L6-v2** is the fastest at **~99 embeddings/s**.
* **ArcticXs** is slower: **~70 embeddings/s** (≈ **1.4×** slower than MiniLM).
* **Qwen3-0.6B (ONNX)** is dramatically slower: **~5.9 embeddings/s** (≈ **16–17×** slower than MiniLM).

A useful way to think about it:

* At batch=8, **Qwen3** spends about **1.36 seconds per iteration**, i.e. roughly **170 ms per embedding** (not counting any additional pipeline work).

### 2) Vector size impacts downstream storage and bandwidth

* MiniLM/ArcticXs produce **384-d** embeddings (~1,536 bytes/embedding as float32).
* Qwen3 produces **1024-d** embeddings (~4,096 bytes/embedding as float32).

Even if Qwen3 were equally fast (it is not), it would still:

* consume more storage for vector DB / HNSW indexes,
* increase memory bandwidth costs,
* increase query-time compute for similarity scoring.

### 3) Memory observations (Working Set / GC)

* Qwen3 shows the largest **WorkingSet Δ** at **~152.6 MB**, slightly higher than MiniLM’s **~136.5 MB**.
* ArcticXs shows the smallest **WorkingSet Δ** (**~71.2 MB**).

GC deltas are relatively small compared to Working Set deltas, suggesting most of the footprint is native allocations (runtime/model buffers) rather than managed heap.

### 4) Recommendation (based on these numbers)

* If your primary use-case is **interactive search** (compute query embedding in real-time), **MiniLM** (and possibly ArcticXs) are far more suitable.
* **Qwen3-0.6B** looks better suited as an **offline/high-quality embedding option** unless performance can be improved significantly (e.g., larger batch sizes, better ORT settings, hardware acceleration such as GPU/CoreML).

---

## Notes / next steps

* Repeat with **batch=16/32** to see if Qwen3 throughput improves substantially.
* Capture platform details (CPU model, OS, .NET runtime, ORT version, execution provider) for reproducibility.
* If deploying on macOS, consider experimenting with **CoreML EP** (if supported by the model graph) and newer ORT versions.
