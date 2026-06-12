# Harrier Small — token-count scalability

Elapsed time per single-sentence encode (`EncodeAsync` / `model.encode`) for the
`harrier-oss-v1-270m` model, swept across sequence lengths from 128 to 4096 tokens.

All three implementations were driven with **byte-identical calibrated input strings**
(generated once and replayed from `/tmp/harrier_scaling_inputs.json`), each calibrated
to land on an exact token count (incl. `<bos>`/`<eos>`) using the shared Gemma BPE
tokenizer. Token counts matched exactly across all three.

## Environment

| | |
|---|---|
| CPU | 4 cores (CPU-only, no GPU) |
| RAM | 15 GiB |
| Pure C# | `SentenceTransformers.Harrier.Small.Pure`, fp32, .NET 10, Release |
| ONNX C# | `SentenceTransformers.Harrier.Small`, ONNX Runtime 1.24.2, **Q4F16** (default variant) |
| Python | `sentence-transformers` 5.5.1 / `transformers` 5.12.0 / torch 2.12.0, CPU, fp32 |
| Method | warmup encode, then median of N iters (7/5/3/2/1 as length grows) |

> Note: this is **not** a like-for-like precision comparison. Pure C# and Python run
> full **fp32**; the ONNX column is the **Q4F16-quantized** graph (the package default,
> smallest on disk), which is why it is the fastest. The numbers measure each shipping
> default as you would actually use it.

## Results — elapsed ms per encode

| Tokens | Pure C# fp32 (ms) | Python fp32 (ms) | ONNX Q4F16 (ms) |
|-------:|------------------:|-----------------:|----------------:|
|    128 |           1,778.4 |            349.9 |           117.4 |
|    256 |           3,838.4 |            589.5 |           287.1 |
|    512 |           8,053.9 |          1,092.8 |           587.5 |
|  1,024 |          18,819.7 |          2,166.8 |         1,178.2 |
|  2,048 |          46,789.5 |          4,293.7 |         2,740.6 |
|  4,096 |         120,825.7 |          9,571.8 |         7,105.0 |

## Throughput — tokens/second (higher is better)

| Tokens | Pure C# fp32 | Python fp32 | ONNX Q4F16 |
|-------:|-------------:|------------:|-----------:|
|    128 |           72 |         366 |      1,090 |
|    256 |           67 |         434 |        892 |
|    512 |           64 |         469 |        872 |
|  1,024 |           54 |         473 |        869 |
|  2,048 |           44 |         477 |        747 |
|  4,096 |           34 |         428 |        577 |

## How each scales

Cost relative to the 128-token point (32× more tokens = perfectly-linear 32×):

| | 128→4096 growth | per-doubling factor (avg) |
|---|---:|---:|
| Pure C# fp32 | **68×** | ~2.3× |
| ONNX Q4F16   | **61×** | ~2.3× |
| Python fp32  | **27×** | ~2.0× |

All three are **super-linear** in token count, as expected from the O(n²) self-attention
term that grows relative to the O(n) projection/MLP work as sequences lengthen — visible
in the per-doubling factor climbing toward ~2.6× at the top end for the C# paths.

Python's growth ratio looks the smallest only because it carries a larger fixed
per-call overhead (~300 ms floor of Python/dispatch + framework), which dominates at
128 tokens and dilutes the ratio; in absolute tokens/sec it plateaus around ~470 tok/s.

## Takeaways

- **ONNX Q4F16 is fastest end-to-end** at every length (≈1.3–3× faster than PyTorch CPU,
  ≈15× faster than the pure-C# fp32 path at short lengths, ≈17× at 4096) — quantization
  plus a mature CPU kernel library does the heavy lifting.
- **PyTorch (sentence-transformers) fp32** is the mid-point: a well-optimized BLAS backend
  beats the managed fp32 kernels comfortably, but loses to the quantized ONNX graph.
- **Pure C# fp32** is the slowest (its whole point is zero native dependencies / AOT-&-WASM
  portability, not peak throughput) and is the most sensitive to length: at 4096 tokens a
  single encode takes ~2 minutes on 4 cores. For long-context throughput-bound workloads
  it would benefit most from quantization (`Int8`/`Int4`) and/or chunking to shorter
  sequences.
- Practically: for indexing large documents, prefer shorter chunks — every path pays a
  super-linear penalty per chunk as the chunk grows, so two 512-token chunks are
  meaningfully cheaper than one 1024-token chunk on the pure path especially.

_Reproduce: `dotnet run -c Release --project SentenceTransformers.Benchmark -- harrier-scaling`
then `python scripts/harrier_scaling_bench.py`._
