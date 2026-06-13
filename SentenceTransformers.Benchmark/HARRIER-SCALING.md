# Harrier Small — token-count scalability

Elapsed time per single-sentence encode for the `harrier-oss-v1-270m` model, swept across sequence
lengths from 128 to 4096 tokens, for four implementations of the same model.

All were driven with **byte-identical calibrated input strings** (generated once and replayed from
`/tmp/harrier_scaling_inputs.json`), each calibrated to land on an exact token count (incl.
`<bos>`/`<eos>`) using the shared Gemma BPE tokenizer. Token counts matched exactly across all paths.

## Environment

| | |
|---|---|
| CPU | 4 cores (CPU-only, no GPU) |
| RAM | 15 GiB |
| Pure C# | `SentenceTransformers.Harrier.Small.Pure`, .NET 10, Release, fp32 and Int8 |
| ONNX C# | `SentenceTransformers.Harrier.Small`, ONNX Runtime 1.24.2, **Q4F16** (package default) |
| Python | `sentence-transformers` 5.5.1 / `transformers` 5.12.0 / torch 2.12.0, CPU, fp32 |
| Method | warmup encode, then median of N iters (7/5/3/2/1 as length grows) |

> **Important threading caveat.** The pure columns below were produced by the public document
> `EncodeAsync(string[])` path, which builds a default `ParallelOptions` (`MaxDegreeOfParallelism == -1`).
> `ParallelExecution.ForAsync` only fans out when `MaxDegreeOfParallelism > 1`, and `-1` is *not* `> 1`,
> so **the pure encoder ran single-threaded on 1 of 4 cores**, while ONNX and PyTorch used all 4.
> Re-running the pure path with an explicit `MaxDegreeOfParallelism = 4` is **3.2–3.7× faster** (see the
> profile section). The numbers below reflect each path's *shipping default*, which is itself a finding.
>
> This is also **not** a same-precision comparison: Pure-fp32 and Python are fp32; Pure-Int8 is weight+
> activation int8 (projections only); ONNX is Q4F16. See `int8` notes below.

## Results — elapsed ms per encode

| Tokens | Pure fp32 (1 core) | Pure Int8 (1 core) | Python fp32 (4 core) | ONNX Q4F16 (4 core) |
|-------:|-------------------:|-------------------:|---------------------:|--------------------:|
|    128 |            1,778.4 |              670.8 |                349.9 |               117.4 |
|    256 |            3,838.4 |            1,406.9 |                589.5 |               287.1 |
|    512 |            8,053.9 |            3,310.2 |              1,092.8 |               587.5 |
|  1,024 |           18,819.7 |            8,688.6 |              2,166.8 |             1,178.2 |
|  2,048 |           46,789.5 |           27,208.7 |              4,293.7 |             2,740.6 |
|  4,096 |          120,825.7 |           85,162.8 |              9,571.8 |             7,105.0 |

## Throughput — tokens/second (higher is better)

| Tokens | Pure fp32 | Pure Int8 | Python fp32 | ONNX Q4F16 |
|-------:|----------:|----------:|------------:|-----------:|
|    128 |        72 |       191 |         366 |      1,090 |
|    256 |        67 |       182 |         434 |        892 |
|    512 |        64 |       155 |         469 |        872 |
|  1,024 |        54 |       118 |         473 |        869 |
|  2,048 |        44 |        75 |         477 |        747 |
|  4,096 |        34 |        48 |         428 |        577 |

## How each scales (128 → 4096, i.e. 32× more tokens)

| | growth | note |
|---|---:|---|
| Pure fp32 | **68×** | super-linear: O(n²) attention term grows in |
| Pure Int8 | **127×** | *worse* ratio — Int8 shrinks only the linear projections, so the unquantized fp32 attention dominates faster |
| ONNX Q4F16 | **61×** | super-linear |
| Python fp32 | **27×** | ~300 ms fixed per-call floor dilutes the ratio |

Int8 is 2.6× faster than fp32 at 128 tokens but only **1.4× faster at 4096** — direct evidence that
quantization helps the projections but **not attention**, which is fp32 in both.

## Where the time goes — per-stage profile of the pure fp32 forward

Captured with the opt-in `ForwardProfile` (`harrier-profile <maxDop>`). Stopwatch/lock overhead inflates
the absolute totals ~20–40 % vs the clean benchmark, but the proportions and the DOP speed-ups are
accurate.

**Single-threaded (MaxDop=1):**

| Stage | 512 tok | 2048 tok | 4096 tok |
|---|--:|--:|--:|
| projections — mlp (gate/up/down) | 60.9% | 49.9% | 41.3% |
| projections — qkv | 14.8% | 12.0% | 8.8% |
| projections — o | 10.1% | 7.8% | 6.4% |
| **attention (Q·Kᵀ, softmax, ·V)** | **8.5%** | **26.3%** | **40.5%** |
| geglu | 4.9% | 3.5% | 2.6% |
| norms + residual + rope | 0.8% | 0.6% | 0.4% |
| total (ms) | 11,189 | 62,515 | 167,661 |

**Multi-threaded (MaxDop=4) total ms:** 512 → 3,532 (**3.2×**), 2048 → 17,530 (**3.6×**), 4096 → 45,417 (**3.7×**).

Takeaways from the profile:
1. At short/medium lengths the bottleneck is the **fp32 projection matmuls (~85 % at 512)**, *not*
   attention. Attention only becomes co-dominant at 4096 (40 %). This is why Int8 (which has tiled
   VNNI kernels for the projections) gives a big win at short lengths.
2. The pure path leaves **3.2–3.7× on the table** by defaulting to single-threaded in the document
   `EncodeAsync` path.
3. `geglu`, `rmsnorm`, `rope`, `headnorm` are **serial** loops (not parallelized), so their relative
   share grows under MaxDop=4 (geglu 4.9 % → 15.9 % at 512).

## Why pure is slower than PyTorch/ONNX — architectural review

- **Attention is BLAS-1, not BLAS-3.** `Gemma3Model.AttentionRow` computes scores as `seq²·heads`
  individual `TensorPrimitives.Dot` calls and the context as `seq²·heads` `MultiplyAdd` calls. The HF
  reference (`eager_attention_forward`) computes `attn = matmul(Q, Kᵀ) * scale` and `matmul(softmax, V)`
  as two **batched GEMMs** (or fused `scaled_dot_product_attention`) — cache-blocked, multi-threaded
  oneDNN/MKL, with K/V reused across all query rows instead of re-streamed per pair.
- **Projections are GEMV-per-row, not GEMM.** `Ops.LinearColumn` (fp32) does one `Dot` per
  `(output, seq)` with no register/cache blocking — PyTorch uses a tuned BLAS-3 GEMM. (The **Int8**
  path *does* register-tile, which is exactly why Int8 beats fp32 on projection-bound lengths.)
- **Threading default.** `-1` (the .NET "unbounded" sentinel) is treated as single-threaded by
  `ParallelExecution.ForAsync`; the query path passes `ProcessorCount` explicitly, the document path
  does not.

### int8-specific differences (vs the PyTorch ecosystem / ONNX int8)

- **Scope:** only the 7 linear projections are int8; `Q·Kᵀ`/softmax/`·V` stay fp32. So Int8 cannot
  touch the O(n²) term — its speed-up vanishes as sequences grow.
- **Dynamic per-row activation quant, recomputed & redundant:** `q/k/v` each re-quantize the *same*
  `normed` activation (3×); `gate/up` re-quantize their shared input (2×) — 5 passes/layer for 2
  distinct activations.
- **Scheme:** per-output-channel symmetric weights (no zero-point) + per-row symmetric activations
  (+128 uint8 offset for `vpdpbusd`, corrected by `rowSum`). **No outlier handling** (cf. LLM.int8()'s
  fp16 outlier decomposition).
- **Kernel:** register-tiled (4 out × 2–4 seq, VNNI `vpdpbusd`) but still hand-rolled per-tile, not a
  tuned BLAS-3 library GEMM. Non-VNNI CPUs fall back to weight-only (dequant to fp32).
- **Apples-to-apples int8** would compare against the ONNX `model_quantized.onnx` graph and/or a
  `torch.ao` dynamically-quantized PyTorch model — not the fp32 Python reference.

## Practical takeaways

- The pure package's value is **zero native dependencies / AOT & WASM portability**, not peak CPU
  throughput. For server-side throughput, ONNX (or PyTorch) is materially faster.
- Two cheap wins for the pure path: **(1) pass `MaxDegreeOfParallelism = ProcessorCount`** to the
  encode call (3–4×), and **(2) prefer shorter chunks** — every path pays a super-linear penalty per
  chunk, and the pure path most of all.
- Bigger structural win (not done here): replace the per-pair attention and per-row projection loops
  with blocked GEMM kernels (and quantize the attention matmuls) to attack both the constant factor and
  the quadratic term.

_Reproduce: `dotnet run -c Release --project SentenceTransformers.Benchmark -- harrier-scaling`,
`-- harrier-scaling-pure Int8`, `-- harrier-profile 1` / `-- harrier-profile 4`, then
`python scripts/harrier_scaling_bench.py`._
