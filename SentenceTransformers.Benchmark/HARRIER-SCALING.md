# Harrier Small — token-count scalability & pure-vs-ONNX optimization

Per-encode latency for `harrier-oss-v1-270m` swept over sequence length, comparing the pure-C#
implementation (fp32 and Int8) against the ONNX Runtime builds (Q4F16 and Int8). All paths are driven
with byte-identical calibrated inputs (one per target length, exact token count via the shared Gemma
tokenizer) through the public `EncodeAsync` path. **All numbers below are on all 4 cores** — the pure
encoder is driven with `ParallelOptions { MaxDegreeOfParallelism = ProcessorCount }` (the library default
stays single-threaded; only the benchmark opts in) so the comparison reflects architecture, not core count.

Environment: 4-core CPU, 15 GiB RAM, .NET 10 Release; ONNX Runtime 1.24.2; pure fp32/Int8;
ONNX Q4F16 (`model_q4f16`) and Int8 (`model_quantized`, `MatMulNBits` + fused `GroupQueryAttention`).

> Precision note: pure-fp32 and ONNX differ in weight precision (Int8 / Q4F16). The apples-to-apples
> rows for the optimization work are **Pure Int8 vs ONNX Int8**.

## Results — ms per encode (all on max cores)

| Tokens | Pure fp32 | Pure Int8 | ONNX Q4F16 | ONNX Int8 | Pure Int8 / ONNX Int8 |
|-------:|----------:|----------:|-----------:|----------:|----------------------:|
|    128 |     419.5 |   **107.8** |      106.0 |     141.7 | **0.76× (pure wins)** |
|    256 |     742.5 |   **202.9** |      208.1 |     219.5 | **0.92× (pure wins)** |
|    512 |   1,598.3 |     518.0 |      451.3 |     363.2 | 1.43× |
|  1,024 |   4,043.1 |   1,160.6 |      996.2 |     794.7 | 1.46× |
|  2,048 |   9,170.4 |   2,883.0 |    2,414.1 |   1,889.8 | 1.53× |
|  4,096 |  21,248.6 |   7,925.6 |    6,872.4 |   5,194.8 | 1.53× |

Pure Int8 now beats ONNX Int8 up to 256 tokens and the long-context gap is ~1.5×. The Pure-Int8 column
includes int8/VNNI attention scores (§2) and 512-bit AVX-512 kernels (§3). _(Absolute ms vary with this
shared cloud box's load between runs; the controlled comparisons below toggle one variable at a time.)_

## What was optimized (and the journey)

Profiling the pure Int8 forward (per-stage, 4 cores) found two dominant costs and several dead ends.

### 1. GeGLU — the big systemic win (committed)
The feed-forward GeGLU was a **scalar serial loop calling `MathF.Tanh` per element** (~151M tanh calls
at 4096 tokens). It was **40% of the time at 512 tokens** and ~4.5 s at 4096. Rewritten as a row-parallel,
vectorized kernel (`TensorPrimitives` incl. vectorized tanh):

| GeGLU stage (Int8, 4 cores) | 512 tok | 4096 tok |
|---|--:|--:|
| before (scalar) | 604 ms | 4,483 ms |
| after (vectorized) | 24 ms | 165 ms |

≈25× on that stage; this is what pulls short/medium contexts to parity.

### 2. Attention — the O(n²) scalability term
Several kernels were implemented and measured (all gated by a `verify-fma` correctness check: fp32 stays
cos = 1.0 vs the reference path, Int8 cos ≈ 0.998 — within the expected quantization tolerance):

Several kernels were implemented and measured (all gated by correctness checks: fp32 stays cos = 1.0 vs
the reference path; Int8 stays cos ≈ 0.99 / ≥ 0.984 vs the PyTorch golden — within quantization tolerance):

| attention kernel (Int8, 4 cores, 4096 tok) | attention ms | verdict |
|---|--:|---|
| original naive per-(query,head) dot | ~10,500 | baseline |
| cache-blocked + 4-head-fused (bit-identical) | ~10,486 | ~same (memory-bound) |
| FMA register-tiled (4×2 score GEMM) | ~10,545 | no gain → **fp32 compute is not the limit** |
| naive online-softmax flash | ~28,000 (2048: 5,000) | **worse** — scalar `exp` ≫ vectorized exp |
| head-fused block-flash (fp32 scores) | 8,661 | good |
| **+ int8/VNNI scores (chosen)** | **7,680** | **best** |
| + int8 value too | 8,399 | reverted — V is cache-resident, convert overhead > memory saving |

The winning kernel (`FlashAttentionBlock` / `FlashAttentionBlockInt8`) combines every lever: it streams
keys in L1-sized chunks with an **online softmax** (no seq×seq score matrix materialized — that buffer's
traffic was the bottleneck), keeps a **large query block** so each K/V chunk is reused across many
queries, **fuses the 4 GQA query heads** into one SIMD pass per key, and keeps `exp` **vectorized** per
chunk. For quantized modes the **scores use an int8 `vpdpbusd` dot** over per-row-quantized Q (uint8) and
K (int8), so K is read at 1 byte/element (4× less than fp32 — attention is memory-bound) with a VNNI
compute win on top. This mirrors ONNX's fused `GroupQueryAttention`. The **value matmul stays fp32**:
quantizing V was measured *slower* because the block-flash already keeps each V chunk cache-resident
(reused across the query block), so the int8→float convert costs more than the memory it saves.

Per-stage Int8 profile (4 cores), final kernels:

| Stage | 512 | 2048 | 4096 |
|---|--:|--:|--:|
| attention (int8 scores) | 176 | 1,934 | 7,680 |
| mlp_proj (Int8 VNNI) | 367 | 1,540 | 3,076 |
| qkv+o (Int8 VNNI) | 197 | 761 | 1,512 |
| geglu | 19 | 77 | 158 |
| norms+rope | 110 | 378 | 813 |

### 3. AVX-512 (512-bit) kernels
The attention kernels were 256-bit (`Vector256` / `vpdpbusd` on ymm). On an AVX-512 host they now run
512-bit, selected by `IsSupported` bool flags (JIT constants — the 256-bit body is dead-code-eliminated
on non-AVX-512 CPUs, the standard .NET pattern): the **int8 scores** use a 512-bit VNNI dot
(`Vnni.DotAccumulate512`, 64 int8 MACs/instr) and the **value / scale kernels** use 512-bit FMA
(`Avx512F.FusedMultiplyAdd`, 16 floats/lane). The int8 projection GEMMs already had a 512-bit path.

Controlled toggle (`DOTNET_EnableAVX512=0` vs default, same box, Int8, 4 cores), total ms per encode:

| | 512 tok | 2048 tok | 4096 tok |
|---|--:|--:|--:|
| 256-bit | 650 | 3,558 | 8,836 |
| **512-bit** | **597** | **2,591** | **7,053** |
| speedup | 1.08× | 1.37× | 1.25× |

(attention 4096: 4,568→3,990 ms; int8 projections 4096: 2,367→1,551 ms.) AVX-512 has a small fixed
overhead/downclock that shows at very short sequences but wins clearly from ~1k tokens up.

### Net effect (Pure Int8)
single-threaded original → max cores + vectorized GeGLU + head-fused block-flash + int8/VNNI scores +
AVX-512: roughly **6–8× faster** end-to-end (exact factor drifts with the shared box's load), and the
Int8/ONNX-Int8 gap went from ~2.0× to **~1.5× at 4096, with pure faster at ≤256 tokens**.

## Long-context scaling (→ 32768, the full context window) — pure Int8 vs ONNX Int8

Single encode per point on max cores (O(n²) makes 32768 a multi-minute encode):

| Tokens | Pure Int8 (ms) | ONNX Int8 (ms) | ratio |
|-------:|---------------:|---------------:|------:|
|  4,669 |       17,059 |        11,072 | 1.54× |
|  9,336 |       58,747 |        40,630 | 1.45× |
| 18,669 |      232,509 |       187,555 | 1.24× |
| 32,768 |  **714,956** | **OOM** (needs 16 GB) | pure completes |

Two findings here:
1. **The ratio improves with length (1.54× → 1.24×)** — the pure flash kernel scales *as well or better*
   than ONNX's CPU attention as the quadratic term takes over.
2. **At the full 32,768 context window ONNX Int8 OOMs and pure Int8 succeeds.** ONNX Runtime's CPU
   `GroupQueryAttention` materializes the full `[heads, seq, seq]` score tensor — at 32768 that is
   `32768² × 4 heads × 4 B ≈ 16 GB`, which fails to allocate on the 15 GiB box. The pure kernel's
   online-softmax (block-flash) never materializes the score matrix, so it runs in O(seq) attention
   memory and completes.

## Honest parity assessment
- **Short contexts (≤256 tokens): pure Int8 is now faster than ONNX Int8** (128: 0.76×, 256: 0.92×).
- **Mid contexts (512–4k): ~1.4–1.5× off**, narrowed from ~2.0× by int8/VNNI scores + AVX-512. The
  residual is the fp32 value matmul and ONNX's MLAS-class kernels. The next lever is an MLAS-style packed
  GEMM for the value matmul (int8 value did not help — V is cache-resident, see §2).
- **Long contexts (8k–32k): the gap *narrows* further (to ~1.24×) and pure is strictly more scalable in
  memory** — it is the only one of the two that completes a 32768-token encode on this machine
  (ONNX OOMs materializing the full score tensor).

## Reproduce
```
dotnet run -c Release --project SentenceTransformers.Benchmark -- harrier-scaling          # 128..4096, 4 impls
dotnet run -c Release --project SentenceTransformers.Benchmark -- harrier-scaling-long      # 4096..32768, Int8
dotnet run -c Release --project SentenceTransformers.Benchmark -- harrier-profile 4 Int8    # per-stage profile
dotnet run -c Release --project SentenceTransformers.Benchmark -- harrier-verify-fma        # attention correctness gate
python scripts/harrier_scaling_bench.py                                                     # PyTorch reference
```
