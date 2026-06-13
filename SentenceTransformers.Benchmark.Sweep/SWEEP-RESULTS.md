# Multi-model token-count sweep — results

Output of `SentenceTransformers.Benchmark.Sweep` (each model run in its own process, all cores). 4-core
AVX-512 Xeon, 15 GiB RAM, .NET 10 Release, ONNX Runtime 1.24.2. Single encode per point; `ms` is the
median (more iterations at small token counts). Each model sweeps from 128 up to its own context window.
Token counts differ slightly across models because the tokenizers differ (WordPiece vs Gemma/Qwen BPE);
the actual measured count is shown.

## Per-model latency (ms)

| Model (max ctx) | ~136 | ~270 | ~530 | ~1064 | ~2125 | ~4250 | ~8488 | ~16973 | 32768 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| MiniLM-L6-v2 (256) | 19.5 | 24.5¹ | — | — | — | — | — | — | — |
| ArcticXs (512) | 14.1 | 33.2² | 34.9 | — | — | — | — | — | — |
| Qwen3-0.6B (32768) | 121.9 | 246.4 | 634.2 | 1,954 | 6,649 | 23,913 | **OOM-killed** | — | — |
| Harrier.Small.Pure fp32 (32768) | 402 | 756 | 1,546 | 3,762 | 8,459 | 17,020 | 44,283 | 133,400 | **411,162** |
| **Harrier.Small.Pure Int8** (32768) | 167 | 182 | 381 | 845 | 1,875 | 4,986 | **14,234** | **52,438** | **181,592** |
| Harrier.Small ONNX Q4F16 (32768) | 108 | 188 | 381 | 821 | 2,086 | 6,514 | 20,007 | 85,800 | **OOM** |
| Harrier.Small ONNX Int8 (32768) | 126 | 196 | 334 | 673 | 1,678 | 4,929 | 16,276 | 76,114 | **OOM** |
| Harrier.Medium-0.6B ONNX Q4F16 (32768) | 399 | 773 | 1,495 | 3,457 | 9,657 | 25,747 | 80,983³ | **OOM** | **OOM** |

¹ MiniLM 256 token point. ² ArcticXs ~415 token point. ³ Medium ~9365 token point. (Token counts are the
measured values; columns are grouped by nearest target.) "OOM" = ONNX `GroupQueryAttention` could not
allocate the full `[heads, seq, seq]` score tensor (e.g. 16 GB at 32768, 64 GB for Medium); "OOM-killed" =
the process was SIGKILLed by the OS (per-model process isolation kept this from taking down the rest).

## Headline findings

- **Harrier.Small.Pure Int8 is competitive with — and at long contexts faster than — the ONNX builds**,
  and is the **only Harrier build that completes the full 32768-token window** on a 15 GiB box:
  - 8,488 tok: **Pure Int8 14.2 s** vs ONNX Int8 16.3 s vs ONNX Q4F16 20.0 s.
  - 16,973 tok: **Pure Int8 52.4 s** vs ONNX Int8 76.1 s vs ONNX Q4F16 85.8 s.
  - 32,768 tok: **Pure Int8 181.6 s** vs ONNX Int8 / Q4F16 **OOM**.
  This is the payoff of the block-flash attention (O(seq) attention memory + cache-friendly chunks)
  plus the register-blocked value GEMM. The pure encoder's online softmax never materializes the score
  matrix; ONNX Runtime's CPU `GroupQueryAttention` does, so it OOMs past ~16k tokens here.
- At short/mid contexts the four Harrier Small variants are close (within ~1.1–1.5×); Pure Int8 is fastest
  at 8k+ and ties/leads ONNX Int8 around 4k.
- **MiniLM** and **ArcticXs** are tiny, fast BERT models (≤512 ctx, ~15–35 ms) — different model class,
  included for completeness.
- **0.6B models** (Qwen3, Harrier Medium) are ~3–5× slower per token than Harrier Small and hit the
  memory wall earlier (Qwen3 OOM-killed at 8k; Medium OOMs at ~18k).

_Reproduce: `dotnet run -c Release --project SentenceTransformers.Benchmark.Sweep` (or `-- <substring>` to
pick one model; run memory-heavy 0.6B models in their own process to isolate OS OOM-kills)._
