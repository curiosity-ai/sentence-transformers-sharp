# Quantized weights: the `.stq` format, the converter, and the runner

This document covers what PrismML's Bonsai models do to reach ~1.7 bits per weight, the format and
runtime this repository grew to do the same for Harrier Small Pure, and what the measurements
actually support.

**The short version.** The ternary machinery works and is verified, but round-to-nearest ternary is
only usable for one tensor in this model, so the container also carries a 4-bit band and the default
conversion is 4-bit throughout. That default is a **strict improvement on every mode that ships
today** — fp32's STS-B score (0.8177) in 388 MB resident against `Int8`'s 1046, from a file 3.87×
smaller, and **1.11× faster than `Int8` single-threaded, 1.41× on four threads**. §4 has the numbers,
including how the kernel got there and the plausible argument that said it could not; §5 has what
ternary needs to become viable.

## 1. What Bonsai does

Every Bonsai band stores the same thing: each weight is one of `{-1, 0, +1}`, multiplied by an FP16
scale shared across a group of neighbouring weights.

```
w_i = s_g · t_i        t_i ∈ {-1, 0, +1}
```

A trit carries `log2(3) ≈ 1.585` bits, so with one FP16 scale per 128 weights the floor is
`1.585 + 16/128 = 1.71` bits/weight — about 9× smaller than FP16. The three shipped packings differ
only in how the trits are laid out in bytes:

| Bonsai band | group | layout | bytes/group | bits/weight |
|---|---|---|---|---|
| `Q2_0` (mainline llama.cpp) | 64 | 2-bit codes + FP16 scale | 16 + 2 = 18 | 2.25 |
| `PQ2_0` | 128 | 2-bit codes + FP16 scale | 32 + 2 = 34 | 2.125 |
| `PTQ1_0` | 128 | base-3, 5 trits/byte + FP16 scale | 26 + 2 = 28 | 1.75 |

`PTQ1_0`'s trick is that `3^5 = 243 < 256`, so five trits fit in one byte with room to spare. That is
where the whitepaper's "1.76 bits/weight, approaching the 1.72 limit" comes from, and why `PTQ1_0`
moves 18% less weight data than `PQ2_0` while decoding to identical values.

Two further pieces matter as much as the packing:

- **A fixed rotated basis.** Bonsai 2 stores `W` transformed by `R = H_n·S/√n` — a Walsh–Hadamard
  matrix times a fixed diagonal of ±1 signs, blockwise with `n = 1024`. Inference computes
  `f(x) = W(Rx)`. This is an exact identity (`R` is orthogonal), and it is what makes ternary
  survive: a Hadamard block mixes every weight in it, so per-group outliers — which otherwise
  inflate the scale and flatten everything else to zero — disappear into a near-Gaussian spread.
- **Selected tensors kept in full precision.** Bonsai 2 leaves norms and the recurrent state path
  unquantized: 0.0976% of parameters, moving the model from 1.71 to 1.72 bits/weight.

The one thing the whitepapers do *not* claim is that this is a post-training conversion. Bonsai
models are built for ternary; §4 below is what happens when you only do the conversion.

## 2. The `.stq` container

`SentenceTransformers/src/Stq/` implements a model-agnostic container mirroring the above.

```
[0  ..  4)   magic "STQ1"
[4  ..  8)   uint32 header length, little-endian
[8  ..  8+n) UTF-8 JSON header
             pad to a 64-byte boundary
             data section: every blob, each 64-byte aligned
```

Byte ranges in the header are `[begin, end)` offsets into the data section, as in `safetensors`.
Codes and scales are **separate** blobs rather than interleaved blocks (GGUF's choice), so a kernel
can stream a row's scales contiguously.

```json
{
  "__metadata__":  { "format": "stq", "format_version": "2", "architecture": "gemma3-text", ... },
  "__rotations__": { "h640": { "dim": 640, "block": 128, "signs": [0, 80] } },

  "embed_tokens.weight": {
    "band": "tq1_0", "shape": [262144, 640], "group_size": 128,
    "rotation": "h640", "codes": [128, 34078848], "scales": [34078848, 36700288]
  },
  "layers.0.input_layernorm.weight": { "band": "f32", "shape": [640], "data": [...] }
}
```

Bands: `tq1_0` and `tq2_0` are the ternary ones (matching `PTQ1_0` / `PQ2_0`); `q4_0` is symmetric
4-bit with codes in `[-8, +7]`, at 4.5 bits/weight for a group of 32 or 4.125 for 128; `f32`, `f16`
and `bf16` store a tensor verbatim, so norm vectors live in the same file. The band is per tensor, so
one file can mix them — which is what the measurements below end up wanting.

`q4_0`'s two nibbles per byte are **half a group apart**, not adjacent: byte `j` holds code `j` low
and code `j + groupSize/2` high, as llama.cpp's `q4_0` does. Adjacent nibbles make a vector of bytes
decode into interleaved codes, which costs a duplicate-shuffle and a lane-wise merge per vector; half
a group apart it falls out as two contiguous vectors, so unpacking is a mask and a shift. That is
what `format_version` 2 means — a v1 file's ternary tensors still load, its 4-bit ones are rejected
with a message saying to re-run the converter.

`q4_0` is not a Bonsai band. It is here because ternary turned out to be viable for this model's
embedding table and not for its projections (§4), and everything else about the container — group
scales, the rotated basis, the kernel — applies to it unchanged, since 4-bit codes are int8 operands
just as trits are.

Design notes:

- **Sign diagonals are stored, not seeded.** A rotation's ±1 signs are written as a bitmask (80 bytes
  for a 640-wide rotation). A reader never has to reproduce a PRNG to load a file.
- **One rotation per input width**, shared by every tensor that consumes it (`h640`, `h1024`,
  `h2048` for Harrier Small). Beyond keeping the file small, this leaves room for a runtime that
  rotates the hidden state once and feeds q, k and v from it.
- **Rotation blocks adapt to the dimension.** The block is the largest power of two dividing the
  width, capped (default 1024). Harrier's 640-wide hidden state is `2^7 · 5`, so it rotates in blocks
  of 128; the 2048-wide MLP activations use two blocks of 1024.
- **Embeddings are rotated too.** The table is 167.8M of 268.1M parameters, so it is where rotation
  pays most (and rotation helps 4-bit as well as ternary: 0.086 against 0.091 relative error on the
  projections). A lookup has no activation to fold the rotation into, so the row is un-rotated after
  unpacking (`R^T`) — one Walsh–Hadamard transform per token, nothing next to the 18 layers after it.

The same routine rotates weight rows at conversion time and activations at inference time: storing
`W R^T` for a row `w` is `w S H/√n`, whose transpose is `(H/√n) S w^T` — the operation applied to
activations. One routine, two uses, so a basis mistake cannot silently cancel out.

## 3. Tooling and runtime

**Convert** (`SentenceTransformers.Quantize`):

```bash
# the default: 4-bit throughout, including the embedding table
dotnet run --project SentenceTransformers.Quantize -c Release -- \
  convert --input harrier-oss-v1-270m.safetensors --output harrier-small-q4.stq

# the smallest build: ternary embedding table, 4-bit projections
dotnet run --project SentenceTransformers.Quantize -c Release -- \
  convert --input harrier-oss-v1-270m.safetensors --output harrier-small-mixed.stq --embed-band tq1_0
```

Any rank-2 tensor whose input width is a multiple of its band's group size is quantized; everything
else (every rank-1 norm vector) is written as f32. For Harrier Small that is 268.04M parameters
quantized and 56K left alone — the same call Bonsai 2 makes, for the same reason. The token
embedding table gets its own band and group size (`--embed-band`, `--embed-group`), because it is
63% of the parameters and the only tensor whose choice moves the file size on its own.

Ternary group quantization defaults to `--method optimal`, which is exactly L2-optimal rather than a
heuristic: fixing the number of non-zero trits `k`, the best scale is the mean of the `k` largest
magnitudes and the residual is `Σw² − prefix(k)²/k`, so sorting `|w|` and scanning `k` visits every
candidate optimum. `absmean` (BitNet b1.58) and `twn` are available for comparison. The 4-bit band
has no comparably cheap exact optimum — the codes depend on the scale through a rounding — so it
searches a small grid of scales around round-to-nearest and keeps the best, which is free at
inference and cannot come out worse.

**Validate**:

```bash
dotnet run --project SentenceTransformers.Quantize -c Release -- \
  validate --ternary harrier-small-tq1_0.stq --original harrier-oss-v1-270m.safetensors
```

Four levels, cheapest first, so a structural bug is reported as a structural bug rather than
surfacing later as a mysterious quality drop:

1. **Codec (exact).** Every group of every tensor is unpacked and re-packed and must reproduce the
   file's bytes; every rotation must satisfy `RᵀR = I`.
2. **Kernel vs. reference decode (exact up to round-off).** `TernaryMatrix` — packed codes, group
   scales, activation rotation — is run against a `FloatMatrix` built from the same tensor decoded
   and un-rotated by a different route. This is the check that separates a wrong kernel from honest
   quantization loss; the two look identical from the outside.
3. **Per tensor.** Each tensor decoded and compared with the original checkpoint.
4. **End to end.** fp32 and quantized encoders embed the same sentences; reports per-sentence cosine
   and the Spearman correlation between the two pairwise-similarity matrices — the ranking property
   retrieval depends on, which a mean cosine can hide. The gates are a mean cosine of 0.98, a
   per-sentence floor of 0.90 and a ρ of 0.98; the default conversion passes all three, and every
   configuration §4 rejects fails at least one.

`compare` scores the ternary file and the existing Int8/Int4 modes against one fp32 baseline;
`inspect` dumps a file's header.

**Score it on a real task.** `validate` and `compare` measure agreement with the fp32 encoder, which
catches damage but says nothing about absolute quality. For that, the LoRA training CLI's `eval`
command takes a `--weights` path, so any checkpoint - an `.stq` file, or a safetensors checkpoint
quantized at load time - can be scored on the STS Benchmark test split:

```bash
cd SentenceTransformers.LoraTraining
dotnet run -c Release -- download                       # one-time, fetches the STS-B splits

dotnet run -c Release -- eval --model harrier-small --split test \
  --weights ../path/to/harrier-small-q4.stq             # an .stq file
dotnet run -c Release -- eval --model harrier-small --split test \
  --weights ../path/to/model.safetensors --quantization int4
```

It reports STS Spearman plus retrieval accuracy and MRR over 1379 annotated pairs - an absolute
number to judge a build by, rather than a similarity to the fp32 one.

**Runtime.** `SentenceEncoder.LoadQuantizedAsync(path)` / `CreateQuantizedAsync(url)` load an `.stq`
checkpoint. The forward pass is unchanged: packed weights sit behind the same `IWeightMatrix` that
`Int8Matrix`/`Int4Matrix` implement, and the embedding table behind a new `ITokenEmbedding`. A single
`StqMatrix` covers every packed band, because they differ only in how a group of codes unpacks to
signed bytes; after that the kernel is identical. On a VNNI host the codes feed `vpdpbusd` against
dynamically int8-quantized activations, and elsewhere each row dequantizes to float. Unlike
`LoadAsync`, there is no `quantization` argument: the precision was fixed when the file was written,
and loading does no quantization work.

`Int4Matrix` remains the path for weights quantized at load time from safetensors. The difference
that matters is not the arithmetic but *what gets quantized*: it only ever sees the projections,
whereas an `.stq` file also carries the embedding table.

## 4. What the measurements say

Ternary conversion works exactly as designed, on size. Converting every tensor of the released
`harrier-oss-v1-270m` checkpoint:

| | size | bits/weight |
|---|---|---|
| bf16 safetensors | 511.4 MB | 16 |
| `.stq` all-`TQ2_0` | 68.1 MB | 2.131 |
| `.stq` all-`TQ1_0` | 56.2 MB | **1.756** — 9.11× smaller |

Each ternary design choice is measurably doing its job (worst per-tensor relative error):

| variant | worst relative error |
|---|---|
| rotated, optimal (default) | **0.4341** |
| unrotated, optimal | 0.5315 |
| rotated, `absmean` (BitNet) | 0.5125 |
| rotated, `twn` | 0.4368 |

`TQ1_0` and `TQ2_0` decode to identical values — they differ only in packing — so `TQ1_0`'s 17%
smaller file is free.

### Ternary projections do not survive post-training quantization

Against the fp32 baseline over 20 multilingual sentences (`compare`):

| configuration | file | mean cosine | min cosine | ranking ρ |
|---|---|---|---|---|
| Int8, load-time (embeddings stay bf16) | ~440 MB resident | 0.99565 | 0.98528 | 0.99567 |
| Int4, load-time (embeddings stay bf16) | ~410 MB resident | 0.97622 | 0.95592 | 0.97918 |
| **`.stq` all-ternary `TQ1_0`** | 56.2 MB | **0.57374** | 0.43772 | 0.45249 |

On the STS Benchmark that works out to 0.6759 Spearman against fp32's 0.8177 - a 17% relative loss.

That is not a bug, and the validator's level-2 check is what establishes it: the kernel reproduces
the reference decode to 7e-3 (the int8 activation path's own noise) and the embedding lookup is
bit-exact. The loss is entirely in the weights.

Nor is it fixable by tuning the converter. The measured 0.432–0.434 relative error per tensor is
already the theoretical floor: the optimal 3-level quantizer for a Gaussian has an MSE of 0.1902σ²,
i.e. a relative error of 0.436, and the rotation makes each group very nearly Gaussian by
construction. We land marginally below 0.436 only because each group of 128 gets its own scale.
Halving the ternary group size confirms it — the end-to-end cosine moves by 0.00007. **The limit is
three levels, not the scale granularity**, so no finer group, better search or different packing
moves it.

That is consistent with what PrismML publish: Bonsai models are *trained* ternary, in a fixed rotated
basis, not converted after the fact. The format, the packings and the rotation transfer; the free
lunch does not.

### The embedding table is the exception — and it is 63% of the model

Splitting the conversion says exactly where the damage is:

| what is ternary | mean cosine | min cosine | ranking ρ |
|---|---|---|---|
| everything | 0.57374 | 0.43772 | 0.45249 |
| projections only (embeddings f32) | 0.61744 | 0.48183 | 0.54555 |
| **embedding table only** (projections f32) | **0.96449** | 0.92601 | 0.97446 |

Ternarizing the 262144 × 640 table costs little; ternarizing eighteen layers of projections is what
breaks the model. This matters because **the `Int8`/`Int4` modes do not touch the embedding table at
all** — `Gemma3Model` keeps it in bfloat16 either way — so of the ~410 MB an `Int4` encoder holds
resident, about 335 MB is a table no quantization mode previously compressed.

### The resulting size/quality curve

Two things are measured here, and they disagree in an instructive way. **Cosine vs fp32** is what
`compare` reports: how far each embedding moved. **STS Spearman** is what `eval` reports on the STS
Benchmark test split (1379 annotated pairs): how well the embeddings still do the job. Projections
are `q4_0` at group 32 throughout; only the embedding table changes.

| configuration | size | cosine vs fp32 | **STS Spearman** | retrieval acc | MRR |
|---|---|---|---|---|---|
| fp32 (baseline) | 511.4 MB | — | 0.8177 | 0.8707 | 0.9198 |
| `Int8`, load-time (embeddings bf16) | ~440 MB resident | 0.9957 | 0.8179 | 0.8752 | 0.9218 |
| `Int4`, load-time (embeddings bf16) | ~410 MB resident | 0.9762 | 0.8144 | 0.8752 | 0.9220 |
| **`.stq` `q4_0` everywhere (default)** | **136.5 MB** | 0.9843 | **0.8184** | 0.8737 | 0.9205 |
| **`.stq` `tq1_0` embedding table** | **89.0 MB** | 0.9506 | **0.8088** | 0.8588 | 0.9120 |
| `.stq` all-ternary `tq1_0` | 56.2 MB | 0.5737 | 0.6759 | 0.8187 | 0.8809 |

**Cosine systematically overstates the damage.** The default conversion sits 0.0157 below fp32 in
cosine and is level with it on the task (the +0.0007 is noise at this sample size). The ternary
embedding table looks like a real step down at 0.951 cosine, and costs 0.009 Spearman - roughly 1%
relative. The embeddings move; the rankings they produce largely do not, and rankings are what a
retrieval system consumes. Treat cosine as a cheap offline tripwire, not the shipping criterion -
which is why the default validation gates deliberately fail the 89 MB build even though STS says it
is fine. When the decision matters, run `eval`.

That said, cosine is not crying wolf at the bottom of the table: all-ternary really is broken, losing
0.142 Spearman (17% relative). Note it is *degraded*, not random - 0.676 is well above chance, so the
model retains real structure. It is simply not a model anyone should ship.

Three conclusions, and they set the defaults:

- **4-bit throughout matches fp32 on the task at 136.5 MB** - against roughly 410 MB resident for
  `Int4`, which scores slightly lower. This is the default conversion. The whole gain comes from
  being the first path here that compresses the embedding table.
- **A ternary embedding table costs about 1% relative for another 35% off the file**, at 89.0 MB and
  0.8088 Spearman - between `Int4` and fp32 in quality at roughly a fifth of `Int4`'s memory. One
  flag away (`--embed-band tq1_0`); it needs `--min-mean-cosine` lowered because the cosine gate, not
  the model, objects.
- **Ternary projections stay off the table** until weights are trained for them.

Group size barely matters for the embedding table in either band (0.00006 of cosine between `q4_0`
at 32 and at 128), so the default takes the coarser, smaller one.

### Speed

Encode throughput matters as much as size, and the packed path started out badly: it was the
*slowest* way to run the model per embedding, which made the winning quantization the one nobody
would want to serve. It is now the fastest. Everything below is measured over 192 distinct sentences
per iteration on a 4-core host, after three full warm-up passes, with every mode timed in the same
process so a VM or frequency change cannot land between the baseline and the candidate
(`harrier-pure-bench`); the figures are medians of three runs.

Two thread counts matter, and they say different things, so both are reported. Single thread is how
the library is meant to be consumed here — callers parallelize across documents, not inside one.

| mode | 1 thread ms/iter | 4 threads ms/iter | resident |
|---|---|---|---|
| fp32 | 15938 | — | 1332 MB |
| `Int8`, load-time | 4107 | 3375 | 1046 / 540 MB |
| `Int4`, load-time | 13545 | — | 519 MB |
| `.stq` q4 g128 — first working version | 11538 | 5068 | 388 MB |
| **`.stq` q4 g128 — now** | **3704** | **2387** | **388 MB** |

**3.1× faster than where it started, and 1.11× faster than `Int8` on one thread, 1.41× on four** — in
388 MB against 1046, from a file 3.87× smaller, at 0.8177 STS-B Spearman, which is fp32's number to
four decimal places.

The order things were tried in is the order of intuition; the order of payoff was different.

| change | what it was worth |
|---|---|
| vectorizing the 4-bit nibble unpack | ~1% |
| widening the scale groups, 32 → 128 | ~7% (67% once the chains below were covered) |
| **tiling the kernel** | **1.63×, then a further 1.20×** |
| 4-way `vphaddd` group reduction + group-major scales | 1.25× |
| **vectorizing activation quantization** | **1531 → 84 ms/iter** |
| sharing one activation quantization across q/k/v and gate/up | (part of the above) |
| vectorizing the Walsh–Hadamard stages narrower than a vector | 775 → 221 ms/iter |
| row-level 4-bit unpack + the split-nibble layout | 1124 → 607 ms/iter |
| folding the zero-point correction into a per-channel bias, the group scale-accumulate into one FMA | ~3% |
| `SkipLocalsInit` on the tile kernels | a memset of the weight matrix per matmul |
| **blocking the weights so the accumulator lanes are output channels** | **the dot: 3541 → 2017 ms/iter** |

Four of those deserve a note, because the mistake each corrects is easy to repeat.

**The kernel computed one output channel at a time**, so `acc = DotAccumulate(acc, ...)` formed a
single serial dependency chain and stalled on its own ~5-cycle latency no matter how cheap the unpack
was. That is why vectorizing the unpack first was worth 1%. Four channels by four positions gives
sixteen independent chains, and on an AVX-512 host the JIT has 32 vector registers to hold them.

**Activation quantization was scalar**, and at 22% of a packed forward pass it was larger than the
nibble unpacking the whole exercise was about. It is shared with the `Int8` path, so fixing it moved
that baseline too — from 5497 to ~4100 ms/iter. A comparison is only worth making once both sides have
had the obvious work done to them.

**The Walsh–Hadamard stages with `len < 8`** pair elements inside one aligned group of eight, so the
general loop was building a pair of spans to add and subtract a single float — 64 times per
128-element block. Those three stages were three quarters of the rotation's cost and all of them fit
in one register.

**The weight layout decided whether there was a horizontal reduction at all**, and that turned out to
be the largest single item — see below.

Two notes on measuring this. `EncodeAsync` memoizes the last 16 vectors by input hash, so a benchmark
that re-encodes one small batch in a loop times a dictionary lookup rather than inference — the
corpus here is deliberately far larger than that cache. And run-to-run spread on a shared 4-core box
is around 10%, so only differences well beyond that are worth reading; that is why each run prints
the host's SIMD capabilities and times both modes back to back.

### How the packed path overtook `Int8`, and the argument that said it could not

It is worth writing down the wrong reasoning as well as the right answer, because the wrong reasoning
was specific, quantitative, and checked against the profile — and still wrong.

**The argument that it could not.** Both kernels feed the same instruction. `vpdpbusd` multiplies
unsigned bytes by signed bytes, 32 MACs per 256-bit operation, and .NET 10 exposes no AVX-512 VNNI as
a standalone ISA, so neither path can issue wider or fewer of them. Four-bit weights cannot be fed to
it packed either: a byte holding two codes contributes `lo + 16·hi` to one scalar sum, and no shuffle
recovers the two products from that. So the packed path must do **all** of `Int8`'s
multiply-accumulate work, and then unpack the codes (687 ms/iter) and rotate the activations (247
ms/iter) on top — 934 ms against a 4098 ms baseline, a 23% floor. The measured gap at the time was
1039 ms, so the two extras accounted for 90% of it and the arithmetic looked closed.

**What it missed.** Every step of that is true about the *multiply*. None of it is about the
*reduction*, and the reduction was not a fixed cost — it was a consequence of the operand layout.

`vpdpbusd` sums four byte products into each of its eight int32 lanes. Feed it a row of weights and
those eight lanes hold eight partial sums **of the same output channel**, so finishing a dot product
means adding the lanes together. `Int8` pays that once per output channel per row. The packed path
has a scale per 128 weights, so it paid it *every group* — three `vphaddd`s per four channels every
128 weights, and `vphaddd` is three micro-operations. That was about 30% of the kernel and it is
exactly the "per-group reduce-and-rescale" the earlier notes kept attributing to the format, as if
per-group scales necessarily implied a horizontal reduction. They do not.

Feed the same instruction a vector whose lane `L` holds four consecutive weights of output channel
`L`, against an activation vector that is those same four activations broadcast into all eight lanes,
and each lane accumulates a *different* output channel. The dot product is finished when the loop
ends. A scale group then costs one convert and one multiply-add per eight channels — no reduction at
all — and the per-group scales become free.

The weights have to be in that order, so `StqMatrix` rewrites them once at load (`BuildBlocked`): for
one tile of 32 channels and one scale group, the code order is
`q·128 + b·32 + L·4 + m` — input quad, then 8-channel block, then channel, then input within the quad.
Quad-major so the four weight vectors a step needs are 128 contiguous bytes; whole-tile groups so a
tile's codes for a group unpack in one call. It costs about 0.1s of load time and nothing at all on
disk — the file format is untouched, and the float fallback keeps the file's own row-major order.

That took the dot from 3541 to 2017 ms/iter, which is more than the unpacking and the rotation
together, so the packed path came out ahead:

| 1 thread, ms/iter | before blocking | after |
|---|---|---|
| dot | 3541 | **2017** |
| unpack 4-bit codes | 687 | 607 |
| rotate activations | 247 | 221 |
| quantize activations | 84 | 84 |
| **total** | **4981** (1.23× slower than `Int8`) | **3704** (1.11× *faster*) |

The lesson generalizes past this kernel: with a dot-product instruction that reduces within a lane,
what the lanes *mean* is a free variable, and choosing it badly can cost more than the arithmetic the
instruction is doing. "Both paths issue the same multiplies" was true and irrelevant.

What is still real from the old argument: the unpack and the rotation are genuine extra work the
packed path does and `Int8` does not, and neither is removable. Rotation was measured, not assumed —
converting with `--rotate embed` (rotate the embedding table, leave the projections in the original
basis, so no projection needs a rotated activation) drops mean embedding cosine from 0.980 to 0.958.
The advantage also grows with thread count, because half the weight bytes means half the weight
traffic: 1.11× on one thread, 1.41× on four.

### Keeping it honest in CI

`SentenceTransformers.Tests` carries an opt-in `StqStsBenchmarkTests` that runs the real converter,
loads the result through `LoadQuantizedAsync`, and scores it against the same checkpoint in fp32 on
STS-B. It asserts the default conversion stays within 0.02 Spearman of fp32, and that whole-model
ternarization still collapses - so the finding above cannot be quietly undone by a change that only
looks good on file size. It needs the checkpoint, so it is opt-in:

```bash
HARRIER_STQ_STSB=/path/to/harrier-oss-v1-270m.safetensors dotnet test
```

`HARRIER_STQ_STSB_PAIRS` caps the pair count (default 250, for a couple of minutes per configuration;
the full split is 1379).

## 5. Regenerating weights

The container and runner are ready for weights produced *for* ternary rather than squeezed into it.
Two paths work:

- **Trained ternary in the original basis** (weights already `{-s, 0, +s}` per group). Convert with
  `--band tq1_0 --embed-band tq1_0 --no-rotate`: the conversion is then *lossless*, and there is a
  test asserting exactly that (`AlreadyTernaryWeightsSurviveAConversionUnchanged`). Rotating such a
  checkpoint would destroy the structure that makes it lossless, so `--no-rotate` is required, not
  optional.
- **Trained ternary in a rotated basis** (Bonsai's approach). The training pipeline's sign diagonals
  must be the ones written into the file. Today the converter generates its own from `--seed`, so
  this needs the sign masks to be importable — a small addition to `StqWriter.AddRotation`'s wiring,
  flagged here rather than guessed at, since it depends on how the pipeline emits them.

Either way `validate` scores the result against the fp32 reference before it is published, and fails
the run rather than blessing a conversion like the all-ternary one above.

## Summary

- The format, both Bonsai packings, the rotation, the converter, the validator and the runner are
  implemented and verified. The kernel is proven equivalent to the reference decode, so the tooling
  is ready for real ternary weights.
- Ternarizing the whole model post-training does not work (0.676 STS Spearman against fp32's 0.818),
  for a reason that is a property of three-level quantization rather than of this implementation.
  Weights trained for ternary are required - which is how Bonsai does it.
- The embedding table is the exception and ternarizes usefully: 335 MB to 36.7 MB for about 1%
  relative on the task.
- The immediate win is unrelated to bit width: quantizing the embedding table at all, which no
  existing mode does. The default 4-bit conversion matches fp32 on STS-B at 136.5 MB, where the
  `Int4` mode that ships today scores slightly lower at roughly three times the memory.
- Judge builds with `eval` on a real task, not with cosine against fp32 - cosine consistently
  overstates how much a conversion costs.
- On speed the packed path went from 2.13x slower than `Int8` to **1.11x faster** single-threaded
  (1.41x on four threads) at 388 MB against 1046. The decisive change was not the packing or the
  rotation but the weight layout: blocking the weights so a `vpdpbusd` accumulator's lanes are eight
  different output channels removes the per-scale-group horizontal reduction entirely. See "How the
  packed path overtook `Int8`", which also records the careful argument that said this was
  impossible.
