# Quantized weights: the `.stq` format, the converter, and the runner

This document covers what PrismML's Bonsai models do to reach ~1.7 bits per weight, the format and
runtime this repository grew to do the same for Harrier Small Pure, and what the measurements
actually support.

**The short version.** The ternary machinery works and is verified, but round-to-nearest ternary is
only usable for one tensor in this model, so the container also carries a 4-bit band and the default
conversion is 4-bit throughout. That default is a **strict improvement on every mode that ships
today** — fp32's STS-B score (0.8177) in 388 MB resident against `Int8`'s 1046, from a file 3.87×
smaller, and **1.33× faster than `Int8` at the default parallelism** (1.21× on one thread; 1.35× and
1.24× when built for net11.0, which is the first runtime that can reach AVX-512 VNNI). §4 has the numbers,
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
dotnet run --project SentenceTransformers.Quantize -c Release -f net11.0 -- \
  convert --input harrier-oss-v1-270m.safetensors --output harrier-small-q4.stq

# the smallest build: ternary embedding table, 4-bit projections
dotnet run --project SentenceTransformers.Quantize -c Release -f net11.0 -- \
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
dotnet run --project SentenceTransformers.Quantize -c Release -f net11.0 -- \
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
(`harrier-pure-bench`).

| mode | 4 threads, net10 | 4 threads, net11 | 1 thread, net10 | 1 thread, net11 | resident |
|---|---|---|---|---|---|
| fp32 | — | — | 15938 | — | 1332 MB |
| `Int8`, load-time | 3049 | 3381 | 4391 | 3864 | 1046 / 540 MB |
| `Int4`, load-time | — | — | 13545 | — | 519 MB |
| `.stq` q4 g128 — first working version | 5068 | — | 11538 | — | 388 MB |
| **`.stq` q4 g128 — now** | **2301** | **2497** | **3624** | **3127** | **388 MB** |

Medians of three runs each; the box drifts by up to 10%, so read the in-process ratio rather than the
absolute times. The packed path is **1.33× faster than `Int8`** at the default parallelism on net10
and **1.35×** on net11; pinned to one thread, **1.21×** on net10 and **1.24×** on net11. It is 3.7×
faster than where it started single-threaded, at 0.8177 STS-B Spearman — fp32's number to four
decimals — in 388 MB against `Int8`'s 1046.

Targeting net11.0 is worth **14% of a whole single-threaded encode** on this host (3624 → 3127) and
12% for `Int8` (4391 → 3864), because .NET 11 is the first runtime that can reach AVX-512 VNNI. See
the end of this section. At four threads it did not help here — the same build measured 2301 ms on
net10 against 2497 on net11, with CPU time *down* 18% but wall time up, i.e. worse parallel
efficiency. Two plausible causes, not separated: AVX-512 frequency behaviour with all cores issuing
zmm, and Amdahl, since a third of the dot disappearing makes the serial remainder a larger share.
Worth re-measuring per deployment rather than assuming.

The order things were tried in is the order of intuition; the order of payoff was different.

| change | what it was worth |
|---|---|
| vectorizing the 4-bit nibble unpack | ~1% |
| widening the scale groups, 32 → 128 | ~7% (67% once the chains below were covered) |
| **tiling the kernel** | **1.63×, then a further 1.20×** |
| 4-way `vphaddd` group reduction + group-major scales | 1.25× |
| **vectorizing activation quantization** | **1531 → 84 ms/iter** |
| sharing one activation quantization across q/k/v and gate/up | (part of the above) |
| vectorizing the Walsh–Hadamard stages narrower than a vector | 775 → 233 ms/iter |
| row-level 4-bit unpack + the split-nibble layout | 1124 → 584 ms/iter |
| **blocking the weights so the accumulator lanes are output channels** | **the dot: 3541 → 2015 ms/iter** |
| prefetching the next scale group's codes during the dot | 584 → 428 ms/iter |
| **targeting net11.0, for 512-bit `vpdpbusd`** | **the dot: 2015 → 1327 ms/iter** |
| a 2-position tail kernel, and smaller items (per-channel bias, FMA, `SkipLocalsInit`) | a few % |

Three of those deserve a note, because the mistake each corrects is easy to repeat.

**The kernel computed one output channel at a time**, so `acc = DotAccumulate(acc, ...)` formed a
single serial dependency chain and stalled on its own ~5-cycle latency no matter how cheap the unpack
was. That is why vectorizing the unpack first was worth 1%.

**Activation quantization was scalar**, and at 22% of a packed forward pass it was larger than the
nibble unpacking the whole exercise was about. It is shared with the `Int8` path, so fixing it moved
that baseline too — from 5497 to ~4080 ms/iter. A comparison is only worth making once both sides have
had the obvious work done to them.

**The weight layout decided whether there was a horizontal reduction at all**, which was the largest
single item — see below.

Two things were tried and reverted, and are recorded so they are not retried blind. Folding the final
bias-and-scale pass into the last scale group (so it writes `y` directly instead of a second pass over
the accumulators) removed a real pass but cost the dot more than it saved: 2125 against 1996 ms/iter in
one profile, because carrying the extra state through the kernel disturbed the sixteen accumulators.
And prefetching two scale groups ahead instead of one fetches every line twice and measured slower.

Two notes on measuring this. `EncodeAsync` memoizes the last 16 vectors by input hash, so a benchmark
that re-encodes one small batch in a loop times a dictionary lookup rather than inference — the
corpus here is deliberately far larger than that cache. And each run prints the host's SIMD
capabilities and times both modes back to back, because run-to-run spread is around 10%.

### How the packed path overtook `Int8`, and how far ahead it can get

It is worth writing down the wrong reasoning as well as the right answer, because the wrong reasoning
was specific, quantitative, checked against the profile — and still wrong.

**The argument that it could not.** Both kernels feed the same instruction. `vpdpbusd` multiplies
unsigned bytes by signed bytes, 32 MACs per 256-bit operation, and .NET 10 exposes no 512-bit VNNI
(see below), so neither path can issue wider or fewer of them. Four-bit weights cannot be fed to it
packed either: a byte holding two codes contributes `lo + 16·hi` to one scalar sum, and no shuffle
recovers the two products from that. So the packed path must do **all** of `Int8`'s
multiply-accumulate work, and then unpack the codes and rotate the activations on top.

**What it missed.** Every step of that is true about the *multiply*. None of it is about the
*reduction*, and the reduction was not a fixed cost — it was a consequence of the operand layout.

`vpdpbusd` sums four byte products into each of its eight int32 lanes. Feed it a row of weights and
those eight lanes hold eight partial sums **of the same output channel**, so finishing a dot product
means adding the lanes together. `Int8` pays that once per output channel per row. The packed path
has a scale per 128 weights, so it paid it *every group* — three `vphaddd`s per four channels every
128 weights, and `vphaddd` is three micro-operations. That was about 30% of the kernel, and the
earlier notes kept attributing it to the format, as if per-group scales necessarily implied a
horizontal reduction. They do not.

Feed the same instruction a vector whose lane `L` holds four consecutive weights of output channel
`L`, against an activation vector that is those same four activations broadcast into all eight lanes,
and each lane accumulates a *different* output channel. The dot product is finished when the loop
ends. A scale group then costs one convert and one multiply-add per eight channels — no reduction at
all — and the per-group scales become free.

The weights have to be in that order, so `StqMatrix` rewrites them once at load (`BuildBlocked`): for
one tile of 32 channels and one scale group, the code order is `q·128 + b·32 + L·4 + m` — input quad,
then 8-channel block, then channel, then input within the quad. Quad-major so the four weight vectors
a step needs are 128 contiguous bytes; whole-tile groups so a tile's codes for a group unpack in one
call. It costs about 0.1s of load time and nothing at all on disk — the file format is untouched, and
the float fallback keeps the file's own row-major order.

**How far ahead it can get, single-threaded.** The profile now reads, per iteration on one thread:

| | `Int8` | `.stq` q4 |
|---|---|---|
| dot | 3053 | **2015** |
| unpack 4-bit codes | — | 428 |
| rotate activations | — | 233 |
| quantize activations | 85 | 85 |
| attention, norms, RoPE, GeGLU | 915 | 915 |
| **total** | **4053** | **3676** |

The packed dot is 1038 ms cheaper — it reads half the weight bytes and has no horizontal reduction —
and it spends 661 of that back on unpacking and rotating. That is the whole story, and it bounds what
is left. The dot is already at `vpdpbusd` throughput (13.2 G instructions in 2015 ms is better than
two per cycle), so it cannot be made faster; the unpack has a measured floor of ~274 ms (the rate it
runs at when its source is L1-hot rather than streaming from L3) and the rotation perhaps 150. So
**one-threaded the ceiling is about 4053/3439 = 1.18×**, and cutting the shared 915 ms helps almost
not at all, because it comes off both sides equally. That bound is specific to the 256-bit kernels it
was computed from: on net11.0 the dot shrinks for both sides and the measured single-thread ratio is
1.24×, above it. The method holds, the number does not travel.

Multi-threaded is a different regime and better: half the weight bytes matters more as cores contend,
and the measured ratio is 1.33× at the default 4 threads against 1.21× at one (net10; 1.35× and 1.24×
on net11). That trend is the thing to re-measure on a host with more cores.

**512-bit VNNI needs net11.0, and the CPUID flag misleads.** This host reports `avx512_vnni` in
`/proc/cpuinfo`, yet on .NET 10 the kernels ran at 256 bits, because none of the managed APIs reach
that extension: `AvxVnniInt8.V512` is a *different* one (AVX-VNNI-INT8, AVX10.2-class) and is
unsupported here, there is no standalone `Avx512Vnni` class, and `Avx10v1`/`Avx10v1.V512` — which
*are* supported — expose no `MultiplyWideningAndAdd` at all, only `MultiplyLow`. .NET 11 added
`AvxVnni.V512` (dotnet/runtime#128365), and that is the one. Measured in isolation it is 2.7× the MAC
throughput of the 256-bit form; in the model it takes the packed dot from 2015 to 1327 ms/iter.

Every project therefore multi-targets `net10.0;net11.0`. The instruction selection sits behind
`#if NET11_0_OR_GREATER` in `Vnni.DotAccumulate512`; the 512-bit packed kernel itself
(`StqMatrix.Vector512.cs`) is unconditional, since `AvxVnniInt8.V512` could already select it on
net10 for AVX10.2 parts. Which kernel a matrix uses is decided once at load by `Vnni.Has512Dot` —
deliberately *not* `Vnni.Use512`, which is also true for the widen-and-`vpmaddwd` emulation that is
slower than 256-bit `vpdpbusd` and would be the wrong choice. The blocked layout is rebuilt to match:
an accumulator covers sixteen channels instead of eight, so a tile is 64 channels rather than 32.

**A prediction this document got wrong.** It previously said that if a runtime ever exposed 512-bit
VNNI, both kernels would roughly halve their dot and that would *shrink* the packed path's lead,
since its unpack and rotation are fixed costs. Measured, the lead slightly **widened**: 1.21× → 1.24×
single-threaded. The fixed-cost reasoning was right as far as it went and still misses the layout,
exactly as the original impossibility argument did. `Int8`'s row-major tile still finishes each dot
product with a horizontal reduction, and that reduction gets *worse* at 512 bits — sixteen lanes to
fold instead of eight — while the packed kernel's blocked layout just widens to sixteen-channel
accumulators and keeps having no reduction at all. Widening the vector helps the layout that does not
pay per-group reduction more than the one that does.

What is still real from the old argument: the unpack and the rotation are genuine extra work the
packed path does and `Int8` does not, and neither is removable. Rotation was measured, not assumed —
converting with `--rotate embed` (rotate the embedding table, leave the projections in the original
basis, so no projection needs a rotated activation) drops mean embedding cosine from 0.980 to 0.958.

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
- On speed the packed path went from 2.13x slower than `Int8` to **1.33x faster at the default
  parallelism** (1.21x pinned to one thread; 1.35x / 1.24x on net11.0) at 388 MB against 1046. The decisive change was not the
  packing or the rotation but the weight layout: blocking the weights so a `vpdpbusd` accumulator's
  lanes are eight different output channels removes the per-scale-group horizontal reduction
  entirely. See "How the packed path overtook `Int8`", which records both the careful argument that
  said this was impossible and the arithmetic bounding how much further it can go.
