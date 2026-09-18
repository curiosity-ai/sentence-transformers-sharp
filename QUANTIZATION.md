# Quantized weights: the `.stq` format, the converter, and the runner

This document covers what PrismML's Bonsai models do to reach ~1.7 bits per weight, the format and
runtime this repository grew to do the same for Harrier Small Pure, and what the measurements
actually support.

**The short version.** The ternary machinery works and is verified, but round-to-nearest ternary is
only usable for one tensor in this model, so the container also carries a 4-bit band and the default
conversion is 4-bit throughout. That default is a **strict improvement on the shipped `Int4` mode —
better quality (0.984 vs 0.976 cosine) at about a third of the memory** — because it is the first
path here that quantizes the token embedding table at all. §4 has the numbers and §5 what ternary
needs to become viable.

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
  "__metadata__":  { "format": "stq", "format_version": "1", "architecture": "gemma3-text", ... },
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

Every row is a real conversion of the released checkpoint, scored the same way. Projections are
`q4_0` at group 32 throughout; only the embedding table changes.

| embedding table | file on disk | mean cosine | min cosine | ranking ρ |
|---|---|---|---|---|
| `tq1_0`, group 128 | **89.0 MB** | 0.95056 | 0.91290 | 0.96201 |
| `tq1_0`, group 64 | 91.5 MB | 0.95049 | 0.90276 | 0.96490 |
| `tq2_0`, group 64 | 99.0 MB | 0.95049 | 0.90276 | 0.96490 |
| **`q4_0`, group 128 (default)** | **136.5 MB** | **0.98430** | 0.95132 | **0.98931** |
| `q4_0`, group 32 | 144.0 MB | 0.98436 | 0.95154 | 0.98923 |
| — for reference, `Int4` load-time | ~410 MB resident | 0.97622 | 0.95592 | 0.97918 |

Two things follow, and they set the defaults:

- **4-bit throughout beats the shipped `Int4` mode on both axes** — 0.984 against 0.976 cosine, at
  136.5 MB against roughly 410 MB resident — purely because it is the first path here that
  compresses the embedding table. This is the default conversion.
- **A ternary embedding table is the smallest useful build**, 89.0 MB at 0.951 cosine. That is a real
  quality step down, so it is one flag away (`--embed-band tq1_0`) rather than the default, and it
  does not clear the default validation gates — lower `--min-mean-cosine` deliberately if you want
  it.

Group size barely matters for the embedding table in either band (0.00006 of cosine between `q4_0`
at 32 and at 128), so the default takes the coarser, smaller one.

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
- Ternarizing the whole model post-training does not work, for a reason that is a property of
  three-level quantization rather than of this implementation. Weights trained for ternary are
  required — which is how Bonsai does it.
- The embedding table is the exception and ternarizes usefully on its own (0.964 cosine,
  335 MB → 36.7 MB).
- The immediate win is unrelated to bit width: quantizing the embedding table at all. The default
  4-bit conversion is better and roughly 3× smaller than the `Int4` mode that ships today.
