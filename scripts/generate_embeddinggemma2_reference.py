#!/usr/bin/env python3
"""
Reference generator for the pure C# EmbeddingGemma 2 port (SentenceTransformers.EmbeddingGemma2).

It runs Google's own LiteRT-LM runtime (the `litert-lm` python runner) and the stock TFLite
interpreter over the exact `.litertlm` bundles published on Hugging Face, and writes:

  * tokenizer fixtures      - text -> SentencePiece ids, produced by the reference `sentencepiece`
                              library from the tokenizer embedded in the bundle;
  * engine embeddings       - final normalized vectors from `litert_lm.EmbeddingEngine` (the
                              end-to-end ground truth);
  * per-layer tensor dumps  - every intermediate the C# tests compare against layer by layer.
                              They are captured by running each sub-model (with its external weight
                              section inlined so the plain interpreter can load it) with
                              `experimental_preserve_all_tensors=True`.

Usage:
    python -m venv venv && ./venv/bin/pip install litert-lm ai-edge-litert sentencepiece numpy pillow
    gcc -O2 -shared -fPIC -o libstbref.so scripts/stbref.c -lm     # needs stb_image.h + stb_image_resize.h v0.97, see scripts/stbref.c (vision only)
    ./venv/bin/python scripts/generate_embeddinggemma2_reference.py --models <dir with .litertlm files> \
        --out <dump dir> --parts text,vision,audio --stb-lib ./libstbref.so \
        [--fixtures SentenceTransformers.Tests/Resources/embeddinggemma2]

Parts: `text` uses the 270M bundle, `vision` the 440M bundle and `audio` the 740M bundle. Each run writes
one manifest per --out directory; several directories can be passed to the tests separated by the
platform path separator.

The dump directory is consumed by the opt-in C# tests via EMBEDDINGGEMMA2_REFERENCE_DIR; the small
fixtures (tokenizer ids + engine embeddings) are committed under SentenceTransformers.Tests/Resources.

Binary format: every tensor is a raw little-endian float32 (or int32) file; `manifest.json` lists
name, dtype and shape for each.
"""
import argparse
import io
import json
import os
import sys
import tempfile

import numpy as np

MODEL_FILES = {
    "text-270m": "embeddinggemma-2-text-270m.litertlm",
    "text-vision-440m": "embeddinggemma-2-text-vision-440m.litertlm",
    "740m": "embeddinggemma-2-740m.litertlm",
}

# Diverse strings for tokenizer parity: whitespace runs, digits, code, emoji (byte fallback),
# CJK, RTL, combining marks, user-defined symbols, and the model's task prefixes.
TOKENIZER_CASES = [
    "", " ", "  ", "\n", "\n\n", "\t", "a", "Hello world", " Hello world ", "Hello  world",
    "Hello world!!!", "The year 2024 had 365 days; 3.14159 is pi.", "1234567890", "12 345 6.7e-8",
    "task: search result | query: What causes the northern lights?",
    "title: none | text: The northern lights are caused by charged particles from the sun.",
    "def f(x):\n    return x ** 2  # square\n", "for (int i = 0; i < n; i++) { sum += a[i]; }",
    "Buenos días, ¿cómo estás?", "おはようございます、お元気ですか？", "我爱自然语言处理。",
    "Привет, как дела?", "مرحبا بالعالم", "שלום עולם", "नमस्ते दुनिया", "Ελληνικά κείμενα",
    "naïve café résumé", "e\u0301 combining", "emoji 😀🎉👩‍👩‍👧 test", "🦀🦀🦀", "\u200b zero width",
    "<mask> is special", "an <|image|> placeholder", "<bos> literal", "<unused0><unused1>",
    "x" * 300, "ab" * 50, "The quick brown fox jumps over the lazy dog. " * 3,
    "URL: https://example.com/path?q=1&r=2#frag", "email@example.org", "C:\\Windows\\System32",
    "Tabs\tand\nnewlines\r\nmixed", "   leading and trailing   ", "ALL CAPS SHOUTING", "MiXeD CaSe",
    "a\u00a0b nbsp", "ﬁ ligature", "Ⅷ roman", "½ fraction", "\U0001F600", "\x00 null", "\ufeffbom",
]

ENGINE_CASES = [
    "Hello world",
    "task: search result | query: What causes the northern lights?",
    "title: none | text: The northern lights are caused by charged particles from the sun.",
    "task: sentence similarity | query: The cat sat on the mat",
    "Buenos días, ¿cómo estás?",
    "おはようございます、お元気ですか？",
    "def f(x):\n    return x ** 2  # square\n",
    "emoji 😀🎉 test",
    "The quick brown fox jumps over the lazy dog. " * 30,   # > 128 tokens -> 512 signature
]

LAYER_CASES = [
    "Hello world, this is a slightly longer sentence to test.",
    "task: search result | query: What causes the northern lights?",
]


# --------------------------------------------------------------------------------------------
# .litertlm unpacking + external-buffer inlining
# --------------------------------------------------------------------------------------------

def unpack(litertlm_path, out_dir):
    """Dumps every section of a .litertlm bundle using the official peek tool."""
    from litert_lm_builder import litertlm_peek
    os.makedirs(out_dir, exist_ok=True)
    litertlm_peek.peek_litertlm_file(litertlm_path, out_dir, io.StringIO())
    sections = {}
    for f in os.listdir(out_dir):
        if f.endswith(".tflite"):
            sections.setdefault(f.split("_tf_lite_")[1][:-7], {})["tflite"] = os.path.join(out_dir, f)
        elif f.endswith(".weight"):
            sections.setdefault(f.split("_tf_lite_")[1][:-7], {})["weight"] = os.path.join(out_dir, f)
        elif f.endswith(".spiece"):
            sections["spiece"] = os.path.join(out_dir, f)
    return sections


def inline_external_buffers(tflite_path, weight_path, dst):
    """Rewrites a .tflite whose tensors reference an external weight section into a self-contained
    model the stock interpreter can load."""
    from ai_edge_litert import schema_py_generated as S
    from ai_edge_litert.tools import flatbuffer_utils as fu
    m = fu.read_model_from_bytearray(bytearray(open(tflite_path, "rb").read()))
    w = np.fromfile(weight_path, dtype=np.uint8) if weight_path else None
    ext = {e.id: e for e in (m.externalBuffers or [])}
    cache = {}
    for sg in m.subgraphs:
        for t in sg.tensors:
            if t.externalBuffer:
                e = ext[t.externalBuffer]
                key = (e.offset, e.length)
                if key not in cache:
                    b = S.BufferT()
                    b.data = w[e.offset:e.offset + e.length].copy()
                    m.buffers.append(b)
                    cache[key] = len(m.buffers) - 1
                t.buffer = cache[key]
                t.externalBuffer = 0
    m.externalBuffers = None
    m.externalBufferGroups = None
    open(dst, "wb").write(fu.convert_object_to_bytearray(m))
    return dst


class OpList:
    """Op sequence of one subgraph with composite ops resolved to their names."""

    def __init__(self, tflite_path, subgraph):
        from ai_edge_litert import schema_py_generated as S
        self.S = S
        buf = open(tflite_path, "rb").read()
        self.m = S.Model.GetRootAs(buf, 0)
        names = {v: k for k, v in vars(S.BuiltinOperator).items() if not k.startswith("_")}
        sg = self.m.Subgraphs(subgraph)
        self.sg = sg
        self.ops = []
        for o in range(sg.OperatorsLength()):
            op = sg.Operators(o)
            oc = self.m.OperatorCodes(op.OpcodeIndex())
            name = names[max(oc.BuiltinCode(), oc.DeprecatedBuiltinCode())]
            if name == "STABLEHLO_COMPOSITE":
                opt = S.StableHLOCompositeOptions()
                bo = op.BuiltinOptions2()
                opt.Init(bo.Bytes, bo.Pos)
                name = opt.Name().decode()
            self.ops.append((name, [op.Inputs(j) for j in range(op.InputsLength())],
                             [op.Outputs(j) for j in range(op.OutputsLength())]))

    def shape(self, t):
        return tuple(int(v) for v in self.sg.Tensors(t).ShapeAsNumpy())

    def const_f32(self, t):
        b = self.m.Buffers(self.sg.Tensors(t).Buffer())
        if b is None or b.DataLength() == 0:
            return None
        return np.frombuffer(b.DataAsNumpy().tobytes(), np.float32)


class Dumper:
    def __init__(self, out_dir):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.manifest = {}

    def put(self, name, arr):
        arr = np.ascontiguousarray(arr)
        dtype = {np.dtype(np.float32): "f32", np.dtype(np.int32): "i32"}[arr.dtype]
        rel = name + "." + dtype
        os.makedirs(os.path.dirname(os.path.join(self.out_dir, rel)), exist_ok=True)
        arr.tofile(os.path.join(self.out_dir, rel))
        self.manifest[name] = {"file": rel, "dtype": dtype, "shape": list(arr.shape)}

    def close(self, extra=None):
        doc = {"tensors": self.manifest}
        if extra:
            doc.update(extra)
        with open(os.path.join(self.out_dir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)


# --------------------------------------------------------------------------------------------
# text
# --------------------------------------------------------------------------------------------

def text_context(sections):
    """Embedder + text encoder runners shared by the multimodal dumps."""
    import sentencepiece as spm
    from ai_edge_litert.interpreter import Interpreter
    tmp = tempfile.mkdtemp()
    emb_path = inline_external_buffers(sections["embedder"]["tflite"], sections["embedder"]["weight"], os.path.join(tmp, "embedder.tflite"))
    enc_path = inline_external_buffers(sections["text_encoder"]["tflite"], sections["text_encoder"]["weight"], os.path.join(tmp, "text_encoder.tflite"))
    embedder = Interpreter(emb_path).get_signature_runner("embedder_1x1")
    encoder = Interpreter(enc_path)

    def embed(ids):
        return np.stack([embedder(token_ids=np.array([[t]], np.int32))["embeddings"][0, 0] for t in ids])

    def encode(seq):
        n = len(seq)
        L = text_signature(n)
        E = np.zeros((1, L, 512), np.float32)
        M = np.zeros((1, L), np.float32)
        E[0, :n] = seq
        M[0, :n] = 1
        out = encoder.get_signature_runner(f"encoder_1x{L}")(embeddings=E, input_mask=M)["encodings"][0]
        return (out / np.linalg.norm(out)).astype(np.float32)
    return {"sp": spm.SentencePieceProcessor(model_file=sections["spiece"]), "embed": embed, "encode": encode}


def text_signature(n, sizes=(128, 256, 512, 1024)):
    for s in sizes:
        if n <= s:
            return s
    raise ValueError(f"{n} tokens exceed the largest signature")


def dump_text(sections, model_path, dumper, fixtures_dir):
    import sentencepiece as spm
    from ai_edge_litert.interpreter import Interpreter
    from litert_lm.embedding_engine import EmbeddingEngine
    import litert_lm

    sp = spm.SentencePieceProcessor(model_file=sections["spiece"])
    tmp = tempfile.mkdtemp()
    emb_path = inline_external_buffers(sections["embedder"]["tflite"], sections["embedder"]["weight"], os.path.join(tmp, "embedder.tflite"))
    enc_path = inline_external_buffers(sections["text_encoder"]["tflite"], sections["text_encoder"]["weight"], os.path.join(tmp, "text_encoder.tflite"))
    embedder = Interpreter(emb_path).get_signature_runner("embedder_1x1")

    def embed(ids):
        return np.stack([embedder(token_ids=np.array([[t]], np.int32))["embeddings"][0, 0] for t in ids])

    # Tokenizer fixture.
    tok = [{"text": t, "ids": sp.encode(t)} for t in TOKENIZER_CASES]

    # Engine embeddings (end-to-end ground truth).
    engine = EmbeddingEngine(model_path, backend=litert_lm.interfaces.Backend.CPU())
    eng = []
    for t in ENGINE_CASES:
        v = np.array(engine.compute_embedding(t).embedding, np.float32)
        eng.append({"text": t, "ids": [2] + sp.encode(t) + [1], "embedding": v.tolist()})
    engine.close()

    if fixtures_dir:
        os.makedirs(fixtures_dir, exist_ok=True)
        with open(os.path.join(fixtures_dir, "tokenizer_cases.json"), "w", encoding="utf-8") as f:
            json.dump(tok, f, ensure_ascii=False, indent=0)
        with open(os.path.join(fixtures_dir, "engine_text_embeddings.json"), "w", encoding="utf-8") as f:
            json.dump(eng, f, ensure_ascii=False)

    # Per-layer dumps. The encoder signatures are padded to a fixed length; only the first n rows
    # (the real tokens) are dumped - padded rows never influence them (bidirectional padding mask,
    # per-row activation quantization).
    sig_index = {"encoder_1x1024": 0, "encoder_1x128": 1, "encoder_1x2048": 2, "encoder_1x256": 3, "encoder_1x512": 4, "encoder_1x8192": 5}
    cases = []
    for ci, text in enumerate(LAYER_CASES):
        ids = [2] + sp.encode(text) + [1]
        n = len(ids)
        L = text_signature(n)
        sub = sig_index[f"encoder_1x{L}"]
        ops = OpList(sections["text_encoder"]["tflite"], sub)
        it = Interpreter(enc_path, experimental_preserve_all_tensors=True)
        E = np.zeros((1, L, 512), np.float32)
        M = np.zeros((1, L), np.float32)
        E[0, :n] = embed(ids)
        M[0, :n] = 1
        out = it.get_signature_runner(f"encoder_1x{L}")(embeddings=E, input_mask=M)["encodings"][0]
        g = lambda t: it._interpreter.GetTensor(t, sub)[0]
        p = f"text/case{ci}/"
        dumper.put(p + "ids", np.array(ids, np.int32))
        dumper.put(p + "embeddings", E[0, :n])
        # Per-layer-input projection (rms-normed), [n, 24, 512]: input to UNPACK.
        unpack_in = next(ins[0] for name, ins, outs in ops.ops if name == "UNPACK")
        dumper.put(p + "per_layer_inputs", g(unpack_in)[:n])
        # Residual stream after every layer: outputs of the per-layer scalar MUL.
        resid = [outs[0] for name, ins, outs in ops.ops
                 if name == "MUL" and ops.shape(ins[1]) == (1, 1, 1) and ops.const_f32(ins[1]) is not None]
        assert len(resid) == 24, len(resid)
        for l, t in enumerate(resid):
            dumper.put(p + f"layer{l:02d}", g(t)[:n])
        # Final norm output and pre-normalization projection.
        last_norm = [outs[0] for name, ins, outs in ops.ops if name == "odml.rms_norm"][-1]
        dumper.put(p + "final_norm", g(last_norm)[:n])
        dumper.put(p + "encodings", out.astype(np.float32))
        # Layer-0 sub-steps (attention block), handy when a layer-level comparison fails.
        fcs = [(ins, outs) for name, ins, outs in ops.ops if name == "FULLY_CONNECTED"]
        dumper.put(p + "l0_q", g(fcs[1][1][0])[:n])
        dumper.put(p + "l0_attn", g(fcs[4][0][0])[:n])      # input of o-proj = attention output
        dumper.put(p + "l0_oproj", g(fcs[4][1][0])[:n])
        cases.append({"text": text, "signature": L})
    return {"text_cases": cases}


# --------------------------------------------------------------------------------------------
# vision
# --------------------------------------------------------------------------------------------

def _stb(stb_lib):
    import ctypes
    lib = ctypes.CDLL(stb_lib)
    lib.stbref_load.restype = ctypes.POINTER(ctypes.c_ubyte)

    def load(data):
        w = ctypes.c_int()
        h = ctypes.c_int()
        p = lib.stbref_load(data, len(data), ctypes.byref(w), ctypes.byref(h))
        arr = np.ctypeslib.as_array(p, shape=(h.value * w.value * 3,)).copy()
        lib.stbref_free(p)
        return arr.reshape(h.value, w.value, 3)

    def resize(img, nw, nh):
        h, w, _ = img.shape
        out = np.zeros((nh, nw, 3), np.uint8)
        assert lib.stbref_resize(np.ascontiguousarray(img).ctypes.data_as(ctypes.c_void_p), w, h,
                                 out.ctypes.data_as(ctypes.c_void_p), nw, nh)
        return out
    return load, resize


def aspect_preserving_size(width, height, max_patches, patch=16, pool=3):
    """GetAspectRatioPreservingSize from LiteRT-LM (float32 arithmetic)."""
    import math
    f = np.float32
    total_px = f(width * height)
    target_px = f(max_patches * patch * patch)
    factor = f(math.sqrt(f(target_px / total_px)))
    ih = f(factor * f(height))
    iw = f(factor * f(width))
    side = pool * patch
    th = int(math.floor(f(ih / side))) * side
    tw = int(math.floor(f(iw / side))) * side
    max_side = (max_patches // (pool * pool)) * side
    if th == 0:
        th = side
        tw = min(int(math.floor(f(width) / f(height))) * side, max_side)
    elif tw == 0:
        tw = side
        th = min(int(math.floor(f(height) / f(width))) * side, max_side)
    return th, tw


def make_test_images():
    """Deterministic synthetic PNGs: one that needs no resize, two that do."""
    from PIL import Image, ImageDraw
    out = []
    for (w, h) in [(528, 528), (640, 480), (300, 900)]:
        rng = np.random.default_rng(w)
        yy, xx = np.mgrid[0:h, 0:w]
        a = np.stack([(xx * 255 / w), (yy * 255 / h), ((xx + yy) % 97) * 2.6], -1)
        img = Image.fromarray(a.astype(np.uint8))
        d = ImageDraw.Draw(img)
        for _ in range(12):
            x0, y0 = rng.integers(0, w), rng.integers(0, h)
            r = rng.integers(10, max(11, min(w, h) // 4))
            d.ellipse([x0 - r, y0 - r, x0 + r, y0 + r], fill=tuple(int(v) for v in rng.integers(0, 255, 3)))
        b = io.BytesIO()
        img.save(b, "PNG", optimize=True)
        out.append((f"synthetic_{w}x{h}.png", b.getvalue()))
    return out


def dump_vision(sections, model_path, dumper, fixtures_dir, stb_lib, text_ctx):
    from ai_edge_litert.interpreter import Interpreter
    from litert_lm.embedding_engine import EmbeddingEngine, EmbeddingOptions
    from litert_lm._messages import ImageBytes
    import litert_lm
    load, resize = _stb(stb_lib)
    tmp = tempfile.mkdtemp()
    enc_path = inline_external_buffers(sections["vision_encoder"]["tflite"], sections["vision_encoder"]["weight"], os.path.join(tmp, "vision_encoder.tflite"))
    ad_path = inline_external_buffers(sections["vision_adapter"]["tflite"], sections["vision_adapter"]["weight"], os.path.join(tmp, "vision_adapter.tflite"))
    eoi = Interpreter(sections["end_of_vision"]["tflite"]).get_signature_runner("eoi")()["eoi_embedding"].reshape(-1)
    sp, embed, encode = text_ctx["sp"], text_ctx["embed"], text_ctx["encode"]
    soi = sp.piece_to_id("<|image>")

    engine = EmbeddingEngine(model_path, backend=litert_lm.interfaces.Backend.CPU(), vision_backend=litert_lm.interfaces.Backend.CPU())
    images = make_test_images()
    fixtures = []
    cases = []
    for ii, (name, png) in enumerate(images):
        for tokens in (140, 70):
            max_patches = tokens * 9
            img = load(png)
            h, w, _ = img.shape
            th, tw = aspect_preserving_size(w, h, max_patches)
            res = img if (th, tw) == (h, w) else resize(img, tw, th)
            f = (res.astype(np.float32) * np.float32(1.0 / 255.0)).astype(np.float32)
            ph, pw = th // 16, tw // 16
            patches = f.reshape(ph, 16, pw, 16, 3).transpose(0, 2, 1, 3, 4).reshape(ph * pw, 768)
            pos = np.stack(np.meshgrid(np.arange(pw), np.arange(ph)), -1).reshape(-1, 2).astype(np.int32)
            P = np.zeros((1, max_patches, 768), np.float32)
            P[0, :len(patches)] = patches
            X = np.full((1, max_patches, 2), -1, np.int32)
            X[0, :len(pos)] = pos
            sig = {140: 0, 70: 1}[tokens]
            it = Interpreter(enc_path, experimental_preserve_all_tensors=True)
            out = it.get_signature_runner(f"vision_{tokens}")(images=P, positions_xy=X)
            mask = out["mask"][0].astype(bool)
            nv = int(mask.sum())
            feats = out["features"][0]
            A = np.zeros((1, tokens, 768), np.float32)
            A[0, :nv] = feats[:nv]
            mm = Interpreter(ad_path).get_signature_runner(f"vision_adapter_{tokens}")(soft_tokens=A)["mm_embedding"][0][:nv]
            seq = np.concatenate([embed([2, soi]), mm, eoi[None, :], embed([1])])
            mine = encode(seq)
            ref = np.array(engine.compute_embedding([ImageBytes(png)], EmbeddingOptions(vision_tokens_per_image=tokens)).embedding, np.float32)
            p = f"vision/img{ii}_t{tokens}/"
            dumper.put(p + "resized_rgb", res.astype(np.int32))
            dumper.put(p + "patches", P[0])
            dumper.put(p + "positions", X[0])
            ops = OpList(sections["vision_encoder"]["tflite"], sig)
            g = lambda t: it._interpreter.GetTensor(t, sig)[0]
            adds = [outs[0] for name_, ins, outs in ops.ops if name_ == "ADD" and ops.shape(outs[0]) == (1, max_patches, 768)]
            layer_out = [adds[0]] + [adds[2 * l + 2] for l in range(16)]
            for l, t in enumerate(layer_out):
                dumper.put(p + f"layer{l - 1:02d}" if l else p + "embeddings", g(t))
            dumper.put(p + "features", feats)
            dumper.put(p + "mask", mask.astype(np.int32))
            dumper.put(p + "soft_tokens", mm)
            dumper.put(p + "encodings", mine)
            dumper.put(p + "engine_embedding", ref)
            cases.append({"image": name, "tokens": tokens, "size": [th, tw], "valid_tokens": nv, "cos_interpreter_vs_engine": float(mine @ ref)})
            print(name, tokens, (th, tw), nv, "interpreter-vs-engine cos", float(mine @ ref))
            if tokens == 140:
                fixtures.append({"image": name, "embedding": ref.tolist()})
    dumper.put("vision/end_of_vision", eoi.astype(np.float32))
    # Interleaved text + image + text (one embedding).
    name, png = images[0]
    ref = np.array(engine.compute_embedding(["A photo of colorful circles: ", ImageBytes(png), " (synthetic test image)"]).embedding, np.float32)
    fixtures.append({"image": name, "text_before": "A photo of colorful circles: ", "text_after": " (synthetic test image)", "embedding": ref.tolist()})
    engine.close()
    if fixtures_dir:
        for name, png in images:
            with open(os.path.join(fixtures_dir, name), "wb") as fh:
                fh.write(png)
        with open(os.path.join(fixtures_dir, "engine_image_embeddings.json"), "w", encoding="utf-8") as fh:
            json.dump(fixtures, fh)
    return {"vision_cases": cases}


# --------------------------------------------------------------------------------------------
# audio
# --------------------------------------------------------------------------------------------

AUDIO_SECONDS = [1.0, 3.3]


def make_test_wav(seconds, seed, sr=16000):
    """Deterministic 16 kHz mono PCM16 test clip (chirp + gated tone + noise)."""
    import wave
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    sig = 0.3 * np.sin(2 * np.pi * (220 + 180 * t) * t) + 0.2 * np.sin(2 * np.pi * 660 * t) * (t % 0.5 < 0.25) + 0.05 * rng.standard_normal(len(t))
    pcm = np.clip(sig * 32767, -32768, 32767).astype(np.int16)
    b = io.BytesIO()
    w = wave.open(b, "wb")
    w.setnchannels(1)
    w.setsampwidth(2)
    w.setframerate(sr)
    w.writeframes(pcm.tobytes())
    w.close()
    return b.getvalue(), pcm.astype(np.float32) / np.float32(32768)


def log_mel_reference(pcm, cfg):
    """numpy port of LiteRT-LM's miniaudio front-end (semicausal framing, Hann, FFT, HTK mel, log).

    Uses numpy's double-precision FFT, so it agrees with the engine's single-precision kiss_fftr to ~1e-6; the
    C# port reproduces kiss_fftr bit for bit. Only the per-layer interpreter dumps use this."""
    import math
    frame, hop, nfft, nmel = cfg["frame_length"], cfg["hop_length"], cfg["fft_length"], cfg["num_mel_bins"]
    # The engine's bundled cosf/logf are correctly rounded: double cos/log rounded to float reproduce them.
    win = np.array([np.float32(0.5 - 0.5 * float(np.float32(math.cos(float(np.float32(np.float32(np.pi * 2.0 / frame) * np.float32(i))))))) for i in range(frame)], np.float32)
    queue = list(np.zeros(hop, np.float32))
    step = frame - hop
    frames, pos = [], 0
    while True:
        rem = len(pcm) - pos
        if step > rem:
            queue += list(pcm[pos:])
            break
        queue = queue[len(queue) - (frame - step):] + list(pcm[pos:pos + step]) if step < frame else list(pcm[pos + step - frame:pos + step])
        pos += step
        step = hop
        frames.append(np.array(queue, np.float32))
    if queue and len(queue) < frame:
        frames.append(np.array(queue + [0.0] * (frame - len(queue)), np.float32))
    bins = nfft // 2 + 1
    hz = cfg["sample_rate_hz"] / (2.0 * (bins - 1))
    f2m = lambda f: 1127.0 * math.log(1.0 + f / 700.0)
    mlo, mhi = f2m(cfg["mel_low_hz"]), f2m(cfg["mel_high_hz"])
    spacing = (mhi - mlo) / (nmel + 1)
    start, end = int(1.5 + cfg["mel_low_hz"] / hz), int(cfg["mel_high_hz"] / hz)
    rows = []
    for fr in frames:
        x = np.zeros(nfft, np.float64)
        pad = (nfft - frame) // 2
        x[pad:pad + frame] = (fr * win).astype(np.float32)
        X = np.fft.rfft(x)
        p = (X.real.astype(np.float32) ** 2 + X.imag.astype(np.float32) ** 2).astype(np.float32)
        mel = np.zeros(nmel)
        for i in range(start, end + 1):
            mp = (f2m(i * hz) - mlo) / spacing - 1
            ch = int(math.ceil(mp)) - 1
            wgt = 1.0 - (mp - ch)
            sv = math.sqrt(float(p[i]))
            if ch >= 0:
                mel[ch] += sv * wgt
            if ch + 1 < nmel:
                mel[ch + 1] += sv - sv * wgt
        rows.append(np.array([np.float32(math.log(float(v))) for v in (mel.astype(np.float32) + np.float32(cfg["mel_floor"]))], np.float32))
    return np.stack(rows)


def dump_audio(sections, model_path, dumper, fixtures_dir, text_ctx):
    from ai_edge_litert.interpreter import Interpreter
    from litert_lm.embedding_engine import EmbeddingEngine
    from litert_lm._messages import AudioBytes
    import litert_lm
    tmp = tempfile.mkdtemp()
    enc_path = inline_external_buffers(sections["audio_encoder_hw"]["tflite"], sections["audio_encoder_hw"]["weight"], os.path.join(tmp, "audio_encoder.tflite"))
    ad_path = inline_external_buffers(sections["audio_adapter"]["tflite"], sections["audio_adapter"]["weight"], os.path.join(tmp, "audio_adapter.tflite"))
    eoa = Interpreter(sections["end_of_audio"]["tflite"]).get_signature_runner("eoa")()["eoa_embedding"].reshape(-1)
    cfg = dict(sample_rate_hz=16000, frame_length=320, hop_length=160, fft_length=512, num_mel_bins=128, mel_low_hz=0.0, mel_high_hz=8000.0, mel_floor=0.001)
    sp, embed, encode = text_ctx["sp"], text_ctx["embed"], text_ctx["encode"]
    soa = sp.piece_to_id("<|audio>")
    engine = EmbeddingEngine(model_path, backend=litert_lm.interfaces.Backend.CPU(), audio_backend=litert_lm.interfaces.Backend.CPU())
    adapter = Interpreter(ad_path).get_signature_runner("audio_adapter_12")
    ops = OpList(sections["audio_encoder_hw"]["tflite"], 0)
    fixtures, cases = [], []
    for ci, secs in enumerate(AUDIO_SECONDS):
        wav, pcm = make_test_wav(secs, int(secs * 10))
        mel = log_mel_reference(pcm, cfg)
        it = Interpreter(enc_path, experimental_preserve_all_tensors=True)
        runner = it.get_signature_runner("serving_default")
        details = runner.get_input_details()
        state = {k: np.zeros(v["shape"], v["dtype"]) for k, v in details.items() if k not in ("segment_values", "segment_mask")}
        W, OV = 51, 3
        pos, chunk, toks = 0, 0, []
        p = f"audio/clip{ci}/"
        while pos + W <= len(mel) or pos + OV < len(mel):
            L = min(W, len(mel) - pos)
            seg = np.zeros((1, W, 128), np.float32)
            seg[0, :L] = mel[pos:pos + L]
            m = np.zeros((1, W), bool)
            m[0, :L] = True
            out = runner(segment_values=seg, segment_mask=m, **state)
            mask = out["mask"][0]
            nv = int(np.max(np.nonzero(mask)[0]) + 1) if mask.any() else 0
            ad = adapter(features=out["features"], mask=out["mask"])["output_0"][0]
            toks.append(ad[:nv])
            dumper.put(p + f"chunk{chunk}/features", out["features"][0].astype(np.float32))
            dumper.put(p + f"chunk{chunk}/mask", mask.astype(np.int32))
            dumper.put(p + f"chunk{chunk}/adapter", ad.astype(np.float32))
            if chunk < 2:
                # Every op output of the first two chunks (fresh and carried state), dequantized.
                vals = []
                for oi, (name_, ins, outs) in enumerate(ops.ops):
                    t = it._interpreter.GetTensor(outs[0], 0)
                    q = ops.sg.Tensors(outs[0]).Quantization()
                    if t.dtype == np.int8 and q is not None and q.ScaleLength():
                        t = (t.astype(np.float32) - q.ZeroPoint(0)) * q.Scale(0)
                    vals.append(np.asarray(t, np.float32).ravel())
                dumper.put(p + f"chunk{chunk}/op_outputs", np.concatenate(vals).astype(np.float32))
                dumper.put(p + f"chunk{chunk}/op_sizes", np.array([len(v) for v in vals], np.int32))
            for k in state:
                state[k] = out[k]
            pos += W - OV
            chunk += 1
        tokens = np.concatenate(toks)
        seq = np.concatenate([embed([2, soa]), tokens, eoa[None, :], embed([1])])
        mine = encode(seq)
        ref = np.array(engine.compute_embedding([AudioBytes(wav)]).embedding, np.float32)
        dumper.put(p + "pcm", pcm)
        dumper.put(p + "mel", mel)
        dumper.put(p + "tokens", tokens.astype(np.float32))
        dumper.put(p + "encodings", mine)
        dumper.put(p + "engine_embedding", ref)
        name = f"tone_{secs:.1f}s.wav"
        cases.append({"clip": name, "seconds": secs, "chunks": chunk, "tokens": int(len(tokens)), "cos_interpreter_vs_engine": float(mine @ ref)})
        print(name, "chunks", chunk, "tokens", len(tokens), "interpreter-vs-engine cos", float(mine @ ref))
        fixtures.append({"audio": name, "embedding": ref.tolist()})
        if fixtures_dir:
            with open(os.path.join(fixtures_dir, name), "wb") as fh:
                fh.write(wav)
    dumper.put("audio/end_of_audio", eoa.astype(np.float32))
    engine.close()
    if fixtures_dir:
        with open(os.path.join(fixtures_dir, "engine_audio_embeddings.json"), "w", encoding="utf-8") as fh:
            json.dump(fixtures, fh)
    return {"audio_cases": cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixtures", default=None)
    ap.add_argument("--parts", default="text")
    ap.add_argument("--stb-lib", default=None, help="shared library built from stb_image.h + stb_image_resize.h v0.97 (see scripts/stbref.c)")
    args = ap.parse_args()

    extra = {}
    dumper = Dumper(args.out)
    parts = args.parts.split(",")
    work = tempfile.mkdtemp()
    if "text" in parts:
        path = os.path.join(args.models, MODEL_FILES["text-270m"])
        secs = unpack(path, os.path.join(work, "text-270m"))
        extra.update(dump_text(secs, path, dumper, args.fixtures))
    if "audio" in parts:
        path = os.path.join(args.models, MODEL_FILES["740m"])
        secs = unpack(path, os.path.join(work, "740m"))
        extra.update(dump_audio(secs, path, dumper, args.fixtures, text_context(secs)))
    if "vision" in parts:
        path = os.path.join(args.models, MODEL_FILES["text-vision-440m"])
        secs = unpack(path, os.path.join(work, "text-vision-440m"))
        extra.update(dump_vision(secs, path, dumper, args.fixtures, args.stb_lib, text_context(secs)))
    dumper.close(extra)
    print("wrote", args.out)


if __name__ == "__main__":
    sys.exit(main())
