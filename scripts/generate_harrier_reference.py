#!/usr/bin/env python3
"""
Generates golden reference embeddings for the Harrier models using the original
PyTorch / sentence-transformers implementation, so the C# ports (and every ONNX
quantization variant) can be checked for correctness against them.

For each model it encodes the exact same multilingual sentence batch the C#
HarrierVariantsTest uses, then writes the L2-normalized embeddings to
SentenceTransformers.Test/Resources as JSON. The C# test loads those files and
compares each variant's per-sentence embedding to the reference via cosine
similarity.

The sentences below MUST stay byte-for-byte identical to the array in
SentenceTransformers.Test/HarrierVariantsTest.cs.

Usage:
    pip install "torch" --index-url https://download.pytorch.org/whl/cpu
    pip install sentence-transformers
    python scripts/generate_harrier_reference.py
"""
import json
import os

import sentence_transformers
from sentence_transformers import SentenceTransformer

# Keep in sync with HarrierVariantsTest.cs.
SENTENCES = [
    "Good morning, how are you?",      # English
    "Buenos días, ¿cómo estás?",       # Spanish
    "おはようございます、お元気ですか？",  # Japanese
    "Hello world",                     # unrelated
    "The cat sat on the mat",          # unrelated
]

# (Hugging Face model id, output filename). The C# small package maps to 270m,
# the medium package to 0.6b.
MODELS = [
    ("microsoft/harrier-oss-v1-270m", "harrier-small-reference.json"),
    ("microsoft/harrier-oss-v1-0.6b", "harrier-medium-reference.json"),
]

OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "SentenceTransformers.Test",
    "Resources",
)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for model_id, filename in MODELS:
        print(f"Loading {model_id} ...")
        # default_prompt_name is null for both models, so encode() applies no
        # prompt - matching the C# EncodeAsync (document) path. The model's
        # Normalize module L2-normalizes the output, as the C# encoder does.
        model = SentenceTransformer(model_id, device="cpu")
        embeddings = model.encode(SENTENCES, normalize_embeddings=True)
        payload = {
            "model": model_id,
            "library": f"sentence-transformers {sentence_transformers.__version__}",
            "dim": int(embeddings.shape[1]),
            "sentences": SENTENCES,
            "embeddings": [[float(x) for x in row] for row in embeddings],
        }
        out_path = os.path.join(OUT_DIR, filename)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)
        print(f"  wrote {out_path}  (dim={payload['dim']}, {len(SENTENCES)} sentences)")


if __name__ == "__main__":
    main()
