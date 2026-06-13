#!/usr/bin/env python3
"""
Token-count scalability sweep for Harrier Small using the original PyTorch /
sentence-transformers implementation, so it can be compared apples-to-apples with
the C# pure and ONNX builds.

Replays the byte-identical calibrated inputs written by the C# harness
(/tmp/harrier_scaling_inputs.json), encodes each on CPU through the public
SentenceTransformer.encode() path, and records the elapsed ms per encode at each
token count. Writes /tmp/harrier_scaling_python.json for merging with the C# results.

Usage:
    python scripts/harrier_scaling_bench.py
"""
import json
import os
import time
import statistics

import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "microsoft/harrier-oss-v1-270m"
INPUTS = "/tmp/harrier_scaling_inputs.json"
OUT = "/tmp/harrier_scaling_python.json"


def iterations_for(tokens: int) -> int:
    if tokens <= 256:
        return 7
    if tokens <= 512:
        return 5
    if tokens <= 1024:
        return 3
    if tokens <= 2048:
        return 2
    return 1


def main():
    # Match the C# run: CPU only, all 4 cores.
    torch.set_num_threads(os.cpu_count() or 4)
    print(f"torch {torch.__version__}, threads={torch.get_num_threads()}, device=cpu")

    with open(INPUTS, encoding="utf-8") as f:
        inputs = json.load(f)

    print(f"Loading {MODEL_ID} on CPU ...")
    model = SentenceTransformer(MODEL_ID, device="cpu")
    model.eval()
    tok = model.tokenizer

    results = []
    print()
    print("=== Harrier Small (PyTorch / sentence-transformers, CPU fp32) ===")
    for item in inputs:
        text = item["text"]
        target = item["target"]
        cs_tokens = item["tokens"]
        # Token count as the HF tokenizer sees it (should match the C# Gemma count).
        py_tokens = len(tok(text)["input_ids"])

        # Warmup.
        with torch.no_grad():
            model.encode([text], normalize_embeddings=True, show_progress_bar=False)

        n = iterations_for(cs_tokens)
        samples = []
        for _ in range(n):
            t0 = time.perf_counter()
            with torch.no_grad():
                model.encode([text], normalize_embeddings=True, show_progress_bar=False)
            samples.append((time.perf_counter() - t0) * 1000.0)
        samples.sort()
        median = samples[len(samples) // 2]
        print(f"  {cs_tokens:5d} tokens (py={py_tokens:5d}): {median:9.1f} ms  "
              f"(median of {n}, min {samples[0]:.1f})")
        results.append({
            "target": target,
            "tokens": cs_tokens,
            "py_tokens": py_tokens,
            "python_ms": median,
        })

    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print()
    print(f"Wrote Python results -> {OUT}")


if __name__ == "__main__":
    main()
