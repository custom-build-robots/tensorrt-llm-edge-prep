"""
run_engine_qwen_fp16.py
Loads the persistent TensorRT engine for Qwen2.5-7B-Instruct FP16 and runs inference
Target system: Ubuntu 24.04 with RTX A6000 Ada (SM89, FP8-capable)
Author: Ingmar Stapel + AI assistants
Date: 2026-05-16
Version 1.0
Prerequisites: build_qwen_fp16.sh has been run successfully
(engine exists at /workspace/engines/qwen2.5-7b-fp16/rank0.engine)

Purpose:
    Loads the persistent TensorRT engine built by build_qwen_fp16.sh
    and runs inference on three test prompts. Measures engine load time
    and generation throughput (tokens/sec). The counterpart to the build
    script:
        build_qwen_fp16.sh       -> creates the engine
        run_engine_qwen_fp16.py  -> uses the engine (this script)

Details:
    - Engine is loaded directly from rank0.engine, no build step
    - Tokenizer comes from the HF cache (Qwen/Qwen2.5-7B-Instruct)
    - Performance statistics are printed for later comparison with FP8

Important — backend choice:
    from tensorrt_llm._tensorrt_engine import LLM   # TensorRT backend (correct)
    from tensorrt_llm import LLM                     # PyTorch backend (WRONG here)
    Only the TensorRT backend can load a pre-built .engine file.
    Using the default import would try to find an HF checkpoint and
    fail with "TypeError: 'NoneType' object is not subscriptable".

    The `if __name__ == '__main__':` guard is mandatory because
    TensorRT-LLM internally spawns MPI worker processes.

Usage (inside the container):
    python3 /workspace/run_engine_qwen_fp16.py
"""

import time
from tensorrt_llm._tensorrt_engine import LLM
from tensorrt_llm import SamplingParams


ENGINE_DIR = "/workspace/engines/qwen2.5-7b-fp16"
TOKENIZER = "Qwen/Qwen2.5-7B-Instruct"   # Tokenizer aus HF-Cache


def main():
    print("=" * 60)
    print("  TensorRT Engine Inferenz-Test")
    print(f"  Engine: {ENGINE_DIR}")
    print(f"  Tokenizer: {TOKENIZER}")
    print("=" * 60)

    # === Stufe 1: Engine laden ===
    print("\n=== Lade Engine (kein Build mehr — direkt aus rank0.engine) ===")
    t_load_start = time.time()
    llm = LLM(
        model=ENGINE_DIR,
        tokenizer=TOKENIZER,
    )
    load_seconds = time.time() - t_load_start
    print(f"[ OK ] Engine in {load_seconds:.2f}s geladen")

    # === Stufe 2: Inferenz ===
    sp = SamplingParams(temperature=0.7, max_tokens=128)
    prompts = [
        "Erkläre kurz, was eine TensorRT-Engine ist.",
        "Was ist ein KV-Cache und wofür braucht man ihn?",
        "Beschreibe in 3 Sätzen den Unterschied zwischen Prefill und Decode.",
    ]

    print(f"\n=== Generiere Ausgaben für {len(prompts)} Prompts ===")
    t_gen_start = time.time()
    outputs = llm.generate(prompts, sp)
    gen_seconds = time.time() - t_gen_start

    # === Ergebnisse ausgeben ===
    total_tokens = 0
    for i, out in enumerate(outputs, 1):
        gen_text = out.outputs[0].text
        token_count = len(out.outputs[0].token_ids) if hasattr(out.outputs[0], "token_ids") else 0
        total_tokens += token_count

        print(f"\n{'=' * 60}")
        print(f"  Prompt {i}: {out.prompt}")
        print(f"{'=' * 60}")
        print(gen_text)
        if token_count:
            print(f"\n  [Generated tokens: {token_count}]")

    # === Performance-Statistik ===
    print("\n" + "=" * 60)
    print("  Performance-Statistik (Interview-Tabelle)")
    print("=" * 60)
    print(f"  Engine:           {ENGINE_DIR}")
    print(f"  Engine-Load-Zeit: {load_seconds:.2f}s")
    print(f"  Generation-Zeit:  {gen_seconds:.2f}s (für {len(prompts)} Prompts)")
    if total_tokens:
        tokens_per_sec = total_tokens / gen_seconds
        print(f"  Total Tokens:     {total_tokens}")
        print(f"  Tokens/sec:       {tokens_per_sec:.2f}")
    print("=" * 60)


if __name__ == '__main__':
    main()
