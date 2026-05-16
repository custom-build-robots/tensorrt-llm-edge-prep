"""
qwen_fp16.py
Loads Qwen2.5-7B-Instruct via the PyTorch backend and runs sample inference
Target system: Ubuntu 24.04 with RTX A6000 Ada (SM89, FP8-capable)
Author: Ingmar Stapel + AI assistants
Date: 2026-05-16
Version 1.0
Prerequisites: setup_trtllm.sh and start_trtllm.sh have been run successfully
(container running, HF token configured)

Purpose:
    Variant 1: PyTorch backend (default since TRT-LLM 1.x)
    Triggers the HuggingFace download of Qwen2.5-7B-Instruct into the local
    cache and validates that the model runs end-to-end on the target GPU.
    This is the required preparation step before build_qwen_fp16.sh, which
    expects the model to already be present in the HF cache.

Details:
    - No AOT engine build, no ONNX export
    - TRT-LLM uses PyTorch with optimized kernels
    - Faster to iterate with, slightly slower in steady state
    - Same mode that smoke.py uses automatically

What you observe:
    - Model loading phase (weights into VRAM)
    - Direct jump into inference, no "engine build" step
    - KV cache allocation (visible in log output)

Important:
    The `if __name__ == '__main__':` guard is mandatory because
    TensorRT-LLM internally spawns MPI worker processes.

Usage (inside the container):
    python3 /workspace/qwen_fp16.py
"""

from tensorrt_llm import LLM, SamplingParams


def main():
    print("=== Lade Qwen2.5-7B (PyTorch Backend) ===")
    llm = LLM(
        model="Qwen/Qwen2.5-7B-Instruct",
        dtype="float16",
    )

    sp = SamplingParams(temperature=0.7, max_tokens=256)
    prompts = [
        "Erkläre kurz, was eine TensorRT-Engine ist.",
        "Schreibe eine Funktion in Python, die eine Liste sortiert.",
        "Was ist der Unterschied zwischen Prefill und Decode bei autoregressiven Sprachmodellen?",
    ]

    print("\n=== Generiere Ausgaben ===")
    for i, out in enumerate(llm.generate(prompts, sp), 1):
        print(f"\n========== Prompt {i} ==========")
        print(f"Eingabe:  {out.prompt}")
        print(f"\nAntwort:")
        print(out.outputs[0].text)

    print("\n=== Qwen FP16 (PyTorch Backend) abgeschlossen ===")


if __name__ == '__main__':
    main()
