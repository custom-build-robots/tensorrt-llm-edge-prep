"""
smoke.py
Smoke test for NVIDIA TensorRT-LLM with TinyLlama-1.1B-Chat
Target system: Ubuntu 24.04 with RTX A6000 Ada (SM89, FP8-capable)
Author: Ingmar Stapel + AI assistants
Date: 2026-05-16
Version 1.0
Prerequisites: setup_trtllm.sh and start_trtllm.sh have been run successfully
(container running, HF token configured)

Purpose:
    Validates that the TensorRT-LLM installation works end-to-end.
    Tests model download, engine build, and basic inference.

Details:
    - Model:        TinyLlama-1.1B-Chat (~2 GB download on first run)
    - Engine build: ~1-2 minutes on first run, cached afterwards
    - Runtime:      3-5 minutes on first run, <10 seconds afterwards

Important:
    The `if __name__ == '__main__':` guard is mandatory because
    TensorRT-LLM internally spawns MPI worker processes.

Usage (inside the container):
    python3 /workspace/smoke.py
"""

from tensorrt_llm import LLM, SamplingParams


def main():
    print("=== Lade Modell und baue Engine (kann beim ersten Mal dauern) ===")
    llm = LLM(model="TinyLlama/TinyLlama-1.1B-Chat-v1.0")

    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=64)

    prompts = [
        "Hallo, ich heiße",
        "Souveräne KI bedeutet",
    ]

    print("\n=== Generiere Ausgaben ===")
    for out in llm.generate(prompts, sp):
        print(f"\nPrompt:    {out.prompt!r}")
        print(f"Generated: {out.outputs[0].text!r}")

    print("\n=== Smoke Test erfolgreich abgeschlossen ===")


if __name__ == '__main__':
    main()
