# tensorrt-llm-edge-prep

Reproducible TensorRT-LLM pipeline on NVIDIA RTX A6000 Ada (SM89) — as practical preparation for the NVIDIA TensorRT Edge-LLM ecosystem.

## What's in here?

- Setup scripts for TRT-LLM on Ubuntu 24.04 with Docker and NGC container
- Build scripts for persistent engines in FP16 and FP8
- Run scripts with timing measurements and output verification
- Companion material to the 4-part blog series on ai-box.eu

## Blog series

1. [Preparation for the Edge-LLM ecosystem](URL)
2. [Setup with Docker and helper scripts](URL)
3. [Persistent engines with FP16 and FP8](URL)
4. [FP16 vs. FP8 — the measurements](URL)

## Quick start

(...)

## Hardware requirements

NVIDIA GPU with Ada architecture or newer (SM89+) for the FP8 paths; for FP16, Ampere (SM86) is sufficient. 48 GB VRAM recommended, 24 GB works for 7B models in FP8.

## License

MIT
