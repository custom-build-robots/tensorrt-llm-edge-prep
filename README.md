# tensorrt-llm-edge-prep
Reproducible TensorRT-LLM pipeline on NVIDIA RTX A6000 Ada (SM89) — as practical preparation for the NVIDIA TensorRT Edge-LLM ecosystem.

<img src="https://ai-box.eu/wp-content/uploads/2026/05/NVIDIA_RTX_A6000_ADA-1280x640.jpg" alt="NVIDIA RTX A6000 Ada" width="400">

## What's in here?

- Setup scripts for TRT-LLM on Ubuntu 24.04 with Docker and NGC container
- Build scripts for persistent engines in FP16 and FP8
- Run scripts with timing measurements and output verification
- Companion material to the blog series on ai-box.eu (in German)

## Blog series

Before you start, prepare your server with the foundation post:

- [Ubuntu 24.04 Server für KI-Inferenz vorbereiten: CUDA, Docker, NVIDIA Container Toolkit](https://ai-box.eu/hardware/ubuntu-24-04-server-fuer-ki-inferenz-vorbereiten-cuda-docker-nvidia-container-toolkit/2230/)

Then follow the four-part TensorRT-LLM series:

1. [TensorRT-LLM auf der RTX A6000 Ada: Vorbereitung auf das Edge-LLM Ökosystem](https://ai-box.eu/large-language-models/tensorrt-llm-auf-der-rtx-a6000-ada-vorbereitung-auf-das-edge-llm-oekosystem/2216/)
2. [TensorRT-LLM auf Ubuntu 24.04: Setup mit Docker und Helper-Skripten](https://ai-box.eu/large-language-models/tensorrt-llm-auf-ubuntu-24-04-setup-mit-docker-und-helper-skripten/2219/)
3. [TensorRT-LLM Pipeline: Persistente Engines bauen mit FP16 und FP8](https://ai-box.eu/large-language-models/tensorrt-llm-pipeline-persistente-engines-bauen-mit-fp16-und-fp8/2221/)
4. [TensorRT-LLM in Zahlen: FP16 vs. FP8 auf der RTX A6000 Ada](https://ai-box.eu/large-language-models/tensorrt-llm-in-zahlen-fp16-vs-fp8-auf-der-rtx-a6000-ada/2223/)

## Quick start

1. Prepare the server using the foundation post linked above
2. Run `setup_trtllm.sh` once to pull the NGC container and configure paths
3. Use `start_trtllm.sh exec` to enter the container
4. Run `qwen_fp16.py` to download Qwen2.5-7B-Instruct into the HF cache
5. Build a persistent engine: `build_qwen_fp16.sh` or `build_qwen_fp8.sh`
6. Test the engine: `run_engine_qwen_fp16.py` or `run_engine_qwen_fp8.py`

## Hardware requirements

NVIDIA GPU with Ada architecture or newer (SM89+) for the FP8 paths; for FP16, Ampere (SM86) is sufficient. 48 GB VRAM recommended, 24 GB works for 7B models in FP8.

## License

MIT
