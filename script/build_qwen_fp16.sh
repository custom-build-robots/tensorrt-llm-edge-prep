#!/usr/bin/env bash
# =============================================================================
# build_qwen_fp16.sh
# Builds a persistent TensorRT engine for Qwen2.5-7B-Instruct in FP16
# Target system: Ubuntu 24.04 with RTX A6000 Ada (SM89, FP8-capable)
# Author: Ingmar Stapel + AI assistants
# Date: 2026-05-16
# Version 1.0
# Prerequisites: setup_trtllm.sh and start_trtllm.sh have been run successfully
# (container running, HF token configured, Qwen2.5-7B-Instruct in HF cache)
#
# Execution context: INSIDE the trtllm container
#   ./start_trtllm.sh exec
#   /workspace/build_qwen_fp16.sh
#
# Workflow (classic TRT-LLM pipeline):
#   1. HF checkpoint        -> TRT-LLM checkpoint  (convert_checkpoint.py)
#   2. TRT-LLM checkpoint   -> TensorRT engine     (trtllm-build)
#
# Output:
#   /workspace/engines/qwen2.5-7b-fp16/rank0.engine
#   /workspace/engines/qwen2.5-7b-fp16-build.log  (timing statistics)
#
# Idempotent: checks if the engine already exists before rebuilding
# =============================================================================

set -euo pipefail

# === Konfiguration ===
MODEL_NAME="Qwen2.5-7B-Instruct"
HF_MODEL_CACHE="/workspace/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
CHECKPOINT_DIR="/workspace/checkpoints/qwen2.5-7b-fp16"
ENGINE_DIR="/workspace/engines/qwen2.5-7b-fp16"
LOG_FILE="/workspace/engines/qwen2.5-7b-fp16-build.log"
EXAMPLE_DIR="/app/tensorrt_llm/examples/models/core/qwen"

DTYPE="float16"
GEMM_PLUGIN="float16"
MAX_BATCH_SIZE=4
MAX_SEQ_LEN=4096

# === Farben ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR ]${NC} $*" >&2; }
die()  { err "$*"; exit 1; }

# Sekunden in lesbares h:m:s Format
format_duration() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $((s%3600/60)) $((s%60))
}

# === Kontext-Check ===
if [[ ! -f /.dockerenv ]] && [[ ! -d /app/tensorrt_llm ]]; then
    die "Dieses Skript muss INNERHALB des trtllm-Containers laufen.
        Erst:  ./start_trtllm.sh exec
        Dann:  /workspace/build_qwen_fp16.sh"
fi

if ! command -v trtllm-build &>/dev/null; then
    die "trtllm-build nicht im PATH — falsche Container-Version?"
fi

# === Banner ===
echo
echo "=================================================="
echo -e "  ${CYAN}TensorRT Engine Build: ${MODEL_NAME}${NC}"
echo "  Präzision: ${DTYPE}"
echo "  Target:    ${ENGINE_DIR}"
echo "=================================================="
echo

# === Idempotenz-Check ===
if [[ -f "${ENGINE_DIR}/rank0.engine" ]]; then
    warn "Engine existiert bereits:"
    ls -lh "${ENGINE_DIR}/"
    echo
    read -r -p "Trotzdem neu bauen? (y/N): " confirm
    if [[ "${confirm}" != "y" && "${confirm}" != "Y" ]]; then
        log "Abbruch — Engine bleibt unverändert"
        exit 0
    fi
    log "Lösche alten Build..."
    rm -rf "${ENGINE_DIR}" "${CHECKPOINT_DIR}"
fi

# === HF-Modell im Cache finden ===
log "Suche gecachtes HF-Modell..."
if [[ ! -d "${HF_MODEL_CACHE}" ]]; then
    die "HF-Modell nicht im Cache unter ${HF_MODEL_CACHE}.
        Erst qwen_fp16.py laufen lassen, damit das Modell heruntergeladen wird."
fi

QWEN_HF=$(find "${HF_MODEL_CACHE}" -mindepth 1 -maxdepth 1 -type d | head -1)
if [[ -z "${QWEN_HF}" ]]; then
    die "Kein Snapshot-Verzeichnis gefunden in ${HF_MODEL_CACHE}"
fi
ok "HF-Pfad: ${QWEN_HF}"

# Sanity: Hat es config.json?
if [[ ! -f "${QWEN_HF}/config.json" ]]; then
    die "Kein config.json in ${QWEN_HF} — Cache scheint kaputt"
fi

# Sanity: GPU sichtbar?
log "GPU-Check:"
nvidia-smi --query-gpu=name,memory.free,memory.total --format=csv,noheader
echo

# === Verzeichnisse anlegen ===
mkdir -p "$(dirname "${CHECKPOINT_DIR}")"
mkdir -p "$(dirname "${ENGINE_DIR}")"

# === STUFE 1: Convert HF -> TRT-LLM Checkpoint ===
echo
echo "=================================================="
log "STUFE 1: HF Checkpoint -> TRT-LLM Checkpoint"
log "Geschätzte Dauer: 2-3 Minuten"
echo "=================================================="

start_convert=$(date +%s)

# In Subshell wegen cd
(
    cd "${EXAMPLE_DIR}"
    python3 convert_checkpoint.py \
        --model_dir "${QWEN_HF}" \
        --output_dir "${CHECKPOINT_DIR}" \
        --dtype "${DTYPE}"
)

end_convert=$(date +%s)
convert_seconds=$((end_convert - start_convert))
convert_formatted=$(format_duration "${convert_seconds}")

ok "Convert fertig in ${convert_formatted} (${convert_seconds}s)"
log "Checkpoint-Inhalt:"
ls -lh "${CHECKPOINT_DIR}/"

# === STUFE 2: Build TensorRT Engine ===
echo
echo "=================================================="
log "STUFE 2: TRT-LLM Checkpoint -> TensorRT Engine"
log "Geschätzte Dauer: 5-10 Minuten (Kernel-Auto-Tuning)"
log "Tipp: In zweitem Terminal 'watch -n 1 nvidia-smi' mitlaufen lassen"
echo "=================================================="

start_build=$(date +%s)

trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${ENGINE_DIR}" \
    --gemm_plugin "${GEMM_PLUGIN}" \
    --max_batch_size "${MAX_BATCH_SIZE}" \
    --max_seq_len "${MAX_SEQ_LEN}"

end_build=$(date +%s)
build_seconds=$((end_build - start_build))
build_formatted=$(format_duration "${build_seconds}")

ok "Build fertig in ${build_formatted} (${build_seconds}s)"

# === STUFE 3: Verifikation ===
echo
echo "=================================================="
log "STUFE 3: Verifikation"
echo "=================================================="

if [[ ! -f "${ENGINE_DIR}/rank0.engine" ]]; then
    die "rank0.engine wurde NICHT erzeugt — Build hat formal funktioniert, aber Output fehlt"
fi

engine_size_bytes=$(stat -c%s "${ENGINE_DIR}/rank0.engine")
engine_size_h=$(du -h "${ENGINE_DIR}/rank0.engine" | cut -f1)
total_size_h=$(du -sh "${ENGINE_DIR}" | cut -f1)
checkpoint_size_h=$(du -sh "${CHECKPOINT_DIR}" | cut -f1)

ok "Engine erfolgreich erzeugt:"
ls -lh "${ENGINE_DIR}/"

# === Build-Statistik ===
total_seconds=$((convert_seconds + build_seconds))
total_formatted=$(format_duration "${total_seconds}")
timestamp=$(date -Iseconds)
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)

echo
echo -e "${CYAN}=================================================="
echo "  Build-Statistik (für Interview-Tabelle)"
echo -e "==================================================${NC}"
printf "  %-20s %s\n" "Modell:"          "${MODEL_NAME}"
printf "  %-20s %s\n" "Präzision:"       "${DTYPE}"
printf "  %-20s %s\n" "GPU:"             "${gpu_name}"
printf "  %-20s %s\n" "Convert-Zeit:"    "${convert_formatted} (${convert_seconds}s)"
printf "  %-20s %s\n" "Build-Zeit:"      "${build_formatted} (${build_seconds}s)"
printf "  %-20s %s\n" "Gesamt-Zeit:"     "${total_formatted} (${total_seconds}s)"
printf "  %-20s %s\n" "Checkpoint:"      "${checkpoint_size_h}"
printf "  %-20s %s\n" "Engine-Datei:"    "${engine_size_h}"
printf "  %-20s %s\n" "Engine-Verz.:"    "${total_size_h}"
printf "  %-20s %s\n" "Pfad:"            "${ENGINE_DIR}"
echo -e "${CYAN}==================================================${NC}"
echo

# === Log persistieren (Append-Mode für historische Vergleiche) ===
{
    echo "=== Build $(date -Iseconds) ==="
    echo "Modell:        ${MODEL_NAME}"
    echo "Präzision:     ${DTYPE}"
    echo "GPU:           ${gpu_name}"
    echo "Convert-Zeit:  ${convert_seconds}s (${convert_formatted})"
    echo "Build-Zeit:    ${build_seconds}s (${build_formatted})"
    echo "Gesamt-Zeit:   ${total_seconds}s (${total_formatted})"
    echo "Checkpoint:    ${checkpoint_size_h}"
    echo "Engine-Datei:  ${engine_size_h}"
    echo "Engine-Verz.:  ${total_size_h}"
    echo "Pfad:          ${ENGINE_DIR}"
    echo
} >> "${LOG_FILE}"

ok "Statistik geloggt nach: ${LOG_FILE}"
echo

log "Nächster Schritt: Engine mit Python laden und Tokens generieren"
echo "    Beispiel:"
echo "      from tensorrt_llm import LLM, SamplingParams"
echo "      llm = LLM(model=\"${ENGINE_DIR}\")"
echo "      out = llm.generate([\"Hallo\"], SamplingParams(max_tokens=50))"
echo
