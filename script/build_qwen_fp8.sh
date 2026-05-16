#!/usr/bin/env bash
# =============================================================================
# build_qwen_fp8.sh
# Quantizes Qwen2.5-7B-Instruct to FP8 and builds a persistent TensorRT engine
# Target system: Ubuntu 24.04 with RTX A6000 Ada (SM89, FP8-capable)
# Author: Ingmar Stapel + AI assistants
# Date: 2026-05-16
# Version 1.0
# Prerequisites: setup_trtllm.sh and start_trtllm.sh have been run successfully
# (container running, HF token configured, Qwen2.5-7B-Instruct in HF cache)
#
# Execution context: INSIDE the trtllm container
#   ./start_trtllm.sh exec
#   /workspace/build_qwen_fp8.sh
#
# Workflow (3-stage, PTQ variant):
#   1. HF checkpoint       -> ModelOpt PTQ  -> TRT-LLM checkpoint (FP8-scaled)
#   2. TRT-LLM checkpoint  -> trtllm-build  -> TensorRT engine (FP8)
#   3. Verification + statistics
#
# Activates the Hardware-FP8 Transformer Engine on the Ada architecture (SM89).
#
# Output:
#   /workspace/engines/qwen2.5-7b-fp8/rank0.engine
#   /workspace/engines/qwen2.5-7b-fp8-build.log  (timing statistics)
#
# Counterpart: run_engine_qwen_fp8.py loads this engine
#
# Idempotent: checks if the engine already exists before rebuilding
# =============================================================================

set -euo pipefail

# === Konfiguration ===
MODEL_NAME="Qwen2.5-7B-Instruct"
HF_MODEL_CACHE="/workspace/cache/hub/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
CHECKPOINT_DIR="/workspace/checkpoints/qwen2.5-7b-fp8"
ENGINE_DIR="/workspace/engines/qwen2.5-7b-fp8"
LOG_FILE="/workspace/engines/qwen2.5-7b-fp8-build.log"

# Quantisierungs-Parameter
# WICHTIG: KV_CACHE_DTYPE leer lassen = KV-Cache bleibt bei float16 (default).
# Mögliche Werte: "fp8", "int8", oder leer für native float16.
# Bei 7B-Modellen führt FP8-KV-Cache oft zu komplettem Output-Quality-Collapse.
QFORMAT="fp8"
KV_CACHE_DTYPE=""             # Leer = float16 KV-Cache
CALIB_SIZE=1024
DTYPE="float16"               # Aktivierungs-Dtype, Weights werden FP8
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

format_duration() {
    local s=$1
    printf '%02d:%02d:%02d' $((s/3600)) $((s%3600/60)) $((s%60))
}

# === Kontext-Check ===
if [[ ! -f /.dockerenv ]] && [[ ! -d /app/tensorrt_llm ]]; then
    die "Im trtllm-Container ausführen: ./start_trtllm.sh exec"
fi

# === GPU-FP8-Capability prüfen ===
gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
if [[ "$gpu_name" != *"Ada"* ]] && [[ "$gpu_name" != *"H100"* ]] && \
   [[ "$gpu_name" != *"H200"* ]] && [[ "$gpu_name" != *"L40"* ]] && \
   [[ "$gpu_name" != *"B"* ]]; then
    warn "GPU ($gpu_name) unterstützt vermutlich kein Hardware-FP8."
    warn "FP8 läuft trotzdem, aber ohne Hardware-Beschleunigung — Performance wird nicht beeindrucken."
    read -r -p "Trotzdem fortfahren? (y/N): " confirm
    [[ "${confirm}" == "y" || "${confirm}" == "Y" ]] || exit 0
fi

# === Quantize.py lokalisieren ===
log "Suche quantize.py..."
QUANTIZE_SCRIPT=""
for candidate in \
    "/app/tensorrt_llm/examples/quantization/quantize.py" \
    "/app/tensorrt_llm/examples/models/core/qwen/../../../quantization/quantize.py" \
    "/app/tensorrt_llm/examples/models/quantization/quantize.py"; do
    if [[ -f "$candidate" ]]; then
        QUANTIZE_SCRIPT="$(realpath "$candidate")"
        break
    fi
done

if [[ -z "$QUANTIZE_SCRIPT" ]]; then
    log "Standard-Pfade nicht gefunden, breite Suche..."
    QUANTIZE_SCRIPT=$(find /app/tensorrt_llm/examples -name "quantize.py" -type f 2>/dev/null | head -1)
fi

if [[ -z "$QUANTIZE_SCRIPT" ]] || [[ ! -f "$QUANTIZE_SCRIPT" ]]; then
    die "quantize.py nicht gefunden — examples-Struktur unklar.
        Manuell suchen: find /app/tensorrt_llm -name 'quantize.py'"
fi
ok "Quantize-Skript: $QUANTIZE_SCRIPT"

# === Banner ===
echo
echo "=================================================="
echo -e "  ${CYAN}TensorRT FP8 Build: ${MODEL_NAME}${NC}"
echo "  GPU:        $gpu_name"
echo "  QFormat:    $QFORMAT"
echo "  KV-Cache:   ${KV_CACHE_DTYPE:-float16 (native)}"
echo "  Calib-Size: $CALIB_SIZE Samples"
echo "  Target:     $ENGINE_DIR"
echo "=================================================="
echo

# === Idempotenz-Check ===
if [[ -f "${ENGINE_DIR}/rank0.engine" ]]; then
    warn "FP8-Engine existiert bereits:"
    ls -lh "${ENGINE_DIR}/"
    read -r -p "Neu bauen? (y/N): " confirm
    [[ "${confirm}" == "y" || "${confirm}" == "Y" ]] || exit 0
    log "Lösche alte Artefakte..."
    rm -rf "${ENGINE_DIR}" "${CHECKPOINT_DIR}"
fi

# === HF-Modell finden ===
log "Suche HF-Modell im Cache..."
QWEN_HF=$(find "${HF_MODEL_CACHE}" -mindepth 1 -maxdepth 1 -type d | head -1)
[[ -n "$QWEN_HF" ]] || die "HF-Modell nicht im Cache. Erst qwen_fp16.py laufen lassen."
[[ -f "${QWEN_HF}/config.json" ]] || die "config.json fehlt in ${QWEN_HF}"
ok "HF-Pfad: ${QWEN_HF}"

# === Verzeichnisse ===
mkdir -p "$(dirname "${CHECKPOINT_DIR}")"
mkdir -p "$(dirname "${ENGINE_DIR}")"

# === STUFE 1: ModelOpt PTQ ===
echo
echo "=================================================="
log "STUFE 1: FP8 Post-Training Quantization (PTQ)"
log "Geschätzte Dauer: 5-15 Minuten (abhängig von Calib-Size)"
log "Was passiert: Modell läuft auf ${CALIB_SIZE} Kalibrierungs-Samples,"
log "ModelOpt ermittelt pro Layer optimale FP8-Skalierungsfaktoren."
echo "=================================================="

start_quant=$(date +%s)

# Argumente konditionell zusammenbauen — kv_cache_dtype nur weitergeben,
# wenn explizit auf "fp8" oder "int8" gesetzt; leer = native float16
QUANTIZE_ARGS=(
    --model_dir "${QWEN_HF}"
    --output_dir "${CHECKPOINT_DIR}"
    --dtype "${DTYPE}"
    --qformat "${QFORMAT}"
    --calib_size "${CALIB_SIZE}"
)
if [[ -n "$KV_CACHE_DTYPE" ]]; then
    QUANTIZE_ARGS+=(--kv_cache_dtype "$KV_CACHE_DTYPE")
    log "KV-Cache wird nach $KV_CACHE_DTYPE quantisiert"
else
    log "KV-Cache bleibt bei native float16 (nicht quantisiert)"
fi

python3 "$QUANTIZE_SCRIPT" "${QUANTIZE_ARGS[@]}"

end_quant=$(date +%s)
quant_seconds=$((end_quant - start_quant))
quant_formatted=$(format_duration "${quant_seconds}")

ok "Quantize fertig in ${quant_formatted} (${quant_seconds}s)"
log "Checkpoint-Inhalt:"
ls -lh "${CHECKPOINT_DIR}/"

# === STUFE 2: Build TensorRT Engine mit FP8-Plugins ===
echo
echo "=================================================="
log "STUFE 2: Build TensorRT Engine (FP8)"
log "Aktiviert FP8 Context FMHA (war beim FP16-Build deaktiviert)"
log "Tipp: Zweites Terminal mit 'watch -n 1 nvidia-smi'"
echo "=================================================="

start_build=$(date +%s)

trtllm-build \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --output_dir "${ENGINE_DIR}" \
    --gemm_plugin auto \
    --use_fp8_context_fmha enable \
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

[[ -f "${ENGINE_DIR}/rank0.engine" ]] || die "rank0.engine fehlt"

engine_size_h=$(du -h "${ENGINE_DIR}/rank0.engine" | cut -f1)
total_size_h=$(du -sh "${ENGINE_DIR}" | cut -f1)
checkpoint_size_h=$(du -sh "${CHECKPOINT_DIR}" | cut -f1)

ok "Engine erfolgreich gebaut:"
ls -lh "${ENGINE_DIR}/"

# === Statistik ===
total_seconds=$((quant_seconds + build_seconds))
total_formatted=$(format_duration "${total_seconds}")

echo
echo -e "${CYAN}=================================================="
echo "  Build-Statistik FP16 (für Interview-Tabelle)"
echo -e "==================================================${NC}"
printf "  %-22s %s\n" "Modell:"          "${MODEL_NAME}"
printf "  %-22s %s\n" "Präzision:"       "FP8 (Weights + KV-Cache)"
printf "  %-22s %s\n" "Activations:"     "${DTYPE}"
printf "  %-22s %s\n" "GPU:"             "${gpu_name}"
printf "  %-22s %s\n" "Quantize-Zeit:"   "${quant_formatted} (${quant_seconds}s)"
printf "  %-22s %s\n" "Build-Zeit:"      "${build_formatted} (${build_seconds}s)"
printf "  %-22s %s\n" "Gesamt-Zeit:"     "${total_formatted} (${total_seconds}s)"
printf "  %-22s %s\n" "Checkpoint:"      "${checkpoint_size_h}"
printf "  %-22s %s\n" "Engine-Datei:"    "${engine_size_h}"
printf "  %-22s %s\n" "Engine-Verz.:"    "${total_size_h}"
printf "  %-22s %s\n" "Pfad:"            "${ENGINE_DIR}"
echo -e "${CYAN}==================================================${NC}"
echo

# === Log persistieren ===
{
    echo "=== FP8 Build $(date -Iseconds) ==="
    echo "Modell:        ${MODEL_NAME}"
    echo "Präzision:     FP8 weights + ${KV_CACHE_DTYPE} KV-cache"
    echo "GPU:           ${gpu_name}"
    echo "Calib-Size:    ${CALIB_SIZE}"
    echo "Quantize-Zeit: ${quant_seconds}s (${quant_formatted})"
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
log "Nächster Schritt: FP8-Engine testen mit run_engine_qwen_fp8.py"
echo "    Vergleichswerte aus FP16-Lauf:"
echo "      Engine-Größe:     14.5 GB"
echo "      Engine-Load:      ~13 s"
echo "      Tokens/sec:       ~154 (batched)"
echo "    Erwartung FP8:"
echo "      Engine-Größe:     ~8 GB (44% kleiner)"
echo "      Tokens/sec:       1,4-1,8x höher dank Hardware-FP8"
echo
