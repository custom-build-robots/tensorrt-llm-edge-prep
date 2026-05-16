#!/usr/bin/env bash
# =============================================================================
# setup_trtllm.sh
# One-time installation of NVIDIA TensorRT-LLM
# Target system: Ubuntu 24.04 with RTX A6000 Ada (SM89, FP8-capable)
# Author: Ingmar Stapel + AI assistants
# Date: 2026-05-16
# Version 1.0
# Prerequisites: NVIDIA driver, Docker, NVIDIA Container Toolkit
# (all already installed by server_setup.sh)
#
# Idempotent: Safe to run multiple times
# =============================================================================

set -euo pipefail

# === Konfiguration ===
TRTLLM_VERSION="${TRTLLM_VERSION:-1.2.1}"
TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:${TRTLLM_VERSION}"
DATA_DIR="${DATA_DIR:-/data/trtllm}"
ENV_FILE="${HOME}/.env_trtllm"
MIN_DRIVER_MAJOR=545

# === Farben ===
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
err()  { echo -e "${RED}[ERR ]${NC} $*" >&2; }
die()  { err "$*"; exit 1; }

# === Schritt 0: Sanity Check ===
check_prerequisites() {
    log "Prüfe Vorbedingungen..."

    # nvidia-smi
    if ! command -v nvidia-smi &>/dev/null; then
        die "nvidia-smi nicht gefunden — NVIDIA-Treiber installieren"
    fi

    local driver_version driver_full
    driver_full=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)
    driver_version=$(echo "$driver_full" | cut -d. -f1)
    if [[ "$driver_version" -lt "$MIN_DRIVER_MAJOR" ]]; then
        die "Treiber zu alt ($driver_full), brauche >= $MIN_DRIVER_MAJOR"
    fi
    ok "NVIDIA-Treiber: $driver_full"

    # GPU
    local gpu_name
    gpu_name=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    ok "GPU: $gpu_name"

    # FP8-Hinweis
    if [[ "$gpu_name" == *"Ada"* ]] || [[ "$gpu_name" == *"H100"* ]] || [[ "$gpu_name" == *"L40"* ]]; then
        ok "GPU unterstützt FP8 (Hardware Transformer Engine)"
    else
        warn "GPU unterstützt vermutlich kein Hardware-FP8 — INT4/FP16 nutzen"
    fi

    # Docker
    if ! command -v docker &>/dev/null; then
        die "Docker nicht gefunden"
    fi
    if ! docker ps &>/dev/null; then
        die "Docker läuft nicht oder fehlende Rechte (Nutzer in docker-Gruppe?)"
    fi
    ok "Docker: $(docker --version | cut -d, -f1)"

    # GPU-Passthrough in Container
    log "Teste GPU-Passthrough in Container..."
    if ! docker run --rm --gpus all ubuntu:24.04 nvidia-smi -L &>/dev/null; then
        die "GPU nicht in Container sichtbar — NVIDIA Container Toolkit installiert?"
    fi
    ok "GPU im Container sichtbar"

    # Festplattenplatz
    local free_gb
    free_gb=$(df -BG /var/lib/docker 2>/dev/null | awk 'NR==2 {print $4}' | sed 's/G//' || echo "0")
    if [[ "$free_gb" -lt 50 ]]; then
        warn "Nur ${free_gb} GB frei auf /var/lib/docker — Image ist 20 GB, Modelle weitere 30-50 GB"
    else
        ok "Freier Speicher: ${free_gb} GB"
    fi
}

# === Schritt 1: HuggingFace Token ===
setup_hf_token() {
    log "HuggingFace Token Setup"

    # Bereits in der Umgebung gesetzt?
    if [[ -n "${HF_TOKEN:-}" ]]; then
        ok "HF_TOKEN ist in Umgebung gesetzt"
        echo "export HF_TOKEN=\"${HF_TOKEN}\"" > "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        return
    fi

    # Datei existiert?
    if [[ -f "$ENV_FILE" ]]; then
        ok "Token-Datei existiert: $ENV_FILE"
        # shellcheck source=/dev/null
        source "$ENV_FILE"
        return
    fi

    # Interaktive Abfrage
    warn "Kein HF_TOKEN gefunden"
    echo
    echo "  Token holen: https://huggingface.co/settings/tokens"
    echo "  Read-Permission reicht. Für Llama-Modelle vorher Lizenz akzeptieren."
    echo "  (Qwen2.5 und TinyLlama brauchen keinen Token.)"
    echo
    read -r -p "  HF Token (oder Enter zum Überspringen): " token

    if [[ -n "$token" ]]; then
        echo "export HF_TOKEN=\"${token}\"" > "$ENV_FILE"
        chmod 600 "$ENV_FILE"
        ok "Token gespeichert in $ENV_FILE"
        log "Empfehlung: In ~/.bashrc ergänzen:"
        echo "    [[ -f ~/.env_trtllm ]] && source ~/.env_trtllm"
    else
        warn "Übersprungen — kann später nachgeholt werden"
    fi
}

# === Schritt 2: Verzeichnisstruktur ===
setup_directories() {
    log "Verzeichnisstruktur"

    if [[ ! -d "$DATA_DIR" ]]; then
        sudo mkdir -p "$DATA_DIR"/{models,engines,cache}
        sudo chown -R "$USER:$USER" "$DATA_DIR"
        ok "Angelegt: $DATA_DIR/{models,engines,cache}"
    else
        ok "Verzeichnisse existieren: $DATA_DIR"
    fi
}

# === Schritt 3: Container ziehen ===
pull_container() {
    log "TensorRT-LLM Container ($TRTLLM_VERSION)"

    if docker image inspect "$TRTLLM_IMAGE" &>/dev/null; then
        ok "Image bereits lokal vorhanden: $TRTLLM_IMAGE"
        return
    fi

    log "Lade ca. 20 GB ... das dauert (Kaffeezeit)"
    if docker pull "$TRTLLM_IMAGE"; then
        ok "Image geladen: $TRTLLM_IMAGE"
    else
        die "Pull fehlgeschlagen"
    fi
}

# === Hauptablauf ===
main() {
    echo "=================================================="
    echo "  TensorRT-LLM Setup"
    echo "  Version: $TRTLLM_VERSION"
    echo "  Datenverzeichnis: $DATA_DIR"
    echo "=================================================="
    echo

    check_prerequisites
    echo
    setup_hf_token
    echo
    setup_directories
    echo
    pull_container
    echo

    echo "=================================================="
    ok "Setup abgeschlossen!"
    echo "=================================================="
    echo
    echo "Nächste Schritte:"
    echo "  1. Container starten:    ./start_trtllm.sh"
    echo "  2. In Container rein:    docker exec -it trtllm bash"
    echo "  3. Smoke-Test:           python3 /workspace/smoke.py"
    echo
}

main "$@"
