#!/usr/bin/env bash
# =============================================================================
# start_trtllm.sh
# Starts the NVIDIA TensorRT-LLM container and manages its lifecycle
# Target system: Ubuntu 24.04 with RTX A6000 Ada (SM89, FP8-capable)
# Author: Ingmar Stapel + AI assistants
# Date: 2026-05-16
# Version 1.0
# Prerequisites: setup_trtllm.sh has been run successfully
# (image pulled, directories created, HF token configured)
#
# Modes:
#   ./start_trtllm.sh           # Start detached (default)
#   ./start_trtllm.sh shell     # Start interactively (--rm, bash)
#   ./start_trtllm.sh exec      # Attach to running container
#   ./start_trtllm.sh stop      # Stop the container
#   ./start_trtllm.sh logs      # Follow container logs
#   ./start_trtllm.sh status    # Show container and GPU status
#
# Idempotent: Safe to run multiple times
# =============================================================================

set -euo pipefail

# === Konfiguration ===
TRTLLM_VERSION="${TRTLLM_VERSION:-1.2.1}"
TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:${TRTLLM_VERSION}"
DATA_DIR="${DATA_DIR:-/data/trtllm}"
CONTAINER_NAME="${CONTAINER_NAME:-trtllm}"
ENV_FILE="${HOME}/.env_trtllm"
PORT="${PORT:-8000}"

# === Farben ===
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()   { echo -e "${GREEN}[ OK ]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }

# === HF Token laden ===
if [[ -f "$ENV_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$ENV_FILE"
fi

# === Funktionen ===

container_exists() {
    docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

container_running() {
    docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"
}

remove_container() {
    if container_exists; then
        log "Stoppe und entferne alten Container..."
        docker stop "$CONTAINER_NAME" &>/dev/null || true
        docker rm "$CONTAINER_NAME" &>/dev/null || true
    fi
}

start_detached() {
    remove_container

    log "Starte $CONTAINER_NAME im Detached-Mode..."
    log "Port-Mapping: ${PORT} (Host) -> 8000 (Container)"
    log "Volume: $DATA_DIR -> /workspace"

    docker run -d \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p "${PORT}:8000" \
        -v "$DATA_DIR":/workspace \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e HF_HOME=/workspace/cache \
        --name "$CONTAINER_NAME" \
        --restart unless-stopped \
        "$TRTLLM_IMAGE" \
        sleep infinity

    ok "Container läuft: $CONTAINER_NAME"
    echo
    echo "Reinkommen:    docker exec -it $CONTAINER_NAME bash"
    echo "Oder kurz:     ./start_trtllm.sh exec"
    echo "Logs:          ./start_trtllm.sh logs"
    echo "Stoppen:       ./start_trtllm.sh stop"
}

start_interactive() {
    remove_container

    log "Starte $CONTAINER_NAME interaktiv (wird bei Exit entfernt)..."

    docker run --rm -it \
        --gpus all \
        --ipc=host \
        --ulimit memlock=-1 \
        --ulimit stack=67108864 \
        -p "${PORT}:8000" \
        -v "$DATA_DIR":/workspace \
        -e HF_TOKEN="${HF_TOKEN:-}" \
        -e HF_HOME=/workspace/cache \
        --name "$CONTAINER_NAME" \
        "$TRTLLM_IMAGE" \
        bash
}

exec_into() {
    if ! container_running; then
        warn "Container läuft nicht. Starte mit: ./start_trtllm.sh"
        exit 1
    fi
    docker exec -it "$CONTAINER_NAME" bash
}

stop_container() {
    if container_running; then
        log "Stoppe $CONTAINER_NAME..."
        docker stop "$CONTAINER_NAME"
        ok "Gestoppt"
    else
        warn "Container läuft nicht"
    fi
}

show_logs() {
    if container_exists; then
        docker logs -f "$CONTAINER_NAME"
    else
        warn "Container existiert nicht"
        exit 1
    fi
}

show_status() {
    echo "=== Container Status ==="
    if container_running; then
        ok "Container läuft"
        docker ps --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
        echo
        echo "=== GPU im Container ==="
        docker exec "$CONTAINER_NAME" nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv
    elif container_exists; then
        warn "Container existiert, läuft aber nicht"
        docker ps -a --filter "name=$CONTAINER_NAME" --format "table {{.Names}}\t{{.Status}}"
    else
        warn "Container existiert nicht"
    fi
}

# === Dispatcher ===
case "${1:-start}" in
    start|"")  start_detached ;;
    shell)     start_interactive ;;
    exec)      exec_into ;;
    stop)      stop_container ;;
    logs)      show_logs ;;
    status)    show_status ;;
    *)
        echo "Usage: $0 [start|shell|exec|stop|logs|status]"
        echo
        echo "  start    Container im Detached-Mode starten (default)"
        echo "  shell    Interaktiv starten (--rm, Exit entfernt Container)"
        echo "  exec     In laufenden Container reinspringen"
        echo "  stop     Container stoppen"
        echo "  logs     Logs verfolgen"
        echo "  status   Status und GPU-Auslastung anzeigen"
        exit 1
        ;;
esac
