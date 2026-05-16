#!/bin/bash
###############################################################################
#  Ubuntu Server – Grundinstallation
#  Datum: 2026-04-05
#  Version: 2.2
#  Beschreibung: Installiert Basissystem, NVIDIA CUDA, Docker,
#                NVIDIA Container Toolkit (ohne Samba).
#  Info des Autors: Bitte das Skript vor dem Start einmal durchlesen und
#  ggf. Versionen (CUDA, NVIDIA Toolkit) an aktuelle Werte anpassen.
###############################################################################

set -euo pipefail
export DEBIAN_FRONTEND=noninteractive

# ── Farben ───────────────────────────────────────────────────────────────────
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${GREEN}[✔] $1${NC}"; }
warn() { echo -e "${YELLOW}[!] $1${NC}"; }
err()  { echo -e "${RED}[✘] $1${NC}"; exit 1; }

# ── Root-Check ───────────────────────────────────────────────────────────────
if [[ $EUID -eq 0 ]]; then
  err "Bitte NICHT als root ausführen – das Script nutzt sudo wo nötig."
fi

SETUP_USER="$USER"
SETUP_HOME="$HOME"
log "Setup läuft als Benutzer: ${SETUP_USER} (Home: ${SETUP_HOME})"

###############################################################################
# 1. System aktualisieren
###############################################################################
log "System-Update & Upgrade …"
sudo apt-get update
sudo apt-get upgrade -y

###############################################################################
# 2. OpenSSH-Server
###############################################################################
log "OpenSSH-Server installieren …"
sudo apt-get install -y openssh-server
sudo systemctl enable --now ssh

###############################################################################
# 3. Grundlegende Pakete
###############################################################################
log "Grundpakete installieren …"
sudo apt-get install -y \
  ca-certificates \
  curl \
  gnupg2 \
  wget \
  python3-venv \
  git \
  mc \
  net-tools

###############################################################################
# 4. NVIDIA CUDA Toolkit & Treiber
###############################################################################
CUDA_KEYRING_URL="https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb"
CUDA_KEYRING_DEB="/tmp/cuda-keyring.deb"

log "CUDA-Keyring herunterladen & installieren …"
wget -q -O "$CUDA_KEYRING_DEB" "$CUDA_KEYRING_URL"
sudo dpkg -i "$CUDA_KEYRING_DEB"
rm -f "$CUDA_KEYRING_DEB"

sudo apt-get update

log "CUDA Toolkit 13.1 installieren …"
sudo apt-get install -y cuda-toolkit-13-1

log "NVIDIA-Treiber installieren …"
sudo apt-get install -y cuda-drivers

###############################################################################
# 5. Docker (offizielle Installationsmethode)
###############################################################################
log "Docker installieren …"

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
  | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) \
  signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
  | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

sudo apt-get update
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin

sudo groupadd -f docker
sudo usermod -aG docker "$SETUP_USER"
log "Docker installiert – Gruppenmitgliedschaft wird nach Neustart/Login aktiv."

###############################################################################
# 6. NVIDIA Container Toolkit
###############################################################################
NVIDIA_CTK_VERSION="1.19.0-1"

log "NVIDIA Container Toolkit (v${NVIDIA_CTK_VERSION}) installieren …"

curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null

sudo apt-get update
sudo apt-get install -y "nvidia-container-toolkit=${NVIDIA_CTK_VERSION}"

sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

###############################################################################
# Fertig
###############################################################################
echo ""
log "════════════════════════════════════════════════════════════"
log "  Server-Grundinstallation abgeschlossen!"
log "════════════════════════════════════════════════════════════"
warn "Bitte den Server jetzt neu starten:  sudo reboot"
warn "Nach dem Reboot prüfen:"
warn "  • nvidia-smi                (GPU & Treiber)"
warn "  • docker run hello-world    (Docker ohne sudo)"
warn "  • nvidia-ctk --version      (Container Toolkit)"
echo ""
