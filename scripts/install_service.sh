#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME=mt5bot
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Substitute repository path into unit file and install
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${SERVICE_NAME}.service" | sudo tee "${SERVICE_FILE}" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
