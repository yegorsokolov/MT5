#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME=mt5bot
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UPDATE_SERVICE_NAME=mt5bot-update
UPDATE_SERVICE_FILE="/etc/systemd/system/${UPDATE_SERVICE_NAME}.service"
UPDATE_TIMER_FILE="/etc/systemd/system/${UPDATE_SERVICE_NAME}.timer"

# Bootstrap InfluxDB credentials unless explicitly disabled.
if [[ "${SKIP_INFLUX_BOOTSTRAP:-0}" != "1" ]]; then
    "${REPO_DIR}/deploy/bootstrap_influxdb.sh"
fi

# Substitute repository path into unit file and install
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${SERVICE_NAME}.service" | sudo tee "${SERVICE_FILE}" > /dev/null
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${UPDATE_SERVICE_NAME}.service" | sudo tee "${UPDATE_SERVICE_FILE}" > /dev/null
sudo sed "s|{{REPO_PATH}}|${REPO_DIR}|g" "deploy/${UPDATE_SERVICE_NAME}.timer" | sudo tee "${UPDATE_TIMER_FILE}" > /dev/null

sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"
sudo systemctl enable --now "${UPDATE_SERVICE_NAME}.timer"
