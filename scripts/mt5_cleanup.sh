#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME=${SERVICE_NAME:-mt5bot}
UPDATE_SERVICE=${UPDATE_SERVICE:-mt5bot-update}
UPDATE_TIMER="${UPDATE_SERVICE}.timer"
INSTALL_ROOT=${INSTALL_ROOT:-/opt/mt5}

echo "Stopping and disabling systemd units..."
sudo systemctl disable --now "${SERVICE_NAME}" 2>/dev/null || true
sudo systemctl disable --now "${UPDATE_SERVICE}" 2>/dev/null || true
sudo systemctl disable --now "${UPDATE_TIMER}" 2>/dev/null || true

echo "Removing unit files..."
sudo rm -f "/etc/systemd/system/${SERVICE_NAME}.service" \
           "/etc/systemd/system/${UPDATE_SERVICE}.service" \
           "/etc/systemd/system/${UPDATE_TIMER}"

sudo systemctl daemon-reload
sudo systemctl reset-failed "${SERVICE_NAME}" "${UPDATE_SERVICE}" "${UPDATE_TIMER}" 2>/dev/null || true

echo "Stopping any lingering Wine processes..."
pkill wineserver 2>/dev/null || true
pkill -f terminal64.exe 2>/dev/null || true
pkill -f python.exe 2>/dev/null || true

echo "Removing repository and generated assets..."
if [[ -d "${INSTALL_ROOT}/deploy/secrets" ]]; then
  sudo rm -rf "${INSTALL_ROOT}/deploy/secrets"
fi
sudo rm -f "${INSTALL_ROOT}/LOGIN_INSTRUCTIONS_WINE.txt" 2>/dev/null || true
sudo rm -rf "${INSTALL_ROOT}/.cache/mt5" 2>/dev/null || true
sudo rm -rf "${INSTALL_ROOT}"

echo "Removing Wine prefixes and caches..."
rm -rf "$HOME/.wine-mt5" "$HOME/.wine-py311" "$HOME/.cache/mt5"

echo "Cleanup complete."
