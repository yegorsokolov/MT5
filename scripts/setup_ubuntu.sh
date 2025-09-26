#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3-dev build-essential wine

if ! command -v wget >/dev/null 2>&1; then
    sudo apt-get install -y wget
fi

MT5_INSTALL_DIR="${MT5_INSTALL_DIR:-/opt/mt5}"
MT5_DOWNLOAD_URL="${MT5_DOWNLOAD_URL:-https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe}"
MT5_SETUP_PATH="$MT5_INSTALL_DIR/mt5setup.exe"

echo "Preparing MetaTrader 5 terminal under $MT5_INSTALL_DIR ..."
if [ ! -f "$MT5_INSTALL_DIR/terminal64.exe" ] && [ ! -f "$MT5_SETUP_PATH" ]; then
    sudo mkdir -p "$MT5_INSTALL_DIR"
    tmpfile="$(mktemp)"
    echo "Downloading MetaTrader 5 setup from $MT5_DOWNLOAD_URL"
    if wget -O "$tmpfile" "$MT5_DOWNLOAD_URL"; then
        sudo mv "$tmpfile" "$MT5_SETUP_PATH"
        sudo chown root:root "$MT5_SETUP_PATH"
        sudo chmod 755 "$MT5_SETUP_PATH"
        sudo tee "$MT5_INSTALL_DIR/LOGIN_INSTRUCTIONS.txt" >/dev/null <<'EOF'
MetaTrader 5 login instructions
================================

1. Launch the installer once: `wine /opt/mt5/mt5setup.exe`.
2. Complete the platform installation when prompted.
3. Sign in with your broker credentials so historical data can be downloaded.
4. Close the terminal once login succeeds. The training pipeline will reuse
   the authenticated terminal to synchronise price history before training.

If you reinstall MetaTrader 5 elsewhere update MT5_INSTALL_DIR before running
setup_ubuntu.sh.
EOF
        echo "MetaTrader 5 setup downloaded to $MT5_SETUP_PATH."
    else
        echo "Warning: Failed to download MetaTrader 5 setup." >&2
        rm -f "$tmpfile"
    fi
else
    echo "MetaTrader 5 already installed at $MT5_INSTALL_DIR"
fi

# Optionally install CUDA drivers if an NVIDIA GPU is detected or WITH_CUDA=1
if command -v nvidia-smi >/dev/null 2>&1 || [[ "${WITH_CUDA:-0}" == "1" ]]; then
    sudo apt-get install -y nvidia-cuda-toolkit
fi

echo "Checking for outdated Python packages before installation..."
python3 -m pip list --outdated || true

echo "Upgrading pip to the latest version..."
python3 -m pip install --upgrade pip

echo "Installing the latest compatible versions of project dependencies..."
python3 -m pip install --upgrade --upgrade-strategy eager -r requirements-core.txt

echo "Outdated packages remaining after upgrade (if any):"
python3 -m pip list --outdated || true

echo "Installing and starting the MT5 bot service..."
sudo ./scripts/install_service.sh
sudo systemctl status mt5bot

echo "Enabling automatic MT5 bot updates..."
sudo systemctl enable --now mt5bot-update.timer

echo "Triggering an immediate MT5 bot update check..."
python -m services.auto_updater --force
