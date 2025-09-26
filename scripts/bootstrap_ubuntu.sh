#!/usr/bin/env bash
set -euo pipefail

# Update package list and install core packages
sudo apt-get update
sudo apt-get install -y software-properties-common

if ! command -v python3.13 >/dev/null 2>&1; then
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update
fi

sudo apt-get install -y python3.13 python3.13-venv python3.13-dev python3.13-distutils build-essential git wine

PYTHON_BIN="${PYTHON_BIN:-$(command -v python3.13)}"

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "python3.13 is required to bootstrap the environment" >&2
    exit 1
fi

# Install NVIDIA drivers if a GPU is detected
if command -v nvidia-smi >/dev/null 2>&1; then
    sudo apt-get install -y nvidia-driver || true
fi

# Install Python dependencies
"${PYTHON_BIN}" -m pip install --upgrade pip
"${PYTHON_BIN}" -m pip install -r requirements.txt

# Post-install checks
"${PYTHON_BIN}" - <<'PYTHON'
import MetaTrader5 as mt5
if not mt5.initialize():
    raise SystemExit("MetaTrader5 initialization failed")
mt5.shutdown()
PYTHON

wine --version >/dev/null

"${PYTHON_BIN}" - <<'PYTHON'
import brokers
print("Broker dependencies loaded successfully")
PYTHON

