#!/usr/bin/env bash
set -euo pipefail

# Update package list and install core packages
sudo apt-get update
sudo apt-get install -y python3-dev build-essential git wine

# Install NVIDIA drivers if a GPU is detected
if command -v nvidia-smi >/dev/null 2>&1; then
    sudo apt-get install -y nvidia-driver || true
fi

# Install Python dependencies
python3 -m pip install --upgrade pip
pip install -r requirements-core.txt

# Post-install checks
python3 - <<'PYTHON'
import MetaTrader5 as mt5
if not mt5.initialize():
    raise SystemExit("MetaTrader5 initialization failed")
mt5.shutdown()
PYTHON

wine --version >/dev/null

python3 - <<'PYTHON'
import brokers
print("Broker dependencies loaded successfully")
PYTHON

