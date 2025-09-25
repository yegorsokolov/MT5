#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3-dev build-essential wine

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
