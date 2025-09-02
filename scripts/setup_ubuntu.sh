#!/usr/bin/env bash
set -euo pipefail

sudo apt-get update
sudo apt-get install -y python3-dev build-essential wine

# Optionally install CUDA drivers if an NVIDIA GPU is detected or WITH_CUDA=1
if command -v nvidia-smi >/dev/null 2>&1 || [[ "${WITH_CUDA:-0}" == "1" ]]; then
    sudo apt-get install -y nvidia-cuda-toolkit
fi

python3 -m pip install --upgrade pip
pip install -r requirements-core.txt
