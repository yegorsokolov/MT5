#!/usr/bin/env bash
set -euo pipefail

# Update package list and install core packages
sudo apt-get update
sudo apt-get install -y software-properties-common

echo "Ensuring a supported Python interpreter is installed..."
if ! sudo apt-get install -y python3 python3-venv python3-dev; then
    echo "Warning: Failed to install python3 toolchain packages via apt; attempting to use any existing interpreter." >&2
fi
if apt-cache show python3-distutils >/dev/null 2>&1; then
    if ! sudo apt-get install -y python3-distutils; then
        echo "Warning: python3-distutils could not be installed; continuing without it." >&2
    fi
fi

sudo apt-get install -y build-essential git wine

PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
    for candidate in python3 python3.13 python3.12; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$(command -v "$candidate")"
            break
        fi
    done
fi

if [[ -z "${PYTHON_BIN}" ]]; then
    echo "A supported python3 interpreter is required to bootstrap the environment." >&2
    exit 1
fi

PYTHON_MAJOR=$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.major)')
PYTHON_MINOR=$("${PYTHON_BIN}" -c 'import sys; print(sys.version_info.minor)')
if (( PYTHON_MAJOR < 3 || (PYTHON_MAJOR == 3 && PYTHON_MINOR < 10) )); then
    echo "Python ${PYTHON_MAJOR}.${PYTHON_MINOR} is not supported. Please install Python 3.10 or newer." >&2
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

