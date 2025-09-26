#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}/.."

cd "${PROJECT_ROOT}"

PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
PYTHON_BIN="/usr/bin/python${PYTHON_VERSION}"

sudo apt-get update
sudo apt-get install -y software-properties-common

install_python_from_apt() {
    sudo apt-get install -y \
        "python${PYTHON_VERSION}" \
        "python${PYTHON_VERSION}-dev" \
        "python${PYTHON_VERSION}-venv" \
        "python${PYTHON_VERSION}-distutils"
}

ensure_python_version() {
    if command -v "python${PYTHON_VERSION}" >/dev/null 2>&1; then
        return 0
    fi

    echo "Python ${PYTHON_VERSION} is not installed; attempting installation from default repositories..."
    if install_python_from_apt; then
        return 0
    fi

    # Capture the distribution codename in a best-effort manner. Older Ubuntu
    # releases ship `lsb_release` while newer minimal images might only expose
    # the data inside /etc/os-release.
    distro_codename=""
    if command -v lsb_release >/dev/null 2>&1; then
        distro_codename="$(lsb_release -cs 2>/dev/null || true)"
    fi
    if [[ -z "${distro_codename}" && -r /etc/os-release ]]; then
        # shellcheck disable=SC1091
        . /etc/os-release
        distro_codename="${UBUNTU_CODENAME:-${VERSION_CODENAME:-}}"
    fi

    supported_codenames=("focal" "jammy" "lunar" "mantic" "noble")
    if [[ -n "${distro_codename}" ]]; then
        for codename in "${supported_codenames[@]}"; do
            if [[ "${distro_codename}" == "${codename}" ]]; then
                echo "Adding deadsnakes PPA for ${distro_codename} to obtain Python ${PYTHON_VERSION}..."
                sudo add-apt-repository -y ppa:deadsnakes/ppa
                sudo apt-get update
                if install_python_from_apt; then
                    return 0
                fi
                break
            fi
        done
    fi

    cat <<EOF >&2
Error: Unable to install Python ${PYTHON_VERSION}. The deadsnakes PPA does not
provide packages for the current Ubuntu release (${distro_codename:-unknown}).
Please install Python ${PYTHON_VERSION} manually and re-run this script.
EOF
    exit 1
}

ensure_python_version

sudo update-alternatives --install /usr/bin/python3 python3 "$PYTHON_BIN" 2
sudo update-alternatives --set python3 "$PYTHON_BIN"

if ! "$PYTHON_BIN" -m pip --version >/dev/null 2>&1; then
    "$PYTHON_BIN" -m ensurepip --upgrade
fi

sudo apt-mark hold python3 "python${PYTHON_VERSION}"

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

if [[ "${SKIP_MT5_LOGIN_PROMPT:-0}" != "1" ]]; then
    if [ -t 0 ]; then
        launch_target=""
        if [ -f "$MT5_INSTALL_DIR/terminal64.exe" ]; then
            launch_target="$MT5_INSTALL_DIR/terminal64.exe"
        elif [ -f "$MT5_SETUP_PATH" ]; then
            launch_target="$MT5_SETUP_PATH"
        fi

        if [ -n "$launch_target" ]; then
            if command -v wine >/dev/null 2>&1; then
                echo "Launching MetaTrader 5 so you can complete the initial login..."
                wine "$launch_target" >/dev/null 2>&1 &
                wine_pid=$!
                while true; do
                    if ! read -r -p "Did you log into MetaTrader 5 successfully? Type 'yes' to continue: " response; then
                        echo "Input closed before confirmation; continuing without verification." >&2
                        break
                    fi
                    response="${response,,}"
                    if [[ "$response" == "yes" ]]; then
                        break
                    fi
                    echo "Please complete the login inside the MetaTrader 5 terminal before proceeding."
                done
                if command -v wineserver >/dev/null 2>&1; then
                    wineserver -k >/dev/null 2>&1 || true
                fi
                if kill -0 "$wine_pid" 2>/dev/null; then
                    wait "$wine_pid" || true
                fi
            else
                echo "Wine is not available; skipping automatic MetaTrader 5 launch." >&2
            fi
        else
            echo "MetaTrader 5 executable not found; skipping automatic login prompt." >&2
        fi
    else
        echo "Skipping MetaTrader 5 login prompt because the script is running non-interactively." >&2
    fi
fi

# Optionally install CUDA drivers if an NVIDIA GPU is detected or WITH_CUDA=1
if command -v nvidia-smi >/dev/null 2>&1 || [[ "${WITH_CUDA:-0}" == "1" ]]; then
    sudo apt-get install -y nvidia-cuda-toolkit
fi

PIP_CMD=("$PYTHON_BIN" -m pip)

echo "Checking for outdated Python packages before installation..."
"${PIP_CMD[@]}" list --outdated || true

echo "Upgrading pip to the latest version..."
"${PIP_CMD[@]}" install --upgrade pip

if "${PIP_CMD[@]}" show ydata-synthetic >/dev/null 2>&1; then
    echo "Removing legacy ydata-synthetic package that is incompatible with modern Python versions..."
    "${PIP_CMD[@]}" uninstall -y ydata-synthetic
fi

echo "Installing the latest compatible versions of project dependencies..."
"${PIP_CMD[@]}" install --upgrade --upgrade-strategy eager -r requirements.txt

echo "Running project package synchronisation script..."
./scripts/update_python_packages.sh

echo "Outdated packages remaining after upgrade (if any):"
"${PIP_CMD[@]}" list --outdated || true

echo "Installing and starting the MT5 bot service..."
sudo ./scripts/install_service.sh
sudo systemctl status mt5bot

echo "Enabling automatic MT5 bot updates..."
if sudo systemctl list-unit-files | grep -q '^mt5bot-update.timer'; then
    sudo systemctl enable --now mt5bot-update.timer
else
    echo "Warning: mt5bot-update.timer is not installed; skipping enable step." >&2
fi

echo "Triggering an immediate MT5 bot update check..."
"$PYTHON_BIN" -m services.auto_updater --force

echo "AutoGluon has been replaced with the built-in tabular trainer."
echo "Run 'python -m mt5.train_tabular' after setup to train the default model."
