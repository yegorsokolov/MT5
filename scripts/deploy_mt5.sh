#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_python_version_config.sh
source "${SCRIPT_DIR}/_python_version_config.sh"

# --------- SETTINGS ---------
export WINEARCH=win64
export WINEDEBUG=-all
MT5_PREFIX="$HOME/.wine-mt5"                # single prefix for MT5 + Windows-Python
REPO_SSH="git@github.com:yegorsokolov/MT5.git"
PROJ_DIR="$HOME/MT5"
WIN_PY_VER="${WIN_PY_VER:-$MT5_PYTHON_PATCH}"
WIN_PY_EXE="python-${WIN_PY_VER}-amd64.exe"
WIN_PY_URL="https://www.python.org/ftp/python/${WIN_PY_VER}/${WIN_PY_EXE}"
LINUX_PY_VERSION="${LINUX_PY_VERSION:-$MT5_PYTHON_PATCH}"
MT5_SETUP_URL="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe"
# Helper that searches the active Wine prefix for a terminal64.exe. MetaTrader 5
# occasionally installs under per-user directories (for example under
# `%LOCALAPPDATA%\Programs`) instead of the default `C:\Program Files\MetaTrader 5`.
# The function mirrors the official documentation by probing common locations
# and finally falling back to a filesystem crawl when required.
find_mt5_terminal() {
  local search_root
  search_root="${WINEPREFIX:-$HOME/.wine}/drive_c"

  # Common installation paths observed across MetaQuotes builds.
  local candidates=(
    "${search_root}/Program Files/MetaTrader 5/terminal64.exe"
    "${search_root}/Program Files/MetaTrader 5 Terminal/terminal64.exe"
    "${search_root}/Program Files (x86)/MetaTrader 5/terminal64.exe"
    "${search_root}/Program Files (x86)/MetaTrader 5 Terminal/terminal64.exe"
    "${search_root}/users/${USER}/AppData/Local/Programs/MetaTrader 5/terminal64.exe"
    "${search_root}/users/${USER}/AppData/Local/MetaTrader 5/terminal64.exe"
    "${search_root}/users/${USER}/AppData/Roaming/MetaQuotes/Terminal/terminal64.exe"
    "${search_root}/users/Public/AppData/Local/Programs/MetaTrader 5/terminal64.exe"
  )

  for candidate in "${candidates[@]}"; do
    if [[ -f "${candidate}" ]]; then
      echo "${candidate}"
      return 0
    fi
  done

  local fallback=""
  fallback=$(find "${search_root}" -maxdepth 12 -type f -iname terminal64.exe -print -quit 2>/dev/null || true)
  if [[ -n "${fallback}" ]]; then
    echo "${fallback}"
  fi
}
# ----------------------------

echo ">>> Cleanup"
cd "$HOME"
rm -rf "${MT5_PREFIX}" ~/.wine "$HOME/${MT5_PYTHON_PREFIX_NAME}" || true
sudo rm -rf /opt/mt5 || true
rm -rf "$PROJ_DIR" || true

echo ">>> Install Wine + helpers"
sudo dpkg --add-architecture i386 || true
sudo apt update
sudo apt install -y wine64 wine32:i386 winetricks cabextract wget unzip git python3-venv

echo ">>> Init Wine prefix: ${MT5_PREFIX}"
export WINEPREFIX="${MT5_PREFIX}"
wineboot --init

echo ">>> Install VC++ runtimes (UCRT etc.)"
winetricks -q vcrun2022 || true
# Force DLL overrides to native,builtin for UCRT & MSVC runtimes
cat > /tmp/dlloverrides.reg <<'REG'
REGEDIT4

[HKEY_CURRENT_USER\Software\Wine\DllOverrides]
"ucrtbase"="native,builtin"
"msvcp140"="native,builtin"
"vcruntime140"="native,builtin"
"vcruntime140_1"="native,builtin"
REG
wine regedit /S /tmp/dlloverrides.reg

echo ">>> Download and install MetaTrader 5"
sudo mkdir -p /opt/mt5 && sudo chown "$USER":"$USER" /opt/mt5
cd /opt/mt5
wget -O mt5setup.exe "${MT5_SETUP_URL}"
# GUI installer gives best results for first-time setup
wine mt5setup.exe || true

MT5_TERMINAL_UNIX=$(find_mt5_terminal)
if [[ -z "${MT5_TERMINAL_UNIX}" ]]; then
  echo "MetaTrader 5 terminal not found inside ${WINEPREFIX}" >&2
  echo "Verify the installer completed successfully and rerun the script." >&2
  exit 1
fi
MT5_INSTALL_DIR_UNIX="$(dirname "${MT5_TERMINAL_UNIX}")"
MT5_TERMINAL_WIN=$(winepath -w "${MT5_TERMINAL_UNIX}")

echo "Detected MetaTrader 5 terminal at ${MT5_TERMINAL_UNIX}"

echo ">>> Start MT5 so you can log in (leave it running)"
wine "${MT5_TERMINAL_UNIX}" &>/dev/null &

echo ">>> Download and install Windows-Python ${WIN_PY_VER} in the SAME prefix"
cd /opt/mt5
wget -O "${WIN_PY_EXE}" "${WIN_PY_URL}"
wine "${WIN_PY_EXE}" /quiet InstallAllUsers=1 PrependPath=1 Include_pip=1
# Find Windows-Python in this prefix
WIN_PY=$(find "$WINEPREFIX/drive_c" -maxdepth 6 -type f -iname python.exe | grep -Ei 'Python3[0-9]{2}/python.exe$' | head -n1)
if [[ -z "${WIN_PY}" ]]; then
  WIN_PY=$(find "$WINEPREFIX/drive_c" -maxdepth 6 -type f -iname python.exe | head -n1)
fi
if [[ -z "${WIN_PY}" ]]; then
  echo "Windows Python executable not found inside ${WINEPREFIX}" >&2
  exit 1
fi
WIN_PY_WINPATH=$(winepath -w "$WIN_PY")

echo ">>> Install MetaTrader5 (and pin NumPy for stability) inside Windows-Python"
wine "$WIN_PY_WINPATH" -m pip install --upgrade pip
wine "$WIN_PY_WINPATH" -m pip install "numpy<2.4" MetaTrader5

echo ">>> Clone your project via SSH"
cd "$HOME"
git clone "$REPO_SSH" "$PROJ_DIR"
cd "$PROJ_DIR"

echo ">>> Create Linux venv (prefer pyenv ${LINUX_PY_VERSION} if available)"
if command -v pyenv >/dev/null 2>&1; then
  eval "$(pyenv init - bash)" 2>/dev/null || true
  pyenv install -s "${LINUX_PY_VERSION}"
  pyenv local "${LINUX_PY_VERSION}"
  PYENV_PY="$(pyenv which python)"
  "$PYENV_PY" -m venv .venv
else
  python3 -m venv .venv
fi
source .venv/bin/activate
python -V

echo ">>> Install Python deps (excluding MetaTrader5 which is Windows-only)"
if grep -qi '^MetaTrader5' requirements.txt; then
  grep -vi '^MetaTrader5' requirements.txt > requirements.nomt5.txt
  pip install --upgrade pip wheel setuptools
  pip install -r requirements.nomt5.txt
else
  pip install --upgrade pip wheel setuptools
  pip install -r requirements.txt
fi

echo ">>> Prepare mt5linux auxiliary environment"
# shellcheck source=./_mt5linux_env.sh
source "${PROJ_DIR}/scripts/_mt5linux_env.sh"
if ! refresh_mt5linux_venv "${PYTHON_BIN:-}"; then
  echo "Failed to prepare the mt5linux auxiliary environment. Review mt5linux-lock.txt and rerun ./use-mt5linux.sh." >&2
fi

echo ">>> Write .env with Wine/MT5 paths"
MT5_TERMINAL_ESCAPED=$(echo "${MT5_TERMINAL_WIN}" | sed 's/\\\\/\\\\\\\\/g')
cat > .env <<EOF
# --- auto-generated by deploy_mt5.sh ---
WINEPREFIX=${MT5_PREFIX}
WINE_PYTHON="$(echo "$WIN_PY_WINPATH" | sed 's/\\\\/\\\\\\\\/g')"
MT5_TERMINAL_PATH="${MT5_TERMINAL_ESCAPED}"
EOF

echo ">>> Install heartbeat script into MT5"
MT5_MQL5_DIR="${MT5_INSTALL_DIR_UNIX}/MQL5"
mkdir -p "${MT5_MQL5_DIR}/Scripts"
if [ -d "scripts/MT5Bridge" ]; then
  cp -r scripts/MT5Bridge "${MT5_MQL5_DIR}/Scripts/" || true
fi

echo
echo "### ACTION REQUIRED ###"
echo "1) In the MT5 window (already launched), log in to your broker and wait until it's connected."
echo "2) Then press ENTER here to continue."
read -r _

echo ">>> Run your Ubuntu setup script (now that MT5 is connected)"
bash scripts/setup_ubuntu.sh

echo ">>> Quick environment check"
python -m utils.environment || true

echo
echo "All done. Notes:"
echo "- Wine logs are suppressed via WINEDEBUG=-all (remove if you want verbose)."
echo "- If a future 'ucrtbase.crealf' appears, re-run: winetricks -q vcrun2022 and keep DLL overrides."

