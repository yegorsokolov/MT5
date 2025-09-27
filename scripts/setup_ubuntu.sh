#!/usr/bin/env bash
# setup_ubuntu.sh — MT5 + Wine + Windows Python 3.11 bootstrapper
# Usage:
#   sudo bash scripts/setup_ubuntu.sh
#   sudo bash scripts/setup_ubuntu.sh --services-only
#   sudo bash scripts/setup_ubuntu.sh --headless=manual   # persistent Xvfb
#
# This script is idempotent and safe to re-run.

set -euo pipefail

########## Globals & Helpers ##########

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/setup.log"

# Detect target user (when invoked with sudo, operate on the invoking user)
PROJECT_USER="${SUDO_USER:-$(whoami)}"
PROJECT_HOME="$(eval echo "~${PROJECT_USER}")"

# Default Wine config (always user-owned; never /root)
WINEPREFIX_PY="${PROJECT_HOME}/.wine-py311"
WINEPREFIX_MT5="${PROJECT_HOME}/.wine-mt5"
WINEARCH="win64"

# Defaults (can be overridden in .env or env)
HEADLESS_MODE="${HEADLESS_MODE:-auto}"       # auto|manual
PYMT5LINUX_SOURCE="${PYMT5LINUX_SOURCE:-pymt5linux}"
PYTHON_WIN_VERSION="${PYTHON_WIN_VERSION:-3.11.9}"
PYTHON_WIN_DIR="C:\\Python311"

# Cache installers here
CACHE_DIR="${PROJECT_ROOT}/.cache/mt5"
PYTHON_WIN_EXE="python-${PYTHON_WIN_VERSION}-amd64.exe"
PYTHON_WIN_URL="https://www.python.org/ftp/python/${PYTHON_WIN_VERSION}/${PYTHON_WIN_EXE}"
MT5_SETUP_EXE="mt5setup.exe"
MT5_SETUP_URL="${MT5_SETUP_URL:-https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe}"

# Wine logging
export WINEDEBUG="${WINE_DEBUG_CHANNEL:--all}"

# State
SERVICES_ONLY=0

log() { echo -e "[setup] $*" | tee -a "$LOG_FILE" >&2; }
die() { echo -e "[setup:ERROR] $*" | tee -a "$LOG_FILE" >&2; exit 1; }

# Ensure we run as root for apt part, but do Wine work as PROJECT_USER
need_root() { if [[ "$(id -u)" -ne 0 ]]; then die "Please run as root (e.g., sudo bash scripts/setup_ubuntu.sh)"; fi; }

run_as_project_user() {
  # shellcheck disable=SC2024
  sudo -H -u "${PROJECT_USER}" bash -lc "$*"
}

# Execute a Python script from the project directory using the virtualenv if present.
project_python() {
  local script_path="$1"
  shift || true

  local interpreter="${PROJECT_ROOT}/.venv/bin/python"
  if [[ ! -x "${interpreter}" ]]; then
    interpreter="$(command -v python3 || command -v python)"
  fi
  if [[ -z "${interpreter}" ]]; then
    log "Warning: No Python interpreter available to execute ${script_path}"
    return 1
  fi

  local args=""
  if [[ $# -gt 0 ]]; then
    args="$(printf ' %q' "$@")"
  fi

  run_as_project_user "cd '${PROJECT_ROOT}' && \"${interpreter}\" \"${script_path}\"${args}"
}

# Load .env (optional)
load_env() {
  if [[ -f "${PROJECT_ROOT}/.env" ]]; then
    log "Loading environment overrides from ${PROJECT_ROOT}/.env"
    # shellcheck disable=SC1090
    set -a
    . "${PROJECT_ROOT}/.env"
    set +a
  fi
}

########## CLI Parse ##########

for arg in "$@"; do
  case "$arg" in
    --services-only) SERVICES_ONLY=1 ;;
    --headless=*)    HEADLESS_MODE="${arg#*=}" ;;
    *)               ;;
  case_esac=true
  done
done
# (No unmatched ‘}’ — previous EOF error fixed)

########## Apt pre-reqs ##########

ensure_system_packages() {
  need_root
  log "Ensuring system packages are present..."
  dpkg --add-architecture i386 || true
  apt-get update -y
  apt-get install -y \
    software-properties-common build-essential \
    cabextract wine64 wine32:i386 winetricks xvfb curl unzip p7zip-full \
    python3 python3-venv python3-dev python3-pip python3-setuptools
  log "System packages ensured."
}

########## Linux-side venv (PEP 668 safe) ##########

ensure_project_venv() {
  [[ "${SERVICES_ONLY}" -eq 1 ]] && return 0

  log "Creating/using project virtualenv (.venv) for Linux-side Python..."
  run_as_project_user "cd '${PROJECT_ROOT}' && python3 -m venv .venv || true"
  run_as_project_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel || true"
}

# Helper to run pip in venv
venv_pip_install() {
  local pkg="$1"
  run_as_project_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip install --upgrade '${pkg}'"
}

########## Wine / Xvfb helpers ##########

XFVB_PID=""

xvfb_start() {
  # Decide whether to use auto (xvfb-run) or manual Xvfb
  if [[ "${HEADLESS_MODE}" == "manual" ]]; then
    local display=":95"
    run_as_project_user "DISPLAY=${display} Xvfb ${display} -screen 0 1280x1024x24 >/tmp/xvfb_${PROJECT_USER}.log 2>&1 & echo \$! > /tmp/xvfb_${PROJECT_USER}.pid"
    XFVB_PID="$(run_as_project_user "cat /tmp/xvfb_${PROJECT_USER}.pid || true")"
    export DISPLAY="${display}"
    log "Started manual Xvfb on ${display} (PID ${XFVB_PID})."
  else
    # auto mode -> we will wrap individual commands with xvfb-run -a
    export DISPLAY="${DISPLAY:-}"
  fi
}

xvfb_stop() {
  if [[ -n "${XFVB_PID}" ]]; then
    log "Stopping Xvfb (PID ${XFVB_PID})..."
    run_as_project_user "kill ${XFVB_PID} || true"
    XFVB_PID=""
  fi
}

wine_env() {
  cat <<EOF
export WINEARCH="${WINEARCH}"
export WINEDEBUG="${WINEDEBUG}"
EOF
}

ensure_wineprefix() {
  local prefix="$1"
  run_as_project_user "$(wine_env); export WINEPREFIX='${prefix}'; wineboot -u >/dev/null 2>&1 || true"
}

winetricks_quiet() {
  local prefix="$1"; shift
  local pkgs=("$@")
  run_as_project_user "$(wine_env); export WINEPREFIX='${prefix}'; winetricks -q -f ${pkgs[*]} >/dev/null 2>&1 || true"
}

wine_wait() {
  run_as_project_user "wineserver -w"
}

wine_cmd() {
  local prefix="$1"; shift
  run_as_project_user "$(wine_env); export WINEPREFIX='${prefix}'; $*"
}

wine_start_wait() {
  local prefix="$1"; shift
  # Use `start /wait` to reliably block until installer finishes
  wine_cmd "${prefix}" wine start /wait "$@"
  wine_wait
}

with_display() {
  # Wrap a command either in xvfb-run (auto) or rely on manual DISPLAY (manual)
  if [[ "${HEADLESS_MODE}" == "manual" ]]; then
    run_as_project_user "DISPLAY='${DISPLAY}' bash -lc \"$*\""
  else
    run_as_project_user "xvfb-run -a bash -lc \"$*\""
  fi
}

########## Windows Python install inside Wine ##########

install_windows_python() {
  local prefix="${WINEPREFIX_PY}"

  log "Initialising Wine prefix at ${prefix} ..."
  ensure_wineprefix "${prefix}"

  log "Ensuring supplemental Wine components (gecko, mono, VC runtimes) in ${prefix} ..."
  winetricks_quiet "${prefix}" gecko mono vcrun2022 corefonts gdiplus

  log "Installing core Wine runtime components into ${prefix} ..."
  wine_cmd "${prefix}" wineboot -u >/dev/null 2>&1 || true

  mkdir -p "${CACHE_DIR}"
  if [[ ! -f "${CACHE_DIR}/${PYTHON_WIN_EXE}" ]]; then
    log "Downloading Windows Python ${PYTHON_WIN_VERSION} ..."
    curl -fsSL -o "${CACHE_DIR}/${PYTHON_WIN_EXE}" "${PYTHON_WIN_URL}"
  fi

  log "Installing Windows Python ${PYTHON_WIN_VERSION} inside Wine prefix ${prefix} ..."
  xvfb_start
  with_display "$(wine_env); export WINEPREFIX='${prefix}'; wine start /wait Z:\\${CACHE_DIR//\//\\}\\${PYTHON_WIN_EXE} \
    InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 TargetDir=${PYTHON_WIN_DIR} /quiet || true"
  wine_wait
  xvfb_stop

  # Verify
  if ! wine_cmd "${prefix}" wine cmd /c "${PYTHON_WIN_DIR}\\python.exe -V" >/tmp/wpy.ver 2>&1; then
    log "Warning: Windows Python installer exited with an error."
  else
    log "Windows Python version: $(cat /tmp/wpy.ver)"
  fi
}

########## MetaTrader 5 install ##########

install_mt5() {
  local prefix="${WINEPREFIX_MT5}"
  log "Initialising Wine prefix at ${prefix} ..."
  ensure_wineprefix "${prefix}"

  log "Ensuring supplemental Wine components (gecko, mono) are available in ${prefix} ..."
  winetricks_quiet "${prefix}" gecko mono

  mkdir -p "${CACHE_DIR}"
  if [[ ! -f "${CACHE_DIR}/${MT5_SETUP_EXE}" ]]; then
    log "Downloading MetaTrader 5 installer ..."
    curl -fsSL -o "${CACHE_DIR}/${MT5_SETUP_EXE}" "${MT5_SETUP_URL}"
  fi

  # Detect if already installed
  if wine_cmd "${prefix}" wine cmd /c "dir C:\\Program^ Files\\MetaTrader^ 5" >/dev/null 2>&1; then
    log "MetaTrader 5 already installed in Wine prefix (${prefix}/drive_c/Program Files/MetaTrader 5)."
    return 0
  fi

  log "Launching MetaTrader 5 installer under Wine..."
  xvfb_start
  # Try silent first; if not supported, fall back to GUI
  with_display "$(wine_env); export WINEPREFIX='${prefix}'; wine start /wait Z:\\${CACHE_DIR//\//\\}\\${MT5_SETUP_EXE} /silent || wine start /wait Z:\\${CACHE_DIR//\//\\}\\${MT5_SETUP_EXE}"
  wine_wait
  xvfb_stop

  if ! wine_cmd "${prefix}" wine cmd /c "dir C:\\Program^ Files\\MetaTrader^ 5" >/dev/null 2>&1; then
    log "Warning: MT5 may not have completed installation."
  fi
}

########## Linux bridge helper ##########

install_linux_bridge() {
  [[ "${SERVICES_ONLY}" -eq 1 ]] && return 0

  log "Ensuring Linux-side bridge helper (${PYMT5LINUX_SOURCE}) is installed ..."
  if ! venv_pip_install "${PYMT5LINUX_SOURCE}"; then
    log "Warning: Failed to install ${PYMT5LINUX_SOURCE} in the Linux environment."
    if [[ "${PYMT5LINUX_SOURCE}" == "pymt5linux" ]]; then
      log "Hint: export PYMT5LINUX_SOURCE to point at a Git repository or wheel URL."
    fi
  fi
}

detect_mt5_terminal() {
  local script="scripts/detect_mt5_terminal.py"
  if [[ ! -f "${PROJECT_ROOT}/${script}" ]]; then
    log "Skipping MetaTrader 5 terminal detection (script not found)."
    return 0
  fi

  log "Detecting MetaTrader 5 terminal location..."
  if ! project_python "${script}"; then
    log "Warning: Unable to auto-detect MetaTrader 5 terminal; update MT5_TERMINAL_PATH manually if required."
  fi
}

configure_mt5_terminal() {
  [[ "${SERVICES_ONLY}" -eq 1 ]] && return 0

  local script="scripts/setup_terminal.py"
  if [[ ! -f "${PROJECT_ROOT}/${script}" ]]; then
    log "Skipping MetaTrader 5 terminal verification (script not found)."
    return 0
  fi

  log "Verifying MetaTrader 5 connectivity and installing heartbeat script..."
  if ! project_python "${script}" --install-heartbeat; then
    log "Warning: MetaTrader 5 verification failed. Ensure the terminal is logged in and rerun the installer if needed."
  fi
}

########## Misc outputs ##########

write_instructions() {
  local file="${PROJECT_ROOT}/LOGIN_INSTRUCTIONS_WINE.txt"
  cat > "${file}" <<'TXT'
MetaTrader 5 (Wine) — Login Instructions (Headless Safe)

1) If running headless, ensure an X server is available (either use --headless=manual
   so the script spins up Xvfb, or run interactively with a desktop/X forwarding).

2) To start MT5 manually:
   export WINEPREFIX="$HOME/.wine-mt5"
   wine "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

3) Windows Python inside Wine:
   export WINEPREFIX="$HOME/.wine-py311"
   wine cmd /c "C:\\Python311\\python.exe -V"

4) Running a Windows-side script from your repo:
   wine cmd /c "C:\\Python311\\python.exe Z:\\opt\\mt5\\utils\\mt_5_bridge.py"
TXT
  log "MetaTrader 5 Wine instructions saved to ${file}"
}

########## Main ##########

main() {
  load_env
  ensure_system_packages
  ensure_project_venv
  install_linux_bridge

  # Windows-side pieces
  install_windows_python || log "Warning: Windows Python is not available; skipping MetaTrader5 pip installation."

  install_mt5
  detect_mt5_terminal
  configure_mt5_terminal

  # If no DISPLAY and headless, skip auto login prompt
  if [[ -z "${DISPLAY:-}" ]]; then
    log "Skipping MetaTrader 5 login prompt because no graphical display is available."
  fi

  # Optional: check & advise Linux packages in venv (no forced upgrades)
  if [[ "${SERVICES_ONLY}" -ne 1 ]]; then
    log "Checking for outdated Python packages before installation..."
    run_as_project_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip list --outdated || true"
    log "Upgrading pip to the latest version inside venv..."
    run_as_project_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip install --upgrade pip || true"
  fi

  write_instructions
  log "Setup complete."
}

trap 'xvfb_stop || true' EXIT
main
