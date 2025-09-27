#!/usr/bin/env bash
# setup_ubuntu.sh — MT5 + Wine + Windows Python 3.11 bootstrapper
# Logs all activity into ~/Downloads/mm.dd.yyyy.log before doing anything else.

set -euo pipefail

#####################################
# Terminal logging (always enabled)
#####################################
USER_HOME="$(eval echo "~${SUDO_USER:-$USER}")"
LOG_DIR="${USER_HOME}/Downloads"
mkdir -p "$LOG_DIR"
LOG_FILE="${LOG_DIR}/$(date +'%m.%d.%Y').log"

# Start logging if not already under 'script'
if [ -z "${TERMINAL_LOGGING:-}" ]; then
  export TERMINAL_LOGGING=1
  echo "[logger] Recording session to $LOG_FILE"
  exec script -q -f -a "$LOG_FILE" /bin/bash -c "$0 $*"
fi

#####################################
# Paths, env, and defaults
#####################################
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SETUP_LOG="${PROJECT_ROOT}/setup.log"

PROJECT_USER="${SUDO_USER:-$(whoami)}"
PROJECT_HOME="$(eval echo "~${PROJECT_USER}")"

WINEPREFIX_PY="${PROJECT_HOME}/.wine-py311"
WINEPREFIX_MT5="${PROJECT_HOME}/.wine-mt5"
WINEARCH="win64"

HEADLESS_MODE="${HEADLESS_MODE:-auto}"   # auto|manual
PYTHON_WIN_VERSION="${PYTHON_WIN_VERSION:-3.11.9}"
PYTHON_WIN_DIR="C:\\Python311"
PYMT5LINUX_SOURCE="${PYMT5LINUX_SOURCE:-}"
MT5_SETUP_URL="${MT5_SETUP_URL:-https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe}"

CACHE_DIR="${PROJECT_ROOT}/.cache/mt5"
PYTHON_WIN_EXE="python-${PYTHON_WIN_VERSION}-amd64.exe"
PYTHON_WIN_URL="https://www.python.org/ftp/python/${PYTHON_WIN_VERSION}/${PYTHON_WIN_EXE}"
MT5_SETUP_EXE="mt5setup.exe"

export WINEDEBUG="${WINE_DEBUG_CHANNEL:--all}"

SERVICES_ONLY=0
DISPLAY_SET_MANUALLY=""

log() { echo "[setup] $*" | tee -a "$SETUP_LOG" >&2; }
die() { echo "[setup:ERROR] $*" | tee -a "$SETUP_LOG" >&2; exit 1; }
need_root() { if [[ "$(id -u)" -ne 0 ]]; then die "Run with sudo"; fi; }
run_as_user() { sudo -H -u "${PROJECT_USER}" bash -lc "$*"; }

#####################################
# Load .env if present
#####################################
if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  log "Loading environment overrides from ${PROJECT_ROOT}/.env"
  set -a
  # shellcheck disable=SC1090
  . "${PROJECT_ROOT}/.env"
  set +a
fi

#####################################
# Parse CLI args
#####################################
for arg in "$@"; do
  case "$arg" in
    --services-only) SERVICES_ONLY=1 ;;
    --headless=manual) HEADLESS_MODE="manual" ;;
    --headless=auto) HEADLESS_MODE="auto" ;;
    *) ;; # ignore unknown
  esac
done

#####################################
# System packages
#####################################
ensure_system_packages() {
  need_root
  log "Ensuring system packages are present..."
  dpkg --add-architecture i386 || true
  apt-get update -y
  apt-get install -y \
    software-properties-common build-essential \
    cabextract wine64 wine32:i386 winetricks xvfb curl unzip p7zip-full \
    python3 python3-venv python3-dev python3-pip python3-setuptools wget
  log "System packages ensured."
}

#####################################
# Linux-side venv
#####################################
ensure_project_venv() {
  [[ "${SERVICES_ONLY}" -eq 1 ]] && return 0
  chown -R "${PROJECT_USER}:${PROJECT_USER}" "${PROJECT_ROOT}" || true
  log "Creating/using project virtualenv (.venv)..."
  run_as_user "cd '${PROJECT_ROOT}' && python3 -m venv .venv || true"
  run_as_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel || true"
}
venv_python() { echo "${PROJECT_ROOT}/.venv/bin/python"; }
venv_pip_install() { run_as_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip install --upgrade '$1'"; }

#####################################
# Xvfb helpers
#####################################
XVFB_PID_FILE="/tmp/xvfb_${PROJECT_USER}.pid"
xvfb_start() {
  if [[ "${HEADLESS_MODE}" == "manual" ]]; then
    if [[ -z "${DISPLAY:-}" ]]; then
      DISPLAY=":95"
      DISPLAY_SET_MANUALLY="${DISPLAY}"
    fi
    run_as_user "DISPLAY='${DISPLAY}' Xvfb '${DISPLAY}' -screen 0 1280x1024x24 >/tmp/xvfb_${PROJECT_USER}.log 2>&1 & echo \$! > '${XVFB_PID_FILE}'"
    log "Started manual Xvfb on ${DISPLAY}"
  fi
}
xvfb_stop() {
  if [[ -n "${DISPLAY_SET_MANUALLY}" ]] && [[ -f "${XVFB_PID_FILE}" ]]; then
    local pid
    pid="$(run_as_user "cat '${XVFB_PID_FILE}'" || true)"
    [[ -n "$pid" ]] && run_as_user "kill '${pid}' || true"
    rm -f "${XVFB_PID_FILE}" || true
  fi
}
with_display() {
  local cmd="$1"
  if [[ "${HEADLESS_MODE}" == "manual" ]]; then
    run_as_user "DISPLAY='${DISPLAY}' bash -lc \"$cmd\""
  else
    run_as_user "xvfb-run -a bash -lc \"$cmd\""
  fi
}

#####################################
# Wine helpers
#####################################
wine_env_block() { echo "export WINEARCH='${WINEARCH}'; export WINEDEBUG='${WINEDEBUG}'"; }
ensure_wineprefix() { run_as_user "$(wine_env_block); export WINEPREFIX='$1'; wineboot -u >/dev/null 2>&1 || true"; }
winetricks_quiet() { [[ $# -gt 1 ]] && run_as_user "$(wine_env_block); export WINEPREFIX='$1'; shift; winetricks -q -f $* >/dev/null 2>&1 || true"; }
wine_wait() { run_as_user "wineserver -w"; }
wine_cmd() { local prefix="$1"; shift; run_as_user "$(wine_env_block); export WINEPREFIX='${prefix}'; $*"; }

#####################################
# Install Windows Python
#####################################
install_windows_python() {
  local prefix="${WINEPREFIX_PY}"
  log "Initialising Wine prefix at ${prefix} ..."
  ensure_wineprefix "${prefix}"
  winetricks_quiet "${prefix}" vcrun2022 corefonts gdiplus
  mkdir -p "${CACHE_DIR}"
  if [[ ! -f "${CACHE_DIR}/${PYTHON_WIN_EXE}" ]]; then
    curl -fsSL -o "${CACHE_DIR}/${PYTHON_WIN_EXE}" "${PYTHON_WIN_URL}"
  fi
  run_as_user "mkdir -p '${prefix}/drive_c/_installers'"
  cp -f "${CACHE_DIR}/${PYTHON_WIN_EXE}" "${prefix}/drive_c/_installers/"
  log "Installing Windows Python ${PYTHON_WIN_VERSION}..."
  xvfb_start
  with_display "$(wine_env_block); export WINEPREFIX='${prefix}'; wine start /wait C:\\\\_installers\\\\${PYTHON_WIN_EXE} InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 TargetDir=${PYTHON_WIN_DIR} /quiet || true"
  wine_wait
  xvfb_stop
  wine_cmd "${prefix}" wine cmd /c "${PYTHON_WIN_DIR}\\python.exe -V" || log "Windows Python not found"
}

#####################################
# Install MetaTrader 5
#####################################
install_mt5() {
  local prefix="${WINEPREFIX_MT5}"
  log "Initialising Wine prefix at ${prefix} ..."
  ensure_wineprefix "${prefix}"
  winetricks_quiet "${prefix}" corefonts gdiplus
  mkdir -p "${CACHE_DIR}"
  if [[ ! -f "${CACHE_DIR}/${MT5_SETUP_EXE}" ]]; then
    curl -fsSL -o "${CACHE_DIR}/${MT5_SETUP_EXE}" "${MT5_SETUP_URL}"
  fi
  if wine_cmd "${prefix}" wine cmd /c "dir C:\\Program^ Files\\MetaTrader^ 5" >/dev/null 2>&1; then
    log "MetaTrader 5 already installed."
    return 0
  fi
  run_as_user "mkdir -p '${prefix}/drive_c/_installers'"
  cp -f "${CACHE_DIR}/${MT5_SETUP_EXE}" "${prefix}/drive_c/_installers/"
  log "Installing MetaTrader 5..."
  xvfb_start
  with_display "$(wine_env_block); export WINEPREFIX='${prefix}'; wine start /wait C:\\\\_installers\\\\${MT5_SETUP_EXE} /silent || wine start /wait C:\\\\_installers\\\\${MT5_SETUP_EXE}"
  wine_wait
  xvfb_stop
}

#####################################
# Linux bridge
#####################################
install_linux_bridge() {
  [[ "${SERVICES_ONLY}" -eq 1 ]] && return 0
  [[ -z "${PYMT5LINUX_SOURCE}" ]] && { log "PYMT5LINUX_SOURCE not set"; return 0; }
  venv_pip_install "${PYMT5LINUX_SOURCE}" || log "Failed to install bridge helper"
}

#####################################
# Output instructions
#####################################
write_instructions() {
  local file="${PROJECT_ROOT}/LOGIN_INSTRUCTIONS_WINE.txt"
  cat > "${file}" <<'TXT'
MetaTrader 5 (Wine) — Quick Usage
---------------------------------
1) Start MT5:
   WINEPREFIX="$HOME/.wine-mt5" wine "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

   First time: Login → Save password

2) Windows Python:
   WINEPREFIX="$HOME/.wine-py311" wine cmd /c "C:\\Python311\\python.exe -V"

3) Run bridge:
   rsync -a --delete /opt/mt5/ "$HOME/.wine-py311/drive_c/mt5/"
   WINEPREFIX="$HOME/.wine-py311" wine cmd /c "C:\\Python311\\python.exe C:\\mt5\\utils\\mt_5_bridge.py"
TXT
  log "Instructions written to ${file}"
}

#####################################
# Main
#####################################
main() {
  ensure_system_packages
  ensure_project_venv
  install_linux_bridge
  install_windows_python || log "Windows Python install failed"
  install_mt5
  if [[ -z "${DISPLAY:-}" && "${HEADLESS_MODE}" != "manual" ]]; then
    log "Skipping MT5 login prompt (no display). Run once manually to save creds."
  fi
  if [[ "${SERVICES_ONLY}" -ne 1 ]]; then
    run_as_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip list --outdated || true"
  fi
  write_instructions
  log "Setup complete."
}

trap 'xvfb_stop || true' EXIT
main
