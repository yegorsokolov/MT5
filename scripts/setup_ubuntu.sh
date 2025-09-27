#!/usr/bin/env bash
# setup_ubuntu.sh — MT5 + Wine + Windows Python 3.11 bootstrapper
# Usage:
#   sudo bash scripts/setup_ubuntu.sh
#   sudo bash scripts/setup_ubuntu.sh --services-only
#   sudo bash scripts/setup_ubuntu.sh --headless=manual   # persistent Xvfb
# Idempotent and safe to re-run.

set -euo pipefail

#####################################
# Paths, env, and defaults
#####################################
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_FILE="${PROJECT_ROOT}/setup.log"

# When called with sudo, do Wine work as invoking user
PROJECT_USER="${SUDO_USER:-$(whoami)}"
PROJECT_HOME="$(eval echo "~${PROJECT_USER}")"

# Wine prefixes (always user-owned; never /root)
WINEPREFIX_PY="${PROJECT_HOME}/.wine-py311"
WINEPREFIX_MT5="${PROJECT_HOME}/.wine-mt5"
WINEARCH="win64"

# Defaults (override in .env or environment before running)
HEADLESS_MODE="${HEADLESS_MODE:-auto}"   # auto|manual
PYTHON_WIN_VERSION="${PYTHON_WIN_VERSION:-3.11.9}"
PYTHON_WIN_DIR="C:\\Python311"
PYMT5LINUX_SOURCE="${PYMT5LINUX_SOURCE:-pymt5linux}"

CACHE_DIR="${PROJECT_ROOT}/.cache/mt5"
PYTHON_WIN_EXE="python-${PYTHON_WIN_VERSION}-amd64.exe"
PYTHON_WIN_URL="https://www.python.org/ftp/python/${PYTHON_WIN_VERSION}/${PYTHON_WIN_EXE}"
MT5_SETUP_EXE="mt5setup.exe"
MT5_SETUP_URL="${MT5_SETUP_URL:-https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe}"

export WINEDEBUG="${WINE_DEBUG_CHANNEL:--all}"

SERVICES_ONLY=0
DISPLAY_SET_MANUALLY=""

log() { echo "[setup] $*" | tee -a "$LOG_FILE" >&2; }
die() { echo "[setup:ERROR] $*" | tee -a "$LOG_FILE" >&2; exit 1; }
need_root() { if [[ "$(id -u)" -ne 0 ]]; then die "Run with sudo: sudo bash scripts/setup_ubuntu.sh"; fi; }
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
# Parse CLI args (simple & robust)
#####################################
for arg in "$@"; do
  case "$arg" in
    --services-only) SERVICES_ONLY=1 ;;
    --headless=manual) HEADLESS_MODE="manual" ;;
    --headless=auto) HEADLESS_MODE="auto" ;;
    *) ;; # ignore unknowns
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
    python3 python3-venv python3-dev python3-pip python3-setuptools
  log "System packages ensured."
}

#####################################
# Linux-side venv (PEP 668-safe)
#####################################
ensure_project_venv() {
  [[ "${SERVICES_ONLY}" -eq 1 ]] && return 0
  log "Creating/using project virtualenv (.venv)..."
  run_as_user "cd '${PROJECT_ROOT}' && python3 -m venv .venv || true"
  run_as_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip install --upgrade pip setuptools wheel || true"
}

venv_pip_install() {
  local pkg="$1"
  run_as_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip install --upgrade '${pkg}'"
}

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
    log "Started manual Xvfb on ${DISPLAY} (PID $(run_as_user "cat '${XVFB_PID_FILE}'"))"
  fi
}

xvfb_stop() {
  if [[ -n "${DISPLAY_SET_MANUALLY}" ]] && [[ -f "${XVFB_PID_FILE}" ]]; then
    local pid
    pid="$(run_as_user "cat '${XVFB_PID_FILE}'" || true)"
    if [[ -n "${pid}" ]]; then
      log "Stopping Xvfb (PID ${pid})..."
      run_as_user "kill '${pid}' || true"
    fi
    rm -f "${XVFB_PID_FILE}" || true
  fi
}

with_display() {
  # $1: command string to run
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
wine_env_block() {
  echo "export WINEARCH='${WINEARCH}'; export WINEDEBUG='${WINEDEBUG}'"
}

ensure_wineprefix() {
  local prefix="$1"
  run_as_user "$(wine_env_block); export WINEPREFIX='${prefix}'; wineboot -u >/dev/null 2>&1 || true"
}

winetricks_quiet() {
  local prefix="$1"; shift
  local pkgs=("$@")
  run_as_user "$(wine_env_block); export WINEPREFIX='${prefix}'; winetricks -q -f ${pkgs[*]} >/dev/null 2>&1 || true"
}

wine_wait() {
  run_as_user "wineserver -w"
}

wine_cmd() {
  local prefix="$1"; shift
  run_as_user "$(wine_env_block); export WINEPREFIX='${prefix}'; $*"
}

wine_start_wait() {
  local prefix="$1"; shift
  wine_cmd "${prefix}" wine start /wait "$@"
  wine_wait
}

#####################################
# Install Windows Python 3.11
#####################################
install_windows_python() {
  local prefix="${WINEPREFIX_PY}"

  log "Initialising Wine prefix at ${prefix} ..."
  ensure_wineprefix "${prefix}"

  log "Ensuring gecko, mono, and VC runtimes in ${prefix} ..."
  winetricks_quiet "${prefix}" gecko mono vcrun2022 corefonts gdiplus

  mkdir -p "${CACHE_DIR}"
  if [[ ! -f "${CACHE_DIR}/${PYTHON_WIN_EXE}" ]]; then
    log "Downloading Windows Python ${PYTHON_WIN_VERSION}..."
    curl -fsSL -o "${CACHE_DIR}/${PYTHON_WIN_EXE}" "${PYTHON_WIN_URL}"
  fi

  log "Installing Windows Python ${PYTHON_WIN_VERSION}..."
  xvfb_start
  with_display "$(wine_env_block); export WINEPREFIX='${prefix}'; wine start /wait Z:\\${CACHE_DIR//\//\\}\\${PYTHON_WIN_EXE} InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 TargetDir=${PYTHON_WIN_DIR} /quiet || true"
  wine_wait
  xvfb_stop

  if ! wine_cmd "${prefix}" wine cmd /c "${PYTHON_WIN_DIR}\\python.exe -V" >/tmp/wpy.ver 2>&1; then
    log "Warning: Windows Python installer exited with an error."
  else
    log "Windows Python: $(cat /tmp/wpy.ver)"
  fi
}

#####################################
# Install MetaTrader 5
#####################################
install_mt5() {
  local prefix="${WINEPREFIX_MT5}"

  log "Initialising Wine prefix at ${prefix} ..."
  ensure_wineprefix "${prefix}"

  log "Ensuring gecko/mono in ${prefix} ..."
  winetricks_quiet "${prefix}" gecko mono

  mkdir -p "${CACHE_DIR}"
  if [[ ! -f "${CACHE_DIR}/${MT5_SETUP_EXE}" ]]; then
    log "Downloading MetaTrader 5 installer..."
    curl -fsSL -o "${CACHE_DIR}/${MT5_SETUP_EXE}" "${MT5_SETUP_URL}"
  fi

  # Already installed?
  if wine_cmd "${prefix}" wine cmd /c "dir C:\\Program^ Files\\MetaTrader^ 5" >/dev/null 2>&1; then
    log "MetaTrader 5 already installed in ${prefix}."
    return 0
  fi

  log "Installing MetaTrader 5..."
  xvfb_start
  with_display "$(wine_env_block); export WINEPREFIX='${prefix}'; wine start /wait Z:\\${CACHE_DIR//\//\\}\\${MT5_SETUP_EXE} /silent || wine start /wait Z:\\${CACHE_DIR//\//\\}\\${MT5_SETUP_EXE}"
  wine_wait
  xvfb_stop

  if ! wine_cmd "${prefix}" wine cmd /c "dir C:\\Program^ Files\\MetaTrader^ 5" >/dev/null 2>&1; then
    log 'Warning: MT5 may not have completed installation.'
  fi
}

#####################################
# Linux bridge helper
#####################################
install_linux_bridge() {
  [[ "${SERVICES_ONLY}" -eq 1 ]] && return 0
  log "Ensuring Linux-side bridge helper (${PYMT5LINUX_SOURCE}) is installed..."
  if ! venv_pip_install "${PYMT5LINUX_SOURCE}"; then
    log "Warning: Failed to install ${PYMT5LINUX_SOURCE}."
    if [[ "${PYMT5LINUX_SOURCE}" == "pymt5linux" ]]; then
      log "Hint: export PYMT5LINUX_SOURCE to a Git URL or wheel."
    fi
  fi
}

#####################################
# MetaTrader terminal auto-config
#####################################
auto_configure_terminal() {
  [[ "${SERVICES_ONLY}" -eq 1 ]] && return 0
  if [[ ! -x "${PROJECT_ROOT}/.venv/bin/python" ]]; then
    log "Virtualenv missing or inactive; skipping MetaTrader auto-configuration."
    return 0
  fi

  local env_file="${PROJECT_ROOT}/.env"
  if [[ ! -f "${env_file}" && -f "${PROJECT_ROOT}/.env.template" ]]; then
    log "Seeding ${env_file} from template..."
    run_as_user "cp -n '${PROJECT_ROOT}/.env.template' '${env_file}'" || true
  fi

  log "Auto-detecting MetaTrader 5 terminal path..."
  local detect_cmd="cd '${PROJECT_ROOT}' && . .venv/bin/activate && python scripts/detect_mt5_terminal.py --env-file '${env_file}'"
  if ! run_as_user "${detect_cmd}"; then
    log "Warning: MetaTrader 5 terminal detection failed; review setup manually."
  fi

  log "Installing MetaTrader heartbeat script and verifying bridge..."
  local heartbeat_cmd="cd '${PROJECT_ROOT}' && . .venv/bin/activate && python scripts/setup_terminal.py --install-heartbeat"
  if ! run_as_user "${heartbeat_cmd}"; then
    log "Warning: Heartbeat installation or bridge verification encountered an error."
  fi
}

#####################################
# Output instructions
#####################################
write_instructions() {
  local file="${PROJECT_ROOT}/LOGIN_INSTRUCTIONS_WINE.txt"
  cat > "${file}" <<'TXT'
MetaTrader 5 (Wine) — Login Instructions

1) If headless, either run setup with --headless=manual (persistent Xvfb) or use a real X session.

2) Start MT5 manually:
   export WINEPREFIX="$HOME/.wine-mt5"
   wine "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

3) Check Windows Python:
   export WINEPREFIX="$HOME/.wine-py311"
   wine cmd /c "C:\\Python311\\python.exe -V"

4) Run your Windows-side bridge script:
   wine cmd /c "C:\\Python311\\python.exe Z:\\opt\\mt5\\utils\\mt_5_bridge.py"
TXT
  log "MetaTrader 5 Wine instructions saved to ${file}"
}

#####################################
# Main
#####################################
main() {
  ensure_system_packages
  ensure_project_venv
  install_linux_bridge

  install_windows_python || log "Warning: Windows Python not available; continuing."
  install_mt5
  auto_configure_terminal

  if [[ -z "${DISPLAY:-}" && "${HEADLESS_MODE}" != "manual" ]]; then
    log "Skipping MT5 login prompt (no graphical display)."
  fi

  if [[ "${SERVICES_ONLY}" -ne 1 ]]; then
    log "Checking for outdated Python packages in venv (no forced upgrades)..."
    run_as_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip list --outdated || true"
    log "Ensuring pip is current in venv..."
    run_as_user "cd '${PROJECT_ROOT}' && . .venv/bin/activate && python -m pip install --upgrade pip || true"
  fi

  write_instructions
  log "Setup complete."
}

trap 'xvfb_stop || true' EXIT
main
