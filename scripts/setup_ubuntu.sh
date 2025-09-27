#!/usr/bin/env bash
# setup_ubuntu.sh — MT5 + Wine + Windows Python 3.11 + Linux venv bootstrap
# Idempotent and safe to re-run.
# Usage:
#   sudo ./scripts/setup_ubuntu.sh
#   sudo ./scripts/setup_ubuntu.sh --services-only     # skip Wine installers, only Python/venv bits, etc.

set -euo pipefail

############################################
# Config & helpers
############################################
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${PROJECT_ROOT}/.env"
LOG_FILE="${PROJECT_ROOT}/setup.log"

SERVICES_ONLY=0
if [[ "${1:-}" == "--services-only" ]]; then
  SERVICES_ONLY=1
fi

# Load .env if present
if [[ -f "$ENV_FILE" ]]; then
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  echo "Loading environment overrides from ${ENV_FILE}"
fi

# Defaults (can be overridden in .env)
DEBIAN_FRONTEND="${DEBIAN_FRONTEND:-noninteractive}"
APT_PKGS="${APT_PKGS:-software-properties-common build-essential cabextract wine64 wine32:i386 winetricks xvfb curl unzip p7zip-full python3 python3-venv python3-dev python3-pip python3-setuptools}"
WINE_DEBUG_CHANNEL="${WINE_DEBUG_CHANNEL:--all}"
PYMT5LINUX_SOURCE="${PYMT5LINUX_SOURCE:-pymt5linux}" # can be a git URL or wheel URL
PYWIN_VERSION="${PYWIN_VERSION:-3.11.9}"             # Windows Python
PYWIN_DIR="${PYWIN_DIR:-C:\\Python311}"              # TargetDir in installer
DISPLAY_NUM="${DISPLAY_NUM:-95}"                     # Xvfb display

# The Linux user that should own Wine prefixes and run Wine commands
PROJECT_USER="${SUDO_USER:-$(id -un)}"
PROJECT_UID="$(id -u "$PROJECT_USER")"
PROJECT_HOME="$(getent passwd "$PROJECT_USER" | cut -d: -f6)"

WINEPREFIX_PY="${PROJECT_HOME}/.wine-py311"
WINEPREFIX_MT5="${PROJECT_HOME}/.wine-mt5"
CACHE_DIR="${PROJECT_ROOT}/.cache/mt5"
mkdir -p "$CACHE_DIR"

# Run a command as the project user with a clean login shell
run_as_project_user() {
  local cmd="$*"
  if [[ "$(id -un)" == "$PROJECT_USER" ]]; then
    bash -lc "$cmd"
  else
    su - "$PROJECT_USER" -c "$cmd"
  fi
}

# Small pretty printer
say() { echo -e "\033[1;32m$*\033[0m"; }
warn() { echo -e "\033[1;33m$*\033[0m" >&2; }
err() { echo -e "\033[1;31m$*\033[0m" >&2; }

############################################
# 1) Base OS deps (idempotent)
############################################
say "Ensuring system packages are present..."
sudo dpkg --add-architecture i386 >/dev/null 2>&1 || true
sudo apt-get update -y
sudo apt-get install -y $APT_PKGS

############################################
# 2) Linux-side Python venv (PEP 668 safe)
############################################
say "Creating/using project virtualenv (.venv) for Linux-side Python..."
if [[ ! -d "${PROJECT_ROOT}/.venv" ]]; then
  python3 -m venv "${PROJECT_ROOT}/.venv"
fi
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
PIP_BIN="${PROJECT_ROOT}/.venv/bin/pip"

# Upgrade pip inside venv (PEP668-safe)
"$PIP_BIN" install --upgrade pip >/dev/null

# Linux helper (pymt5linux) — optional but recommended
say "Ensuring Linux-side bridge helper (${PYMT5LINUX_SOURCE}) is installed ..."
if ! "$PIP_BIN" install --upgrade "${PYMT5LINUX_SOURCE}" >/dev/null 2>&1; then
  warn "Failed to install ${PYMT5LINUX_SOURCE} in the Linux environment."
  if [[ "${PYMT5LINUX_SOURCE}" == "pymt5linux" ]]; then
    warn "Hint: export PYMT5LINUX_SOURCE to point at a Git repository or wheel URL."
  fi
fi

############################################
# 3) Common Wine environment
############################################
export WINEDEBUG="${WINE_DEBUG_CHANNEL}"
export WINEARCH=win64

# Helpful function: ensure core runtime components in a prefix
winetricks_core() {
  local prefix="$1"
  run_as_project_user "export WINEPREFIX='${prefix}' WINEARCH=win64 WINEDEBUG='${WINEDEBUG}'; \
    winetricks -q -f corefonts gdiplus msxml6 vcrun2022 gecko mono"
}

# Check if Windows Python exists in a prefix
have_win_python() {
  local prefix="$1"
  local check_cmd='if exist C:\Python311\python.exe (echo PRESENT) else (echo MISSING)'
  local out
  out=$(run_as_project_user "export WINEPREFIX='${prefix}' WINEARCH=win64 WINEDEBUG='${WINEDEBUG}'; wine cmd /c \"$check_cmd\" 2>/dev/null" || true)
  [[ "$out" =~ PRESENT ]]
}

# Bring up a persistent Xvfb session for reliable GUI installers
start_xvfb() {
  local d=":${DISPLAY_NUM}"
  if pgrep -a Xvfb | grep -q " ${d} "; then
    return 0
  fi
  run_as_project_user "nohup Xvfb ${d} -screen 0 1024x768x24 >/tmp/xvfb-${DISPLAY_NUM}.log 2>&1 & sleep 1"
}

stop_xvfb() {
  local d=":${DISPLAY_NUM}"
  run_as_project_user "pkill -f 'Xvfb ${d}' || true"
}

############################################
# 4) Windows Python 3.11 in Wine (prefix: .wine-py311)
############################################
if [[ "$SERVICES_ONLY" -ne 1 ]]; then
  say "Initialising Wine prefix at ${WINEPREFIX_PY} ..."
  run_as_project_user "export WINEPREFIX='${WINEPREFIX_PY}' WINEARCH=win64; wineboot --init || true"

  say "Ensuring supplemental Wine components (gecko, mono) are available in ${WINEPREFIX_PY} ..."
  winetricks_core "${WINEPREFIX_PY}"

  if ! have_win_python "${WINEPREFIX_PY}"; then
    say "Installing Windows Python ${PYWIN_VERSION} inside Wine prefix ${WINEPREFIX_PY} ..."
    PYEXE="python-${PYWIN_VERSION}-amd64.exe"
    if [[ ! -f "${CACHE_DIR}/${PYEXE}" ]]; then
      say "Downloading Windows Python ${PYWIN_VERSION} ..."
      curl -fsSL -o "${CACHE_DIR}/${PYEXE}" "https://www.python.org/ftp/python/${PYWIN_VERSION}/${PYEXE}"
    fi

    # Start persistent X and run installer with start /wait; then wait for wineserver
    start_xvfb
    export DISPLAY=":${DISPLAY_NUM
