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

# Clean (truncate) today's log file before writing anything new
: >"$LOG_FILE"

# Start logging if not already under 'script'
if [ -z "${TERMINAL_LOGGING:-}" ]; then
  export TERMINAL_LOGGING=1
  echo "[logger] Recording session to $LOG_FILE"

  # Get an absolute path to this script (sudo keeps $PWD but be safe)
  SCRIPT_ABS="$(readlink -f "$0")"

  # Build a safely-quoted command: /bin/bash <script> <args...>
  printf -v _CMD '%q ' /bin/bash "$SCRIPT_ABS" "$@"

  # Run the command inside 'script' using -c (avoid argument parsing issues)
  exec script -q -f -a "$LOG_FILE" -c "$_CMD"
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
# Populated dynamically after installation so we do not rely on a hard coded
# ``C:\\Python311`` target which newer installers sometimes ignore.
WINDOWS_PYTHON_UNIX_PATH=""
WINDOWS_PYTHON_WIN_PATH=""
PYMT5LINUX_SOURCE="${PYMT5LINUX_SOURCE:-}"
MT5_SETUP_URL="${MT5_SETUP_URL:-https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe}"

CACHE_DIR="${PROJECT_ROOT}/.cache/mt5"
PYTHON_WIN_EXE="python-${PYTHON_WIN_VERSION}-amd64.exe"
PYTHON_WIN_URL="https://www.python.org/ftp/python/${PYTHON_WIN_VERSION}/${PYTHON_WIN_EXE}"
MT5_SETUP_EXE="mt5setup.exe"
FORCE_INSTALLER_REFRESH="${FORCE_INSTALLER_REFRESH:-0}"

export WINEDEBUG="${WINE_DEBUG_CHANNEL:--all}"

SERVICES_ONLY=0
DISPLAY_SET_MANUALLY=""

log() { echo "[setup] $*" | tee -a "$SETUP_LOG" >&2; }
die() { echo "[setup:ERROR] $*" | tee -a "$SETUP_LOG" >&2; exit 1; }
need_root() { if [[ "$(id -u)" -ne 0 ]]; then die "Run with sudo"; fi; }
run_as_user() { sudo -H -u "${PROJECT_USER}" bash -lc "$*"; }

discover_windows_python() {
  local prefix="$1"
  if [[ -z "${prefix}" ]]; then
    return 1
  fi

  local finder
  printf -v finder "find %q -maxdepth 6 -type f -iname 'python.exe' 2>/dev/null | sort" "${prefix}/drive_c"
  local results
  results="$(run_as_user "${finder}" || true)"
  if [[ -z "${results}" ]]; then
    return 1
  fi

  local preferred
  preferred="$(printf '%s\n' "${results}" | grep -Ei 'Python3(1[01]|[0-9]+)/python.exe$' | head -n1 || true)"
  if [[ -n "${preferred}" ]]; then
    printf '%s' "${preferred}"
    return 0
  fi

  printf '%s' "${results}" | head -n1
}

to_windows_path() {
  local prefix="$1"
  local unix_path="$2"
  if [[ -z "${prefix}" || -z "${unix_path}" ]]; then
    return 1
  fi

  local converter
  printf -v converter "WINEPREFIX=%q winepath -w %q" "${prefix}" "${unix_path}"
  local converted
  converted="$(run_as_user "${converter}" 2>/dev/null || true)"
  converted="${converted//$'\r'/}"
  [[ -n "${converted}" ]] && printf '%s' "${converted}"
}

ensure_cached_download() {
  local destination="$1"
  local url="$2"
  local label="$3"

  if [[ -s "${destination}" ]]; then
    if [[ "${FORCE_INSTALLER_REFRESH}" != "0" ]]; then
      log "Refreshing cached ${label} at ${destination}"
      rm -f "${destination}"
    else
      log "Using cached ${label} at ${destination}"
      return 0
    fi
  fi

  local tmp_file="${destination}.tmp"
  rm -f "${tmp_file}"
  log "Downloading ${label} from ${url}"
  if ! curl -fsSL --retry 3 --retry-delay 2 -o "${tmp_file}" "${url}"; then
    rm -f "${tmp_file}"
    die "Failed to download ${label}"
  fi
  mv "${tmp_file}" "${destination}"
}

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
    --fresh-installers) FORCE_INSTALLER_REFRESH=1 ;;
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
  local cmd="$*"
  if [[ -z "${cmd}" ]]; then
    return 0
  fi

  if [[ "${HEADLESS_MODE}" == "manual" ]]; then
    run_as_user "DISPLAY='${DISPLAY}' bash -lc \"${cmd}\""
  else
    run_as_user "xvfb-run -a bash -lc \"${cmd}\""
  fi
}

#####################################
# Wine helpers
#####################################
wine_env_block() { echo "export WINEARCH='${WINEARCH}'; export WINEDEBUG='${WINEDEBUG}'"; }
ensure_wineprefix() { run_as_user "$(wine_env_block); export WINEPREFIX='$1'; wineboot -u >/dev/null 2>&1 || true"; }
winetricks_quiet() {
  local prefix="$1"
  shift || true
  [[ $# -eq 0 ]] && return 0
  local cmd
  printf -v cmd "%s; export WINEPREFIX='%s'; winetricks -q -f %s >/dev/null 2>&1 || true" "$(wine_env_block)" "${prefix}" "$*"
  run_as_user "${cmd}"
}
wine_wait() { run_as_user "wineserver -w"; }
wine_cmd() {
  local prefix="$1"
  shift || true
  local quoted
  if [[ $# -eq 0 ]]; then
    return 0
  fi
  printf -v quoted "%q " "$@"
  quoted="${quoted% }"
  run_as_user "$(wine_env_block); export WINEPREFIX='${prefix}'; ${quoted}"
}

#####################################
# Install Windows Python
#####################################
install_windows_python() {
  local prefix="${WINEPREFIX_PY}"
  log "Initialising Wine prefix at ${prefix} ..."
  ensure_wineprefix "${prefix}"
  winetricks_quiet "${prefix}" vcrun2022 corefonts gdiplus

  WINDOWS_PYTHON_UNIX_PATH="$(discover_windows_python "${prefix}" || true)"
  if [[ -n "${WINDOWS_PYTHON_UNIX_PATH}" ]]; then
    WINDOWS_PYTHON_WIN_PATH="$(to_windows_path "${prefix}" "${WINDOWS_PYTHON_UNIX_PATH}" || true)"
    log "Windows Python already present at ${WINDOWS_PYTHON_WIN_PATH:-${WINDOWS_PYTHON_UNIX_PATH}}"
    return 0
  fi

  mkdir -p "${CACHE_DIR}"
  ensure_cached_download \
    "${CACHE_DIR}/${PYTHON_WIN_EXE}" \
    "${PYTHON_WIN_URL}" \
    "Windows Python ${PYTHON_WIN_VERSION} installer"
  run_as_user "mkdir -p '${prefix}/drive_c/_installers'"
  cp -f "${CACHE_DIR}/${PYTHON_WIN_EXE}" "${prefix}/drive_c/_installers/"
  log "Installing Windows Python ${PYTHON_WIN_VERSION}..."
  xvfb_start
  local installer_path="${prefix}/drive_c/_installers/${PYTHON_WIN_EXE}"
  local install_cmd
  printf -v install_cmd "%s; export WINEPREFIX='%s'; wine start /wait /unix '%s' InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=1 /quiet" \
    "$(wine_env_block)" "${prefix}" "${installer_path}"
  with_display "${install_cmd} || true"
  wine_wait
  xvfb_stop

  WINDOWS_PYTHON_UNIX_PATH="$(discover_windows_python "${prefix}" || true)"
  if [[ -z "${WINDOWS_PYTHON_UNIX_PATH}" ]]; then
    log "Windows Python not found after installation"
    return 1
  fi

  WINDOWS_PYTHON_WIN_PATH="$(to_windows_path "${prefix}" "${WINDOWS_PYTHON_UNIX_PATH}" || true)"
  if [[ -z "${WINDOWS_PYTHON_WIN_PATH}" ]]; then
    log "Unable to convert ${WINDOWS_PYTHON_UNIX_PATH} to a Windows path"
    return 1
  fi

  log "Windows Python available at ${WINDOWS_PYTHON_WIN_PATH}"
  wine_cmd "${prefix}" wine cmd /c "${WINDOWS_PYTHON_WIN_PATH}" -V || log "Windows Python not found"
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
  ensure_cached_download \
    "${CACHE_DIR}/${MT5_SETUP_EXE}" \
    "${MT5_SETUP_URL}" \
    "MetaTrader 5 installer"
  if wine_cmd "${prefix}" wine cmd /c "dir C:\\Program^ Files\\MetaTrader^ 5" >/dev/null 2>&1; then
    log "MetaTrader 5 already installed."
    return 0
  fi
  run_as_user "mkdir -p '${prefix}/drive_c/_installers'"
  cp -f "${CACHE_DIR}/${MT5_SETUP_EXE}" "${prefix}/drive_c/_installers/"
  log "Installing MetaTrader 5..."
  xvfb_start
  local mt5_installer="${prefix}/drive_c/_installers/${MT5_SETUP_EXE}"
  local mt5_cmd
  printf -v mt5_cmd "%s; export WINEPREFIX='%s'; wine start /wait /unix '%s' /silent" "$(wine_env_block)" "${prefix}" "${mt5_installer}"
  local mt5_fallback
  printf -v mt5_fallback "%s; export WINEPREFIX='%s'; wine start /wait /unix '%s'" "$(wine_env_block)" "${prefix}" "${mt5_installer}"
  with_display "${mt5_cmd} || ${mt5_fallback}"
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
  local python_cmd
  python_cmd="${WINDOWS_PYTHON_WIN_PATH:-C:\\Python311\\python.exe}"
  cat > "${file}" <<TXT
MetaTrader 5 (Wine) — Quick Usage
---------------------------------
1) Start MT5:
   WINEPREFIX="\$HOME/.wine-mt5" wine "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

   First time: Login → Save password

2) Windows Python:
   WINEPREFIX="\$HOME/.wine-py311" wine cmd /c "${python_cmd}" -V

3) Run bridge:
   rsync -a --delete /opt/mt5/ "\$HOME/.wine-py311/drive_c/mt5/"
   WINEPREFIX="\$HOME/.wine-py311" wine cmd /c "${python_cmd}" C:\\mt5\\utils\\mt_5_bridge.py
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
  auto_configure_terminal
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
