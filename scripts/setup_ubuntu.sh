#!/usr/bin/env bash
# setup_ubuntu.sh — MT5 + Wine + Windows Python bootstrapper
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
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=./_python_version_config.sh
source "${SCRIPT_DIR}/_python_version_config.sh"

PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SETUP_LOG="${PROJECT_ROOT}/setup.log"

PROJECT_USER="${SUDO_USER:-$(whoami)}"
PROJECT_HOME="$(eval echo "~${PROJECT_USER}")"

WINEPREFIX_PY="${WINEPREFIX_PY:-${PROJECT_HOME}/${MT5_PYTHON_PREFIX_NAME}}"
WINEPREFIX_MT5="${PROJECT_HOME}/.wine-mt5"
WINEARCH="win64"

DOWNLOAD_CONNECT_TIMEOUT="${DOWNLOAD_CONNECT_TIMEOUT:-20}"
DOWNLOAD_MAX_TIME="${DOWNLOAD_MAX_TIME:-600}"
PIP_INSTALL_TIMEOUT="${PIP_INSTALL_TIMEOUT:-180}"

HEADLESS_MODE="${HEADLESS_MODE:-auto}"   # auto|manual
PYTHON_WIN_VERSION="${PYTHON_WIN_VERSION:-$MT5_PYTHON_PATCH}"
PYTHON_WIN_MAJOR="${PYTHON_WIN_VERSION%%.*}"
PYTHON_WIN_REMAINDER="${PYTHON_WIN_VERSION#${PYTHON_WIN_MAJOR}.}"
PYTHON_WIN_MINOR="${PYTHON_WIN_REMAINDER%%.*}"
PYTHON_WIN_TAG="${PYTHON_WIN_MAJOR}${PYTHON_WIN_MINOR}"
PYTHON_WIN_TARGET_DIR="${PYTHON_WIN_TARGET_DIR:-$MT5_PYTHON_WIN_DIR}"
PYTHON_WIN_EXE="${PYTHON_WIN_EXE:-$MT5_PYTHON_INSTALLER}"
PYTHON_WIN_EMBED_ZIP="${PYTHON_WIN_EMBED_ZIP:-$MT5_PYTHON_EMBED_ZIP}"
PYTHON_WIN_URL="${PYTHON_WIN_URL:-$MT5_PYTHON_DOWNLOAD_ROOT/${PYTHON_WIN_EXE}}"
PYTHON_WIN_EMBED_URL="${PYTHON_WIN_EMBED_URL:-$MT5_PYTHON_DOWNLOAD_ROOT/${PYTHON_WIN_EMBED_ZIP}}"
# Populated dynamically after installation so we do not rely on a hard coded
# path other than the helper’s configured target.
WINDOWS_PYTHON_UNIX_PATH=""
WINDOWS_PYTHON_WIN_PATH=""
WINDOWS_PYTHON_DEFAULT_WIN_PATH="${PYTHON_WIN_TARGET_DIR}\\python.exe"
PYMT5LINUX_SOURCE="${PYMT5LINUX_SOURCE:-${PROJECT_ROOT}}"
MT5_SETUP_URL="${MT5_SETUP_URL:-https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5setup.exe}"

CACHE_DIR="${PROJECT_ROOT}/.cache/mt5"
MT5_INSTALLER_STAGE="${CACHE_DIR}/installer"
MT5_SETUP_EXE="mt5setup.exe"
FORCE_INSTALLER_REFRESH="${FORCE_INSTALLER_REFRESH:-0}"

export WINEDEBUG="${WINE_DEBUG_CHANNEL:--all}"
DEFAULT_WINEDLLOVERRIDES="ucrtbase=n,b;vcruntime140=n,b;vcruntime140_1=n,b;msvcp140=n,b"

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
  # The per-user Windows Python installer places the interpreter below
  # ``Users/<name>/AppData/Local/Programs/Python/<version>/python.exe`` which
  # exceeds the previous ``-maxdepth 6`` search limit. Recent installers rely on
  # that layout, so widen the traversal depth to ensure we pick it up while
  # still avoiding an unbounded crawl of the Wine prefix.
  printf -v finder "find %q -maxdepth 9 -type f -iname 'python.exe' 2>/dev/null | sort" "${prefix}/drive_c"
  local results
  results="$(run_as_user "${finder}" || true)"
  if [[ -z "${results}" ]]; then
    return 1
  fi

  local preferred
  preferred="$(printf '%s\n' "${results}" | grep -Ei 'Python3[0-9]{2}/python.exe$' | head -n1 || true)"
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
  if ! curl -fsSL --retry 3 --retry-delay 2 \
      --connect-timeout "${DOWNLOAD_CONNECT_TIMEOUT}" \
      --max-time "${DOWNLOAD_MAX_TIME}" \
      -o "${tmp_file}" "${url}"; then
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
wine_env_block() {
  local overrides="${WINEDLLOVERRIDES:-${DEFAULT_WINEDLLOVERRIDES}}"
  printf "export WINEARCH='%s'; export WINEDEBUG='%s'; export WINEDLLOVERRIDES='%s'" \
    "${WINEARCH}" "${WINEDEBUG}" "${overrides}"
}
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

ensure_native_crt_runtime() {
  local prefix="$1"
  shift || true
  if [[ -z "${prefix}" ]]; then
    return 0
  fi

  local -a dlls=(
    "ucrtbase"
    "vcruntime140"
    "vcruntime140_1"
    "msvcp140"
    "api-ms-win-crt-*"
  )

  local dll_path="${prefix}/drive_c/windows/system32/ucrtbase.dll"
  if [[ -f "${dll_path}" ]]; then
    log "Refreshing VC++ runtime overrides for prefix ${prefix}"
  else
    log "Provisioning native VC++ runtime for prefix ${prefix}"
    winetricks_quiet "${prefix}" vcrun2022
  fi

  if [[ ! -f "${dll_path}" ]]; then
    log "Warning: native ucrtbase.dll still missing in ${prefix} after winetricks run"
  fi

  local base_key='HKCU\\Software\\Wine\\DllOverrides'
  for dll in "${dlls[@]}"; do
    wine_cmd "${prefix}" wine reg add "${base_key}" /v "${dll}" /t REG_SZ /d native,builtin /f >/dev/null 2>&1 || true
  done

  local -a target_apps=("python.exe" "terminal64.exe")
  for app in "${target_apps[@]}"; do
    local app_key="HKCU\\Software\\Wine\\AppDefaults\\${app}\\DllOverrides"
    for dll in "${dlls[@]}"; do
      wine_cmd "${prefix}" wine reg add "${app_key}" /v "${dll}" /t REG_SZ /d native,builtin /f >/dev/null 2>&1 || true
    done
  done
}

#####################################
# Install Windows Python
#####################################
install_windows_python() {
  local prefix="${WINEPREFIX_PY}"
  local helper="${PROJECT_ROOT}/scripts/install_windows_python.sh"

  if [[ ! -f "${helper}" ]]; then
    die "Missing helper script at ${helper}"
  fi

  mkdir -p "${CACHE_DIR}"
  ensure_cached_download \
    "${CACHE_DIR}/${PYTHON_WIN_EXE}" \
    "${PYTHON_WIN_URL}" \
    "Windows Python ${PYTHON_WIN_VERSION} installer"

  ensure_cached_download \
    "${CACHE_DIR}/${PYTHON_WIN_EMBED_ZIP}" \
    "${PYTHON_WIN_EMBED_URL}" \
    "Windows Python ${PYTHON_WIN_VERSION} embeddable ZIP"

  local use_xvfb=1
  if [[ "${HEADLESS_MODE}" == "manual" ]]; then
    use_xvfb=0
  fi

  log "Invoking install_windows_python.sh for prefix ${prefix} ..."
  run_as_user \
    "cd '${PROJECT_ROOT}' && PY_WIN_VERSION='${PYTHON_WIN_VERSION}' PY_WIN_EXE_CACHE='${CACHE_DIR}/${PYTHON_WIN_EXE}' PY_WIN_ZIP_CACHE='${CACHE_DIR}/${PYTHON_WIN_EMBED_ZIP}' PY_WIN_DIR='${PYTHON_WIN_TARGET_DIR}' WINEPREFIX='${prefix}' WINEARCH='${WINEARCH}' USE_XVFB='${use_xvfb}' bash '${helper}'"

  WINDOWS_PYTHON_UNIX_PATH="$(discover_windows_python "${prefix}" || true)"
  if [[ -z "${WINDOWS_PYTHON_UNIX_PATH}" ]]; then
    log "Windows Python install helper completed but interpreter was not found"
    return 1
  fi

  WINDOWS_PYTHON_WIN_PATH="$(to_windows_path "${prefix}" "${WINDOWS_PYTHON_UNIX_PATH}" || true)"
  if [[ -z "${WINDOWS_PYTHON_WIN_PATH}" ]]; then
    log "Unable to convert ${WINDOWS_PYTHON_UNIX_PATH} to a Windows path"
    return 1
  fi

  if ! wine_cmd "${prefix}" wine "${WINDOWS_PYTHON_WIN_PATH}" -V >/dev/null 2>&1; then
    log "Windows Python detected at ${WINDOWS_PYTHON_WIN_PATH} but failed to execute"
    return 1
  fi

  winetricks_quiet "${prefix}" gdiplus

  ensure_native_crt_runtime "${prefix}"

  log "Windows Python available at ${WINDOWS_PYTHON_WIN_PATH}"
}

windows_python_win_path() {
  if [[ -n "${WINDOWS_PYTHON_WIN_PATH}" ]]; then
    printf '%s' "${WINDOWS_PYTHON_WIN_PATH}"
  else
    printf '%s' "${WINDOWS_PYTHON_DEFAULT_WIN_PATH}"
  fi
}

verify_windows_python() {
  local prefix="${WINEPREFIX_PY}"
  local python_path
  python_path="$(windows_python_win_path)"

  log "Verifying Windows Python interpreter..."
  if ! wine_cmd "${prefix}" wine "${python_path}" -V >/dev/null 2>&1; then
    die "Windows Python install failed; aborting before mt5/terminal setup"
  fi

  log "Windows Python interpreter available at ${python_path}"
}

install_windows_python_packages() {
  local prefix="${WINEPREFIX_PY}"
  local python_path
  python_path="$(windows_python_win_path)"

  local bridge_backend="${MT5_BRIDGE_BACKEND:-}"
  bridge_backend="${bridge_backend,,}"
  if [[ "${bridge_backend}" == "mql5" || "${bridge_backend}" == "grpc" ]]; then
    log "Skipping Windows mt5 dependency installation (bridge backend: ${bridge_backend})."
    return 0
  fi

  log "Upgrading pip inside Windows Python..."
  if ! wine_cmd "${prefix}" wine "${python_path}" -m pip install -U pip; then
    die "Failed to upgrade pip inside Windows Python"
  fi

  local mt5_requirement
  if ! mt5_requirement="$(resolve_mt5_requirement)" || [[ -z "${mt5_requirement}" ]]; then
    log "Unable to resolve preferred mt5 version; falling back to unconstrained install"
    mt5_requirement="mt5"
  fi

  local -a primary_requirements=("numpy<2.0" "${mt5_requirement}" "MetaTrader5<6")
  log "Installing Windows mt5 dependencies (${primary_requirements[*]})..."

  local -a install_cmd=(
    "env" "PIP_DEFAULT_TIMEOUT=${PIP_INSTALL_TIMEOUT}" "wine" "${python_path}" "-m" "pip" "install"
  )
  install_cmd+=("${primary_requirements[@]}")

  if ! wine_cmd "${prefix}" "${install_cmd[@]}"; then
    log "Primary Windows mt5 dependency install failed; attempting relaxed mt5 constraint..."
    local -a fallback_cmd=(
      "env" "PIP_DEFAULT_TIMEOUT=${PIP_INSTALL_TIMEOUT}" "wine" "${python_path}" "-m" "pip" "install" "--upgrade" "numpy<2.0" "mt5" "MetaTrader5<6"
    )
    if ! wine_cmd "${prefix}" "${fallback_cmd[@]}"; then
      die "Failed to install Windows mt5 dependencies"
    fi
  fi
}

resolve_mt5_requirement() {
  local override="${MT5_WINDOWS_MT5_REQUIREMENT:-}"
  if [[ -n "${override}" ]]; then
    printf '%s' "${override}"
    return 0
  fi

  local resolver
  resolver="$(
python3 - <<'PY' 2>/dev/null || true
import json
import os
import sys
import urllib.request

preferred_min = os.environ.get('MT5_WINDOWS_MT5_MIN_VERSION', '1.26')
preferred_max = os.environ.get('MT5_WINDOWS_MT5_MAX_VERSION', '1.27')
timeout = int(os.environ.get('MT5_PYPI_TIMEOUT', '15'))

def parse_version(tag):
    parts = []
    for chunk in tag.replace('-', '.').split('.'):
        if not chunk:
            continue
        digits = ''.join(ch for ch in chunk if ch.isdigit())
        if not digits:
            return None
        parts.append(int(digits))
    return tuple(parts) if parts else None

def compare(left, right):
    for index in range(max(len(left), len(right))):
        l = left[index] if index < len(left) else 0
        r = right[index] if index < len(right) else 0
        if l != r:
            return (l > r) - (l < r)
    return 0

def in_range(parsed):
    minimum = parse_version(preferred_min) if preferred_min else None
    maximum = parse_version(preferred_max) if preferred_max else None
    if minimum is not None and compare(parsed, minimum) < 0:
        return False
    if maximum is not None and compare(parsed, maximum) >= 0:
        return False
    return True

try:
    with urllib.request.urlopen('https://pypi.org/pypi/mt5/json', timeout=timeout) as response:
        payload = json.load(response)
except Exception:
    sys.exit(1)

releases = payload.get('releases') or {}
parsed = []
for version, files in releases.items():
    if not files:
        continue
    parsed_version = parse_version(version)
    if parsed_version is None:
        continue
    parsed.append((parsed_version, version))

if not parsed:
    sys.exit(1)

parsed.sort(reverse=True)

for current in parsed:
    if in_range(current[0]):
        print(f"mt5=={current[1]}")
        break
else:
    print(f"mt5=={parsed[0][1]}")
PY
)"

  if [[ -n "${resolver}" ]]; then
    printf '%s' "${resolver}"
    return 0
  fi

  return 1
}

#####################################
# Install MetaTrader 5
#####################################
install_mt5() {
  local prefix="${WINEPREFIX_MT5}"
  log "Initialising Wine prefix at ${prefix} ..."
  ensure_wineprefix "${prefix}"
  winetricks_quiet "${prefix}" corefonts gdiplus
  ensure_native_crt_runtime "${prefix}"
  if wine_cmd "${prefix}" wine cmd /c "dir C:\\Program^ Files\\MetaTrader^ 5" >/dev/null 2>&1; then
    log "MetaTrader 5 already installed."
    return 0
  fi
  run_as_user "mkdir -p '${CACHE_DIR}' '${MT5_INSTALLER_STAGE}'"

  local installer_script="${MT5_INSTALLER_STAGE}/mt5linux.sh"
  local installer_url="https://download.mql5.com/cdn/web/metaquotes.software.corp/mt5/mt5linux.sh"

  log "Downloading MetaTrader 5 installer script from ${installer_url}..."
  run_as_user "rm -f '${installer_script}'"
  local download_cmd
  printf -v download_cmd "cd %q && wget --tries=3 --timeout=%q --waitretry=2 --retry-connrefused --retry-on-http-error=429,500,502,503,504 -O mt5linux.sh %q" \
    "${MT5_INSTALLER_STAGE}" "${DOWNLOAD_CONNECT_TIMEOUT}" "${installer_url}"
  run_as_user "${download_cmd}"

  local chmod_cmd
  printf -v chmod_cmd "cd %q && chmod +x mt5linux.sh" "${MT5_INSTALLER_STAGE}"
  run_as_user "${chmod_cmd}"

  INSTALL_STAGE="${MT5_INSTALLER_STAGE}" MT5_PREFIX="${prefix}" python3 - <<'PY'
import os
from pathlib import Path

stage_root = Path(os.environ["INSTALL_STAGE"])
script_path = stage_root / "mt5linux.sh"
if not script_path.exists():
    raise SystemExit("MetaTrader installer script download failed")

text = script_path.read_text()
start_marker = "echo Update and install..."
end_marker = "echo Download MetaTrader and WebView2 Runtime"
start = text.find(start_marker)
end = text.find(end_marker)
if start != -1 and end != -1 and end > start:
    text = text[:start] + text[end:]

prefix = os.environ["MT5_PREFIX"]
text = text.replace("WINEPREFIX=~/.mt5", f"WINEPREFIX='{prefix}'")

script_path.write_text(text)
script_path.chmod(0o755)
PY
  chown "${PROJECT_USER}:${PROJECT_USER}" "${installer_script}" || true

  log "Running MetaTrader 5 installer script..."
  xvfb_start
  local install_cmd
  printf -v install_cmd "cd %q && ./mt5linux.sh" "${MT5_INSTALLER_STAGE}"
  with_display "${install_cmd}"
  wine_wait
  xvfb_stop

  local cleanup_cmd
  printf -v cleanup_cmd "cd %q && rm -f mt5setup.exe webview2.exe" "${MT5_INSTALLER_STAGE}"
  run_as_user "${cleanup_cmd}" || true

  if attempt_mt5_bridge_via_package; then
    log "MetaTrader5 Python package successfully established the bridge."
  else
    log "MetaTrader5 package bridge failed; deploying MQL bridge assets as fallback."
    install_mql_bridge_assets || log "Fallback MQL bridge deployment encountered an error"
  fi
}

attempt_mt5_bridge_via_package() {
  local python_prefix="${WINEPREFIX_PY}"
  local python_path
  python_path="$(windows_python_win_path)"

  if [[ -z "${python_path}" ]]; then
    log "Skipping MetaTrader5 package bridge attempt (Windows Python path unavailable)."
    return 1
  fi

  local terminal_unix="${WINEPREFIX_MT5}/drive_c/Program Files/MetaTrader 5/terminal64.exe"
  if [[ ! -f "${terminal_unix}" ]]; then
    log "MetaTrader terminal not found at ${terminal_unix}; cannot attempt package bridge."
    return 1
  fi

  local terminal_win
  terminal_win="$(to_windows_path "${python_prefix}" "${terminal_unix}" || true)"
  if [[ -z "${terminal_win}" ]]; then
    log "Failed to translate ${terminal_unix} for Windows; cannot attempt package bridge."
    return 1
  fi

  local probe_host="${python_prefix}/drive_c/mt5_bridge_probe.py"
  local writer
  printf -v writer "python3 - <<'PY'\nfrom pathlib import Path\nscript = Path(r\"%s\")\nscript.write_text('''import json\\nimport os\\nimport sys\\nimport time\\n\\nimport MetaTrader5 as mt5\\n\\nTIMEOUT = int(os.environ.get(\"MT5_BRIDGE_TIMEOUT_MS\", \"90000\"))\\nTERMINAL_PATH = os.environ.get(\"MT5_TERMINAL_PATH\")\\nif not TERMINAL_PATH:\\n    print(json.dumps({\"status\": \"fail\", \"error\": \"missing terminal path\"}))\\n    sys.exit(1)\\nif not mt5.initialize(path=TERMINAL_PATH, timeout=TIMEOUT):\\n    code, message = mt5.last_error()\\n    print(json.dumps({\"status\": \"fail\", \"error\": [code, message]}))\\n    sys.exit(1)\\ntime.sleep(1)\\nmt5.shutdown()\\nprint(json.dumps({\"status\": \"ok\"}))\\n''', encoding='utf-8')\nPY" "${probe_host}"
  if ! run_as_user "${writer}"; then
    log "Unable to prepare MetaTrader bridge probe script."
    return 1
  fi

  local python_cmd
  printf -v python_cmd "MT5_TERMINAL_PATH=%q MT5_BRIDGE_TIMEOUT_MS=%q WINEPREFIX=%q wine %q %q" \
    "${terminal_win}" "90000" "${python_prefix}" "${python_path}" "C:\\mt5_bridge_probe.py"

  if run_as_user "${python_cmd}"; then
    run_as_user "rm -f '${probe_host}'" || true
    return 0
  fi

  run_as_user "rm -f '${probe_host}'" || true
  return 1
}

install_mql_bridge_assets() {
  local source_dir="${PROJECT_ROOT}/mt5_bridge_files/MQL5"
  local target_dir="${WINEPREFIX_MT5}/drive_c/Program Files/MetaTrader 5/MQL5"

  if [[ ! -d "${source_dir}" ]]; then
    log "MQL bridge asset directory ${source_dir} missing; skipping fallback deployment."
    return 1
  fi

  local copier
  printf -v copier "SOURCE=%q DEST=%q python3 - <<'PY'\nimport os\nimport shutil\nfrom pathlib import Path\n\nsource = Path(os.environ['SOURCE'])\ndest = Path(os.environ['DEST'])\n\nif not source.exists():\n    raise SystemExit(1)\nfor path in source.rglob('*'):\n    rel = path.relative_to(source)\n    target = dest / rel\n    if path.is_dir():\n        target.mkdir(parents=True, exist_ok=True)\n    else:\n        target.parent.mkdir(parents=True, exist_ok=True)\n        shutil.copy2(path, target)\nprint(f"Copied MQL bridge assets to {dest}")\nPY" "${source_dir}" "${target_dir}"

  if run_as_user "${copier}"; then
    return 0
  fi

  return 1
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
  python_cmd="${WINDOWS_PYTHON_WIN_PATH:-${MT5_PYTHON_WIN_DIR}\\python.exe}"
  cat > "${file}" <<TXT
MetaTrader 5 (Wine) — Quick Usage
---------------------------------
1) Start MT5:
   WINEPREFIX="\$HOME/.wine-mt5" wine "C:\\Program Files\\MetaTrader 5\\terminal64.exe"

   First time: Login → Save password

2) Windows Python:
   WINEPREFIX="\$HOME/${MT5_PYTHON_PREFIX_NAME}" wine cmd /c "${python_cmd}" -V

3) Run bridge:
   rsync -a --delete /opt/mt5/ "\$HOME/${MT5_PYTHON_PREFIX_NAME}/drive_c/mt5/"
   WINEPREFIX="\$HOME/${MT5_PYTHON_PREFIX_NAME}" wine cmd /c "${python_cmd}" C:\\mt5\\utils\\mt_5_bridge.py
TXT
  log "Instructions written to ${file}"
}

#####################################
# Main
#####################################
main() {
  ensure_system_packages
  ensure_project_venv
  if ! install_windows_python; then
    die "Windows Python install failed"
  fi
  verify_windows_python
  install_windows_python_packages
  install_linux_bridge
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
