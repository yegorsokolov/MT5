#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_python_version_config.sh
source "${SCRIPT_DIR}/scripts/_python_version_config.sh"

PROJECT_ROOT="${PROJECT_ROOT:-${SCRIPT_DIR}}"
# shellcheck source=scripts/_mt5linux_env.sh
source "${PROJECT_ROOT}/scripts/_mt5linux_env.sh"

MT5LINUX_WINDOWS_CONSTRAINTS_DEFAULT="${PROJECT_ROOT}/constraints-mt5linux.txt"
MT5LINUX_WINDOWS_CONSTRAINTS="${MT5LINUX_WINDOWS_CONSTRAINTS:-${MT5LINUX_WINDOWS_CONSTRAINTS_DEFAULT}}"
BRIDGE_BACKEND_DEFAULT="mt5linux"
BRIDGE_BACKEND="${MT5_BRIDGE_SERVER:-${BRIDGE_BACKEND_DEFAULT}}"
BRIDGE_BACKEND_EXPLICIT="0"

log() { printf '[bridge] %s\n' "$*" >&2; }
warn() { printf '[bridge:WARN] %s\n' "$*" >&2; }
error() { printf '[bridge:ERROR] %s\n' "$*" >&2; }
die() { error "$*"; exit 1; }

resolve_wine_prefix() {
  local explicit="$1" default_path="$2" fallback="$3" label="$4"

  if [[ -n "$explicit" ]]; then
    printf '%s\n' "$explicit"
    return 0
  fi

  if [[ -n "$default_path" && -d "$default_path" ]]; then
    printf '%s\n' "$default_path"
    return 0
  fi

  if [[ -n "$fallback" && -d "$fallback" ]]; then
    log "Auto-detected ${label} Wine prefix at ${fallback}"
    printf '%s\n' "$fallback"
    return 0
  fi

  printf '%s\n' "$default_path"
}

PY_WINE_PREFIX_DEFAULT="${HOME}/${MT5_PYTHON_PREFIX_NAME}"
MT5_WINE_PREFIX_DEFAULT="${HOME}/.mt5"
PY_WINE_PREFIX_FALLBACK="${HOME}/.wine-mt5"
MT5_WINE_PREFIX_FALLBACK="${HOME}/.wine-mt5"
DEFAULT_TIMEOUT_MS="90000"
DEFAULT_PIP_TIMEOUT="180"

PY_WINE_PREFIX="$(resolve_wine_prefix "${PY_WINE_PREFIX:-}" "$PY_WINE_PREFIX_DEFAULT" "$PY_WINE_PREFIX_FALLBACK" "Python")"
MT5_WINE_PREFIX="$(resolve_wine_prefix "${MT5_WINE_PREFIX:-}" "$MT5_WINE_PREFIX_DEFAULT" "$MT5_WINE_PREFIX_FALLBACK" "MetaTrader")"
export PY_WINE_PREFIX MT5_WINE_PREFIX
WIN_PYTHON="${WIN_PYTHON:-}"
MT5_TERMINAL="${MT5_TERMINAL:-}"
BRIDGE_TIMEOUT_MS="${MT5_BRIDGE_TIMEOUT_MS:-${DEFAULT_TIMEOUT_MS}}"
PIP_TIMEOUT="${PIP_TIMEOUT:-${DEFAULT_PIP_TIMEOUT}}"
MT5LINUX_HOST_DEFAULT="127.0.0.1"
MT5LINUX_PORT_DEFAULT="18812"
MT5LINUX_SERVER_DIR_DEFAULT="${PY_WINE_PREFIX}/drive_c/mt5linux-server"
MT5LINUX_HOST="${MT5LINUX_HOST:-${MT5LINUX_HOST_DEFAULT}}"
MT5LINUX_PORT="${MT5LINUX_PORT:-${MT5LINUX_PORT_DEFAULT}}"
MT5LINUX_SERVER_DIR="${MT5LINUX_SERVER_DIR:-${MT5LINUX_SERVER_DIR_DEFAULT}}"
MT5LINUX_SERVER_WINPATH=""
MT5LINUX_VENV_PATH_DEFAULT="${PROJECT_ROOT}/.mt5linux-venv"
MT5LINUX_VENV_PATH="${MT5LINUX_VENV_PATH:-${MT5LINUX_VENV_PATH_DEFAULT}}"
MT5LINUX_BOOTSTRAP_PYTHON="${MT5LINUX_BOOTSTRAP_PYTHON:-}"
MT5LINUX_PYTHON="${MT5LINUX_PYTHON:-}"

if [[ -z "${MT5LINUX_PACKAGE:-}" ]]; then
  case "${MT5_PYTHON_SERIES}" in
    3.10) MT5LINUX_PACKAGE="mt5linux==0.1.7" ;;
    3.11) MT5LINUX_PACKAGE="mt5linux==0.1.9" ;;
    *) MT5LINUX_PACKAGE="mt5linux" ;;
  esac
fi

usage() {
  cat <<'USAGE'
Usage: install_programmatic_bridge.sh [options]

Options:
  --py-wine-prefix PATH     Wine prefix containing Windows Python (default: $PY_WINE_PREFIX)
  --mt5-wine-prefix PATH    Wine prefix containing MetaTrader terminal (default: $MT5_WINE_PREFIX)
  --win-python PATH         Override path to Windows python.exe inside the Wine prefix
  --terminal PATH           Override path to terminal64.exe inside the MT5 prefix
  --bridge-backend NAME     Bridge server implementation to launch (mt5linux or mt5)
  --host HOST               Hostname for the mt5linux RPyC server (default: $MT5LINUX_HOST)
  --port PORT               TCP port for the mt5linux RPyC server (default: $MT5LINUX_PORT)
  --server-dir PATH         Directory inside the Wine prefix for mt5linux server assets (default: $MT5LINUX_SERVER_DIR)
  --timeout-ms N            Override MetaTrader bridge timeout in milliseconds (default: $BRIDGE_TIMEOUT_MS)
  --pip-timeout N           Override pip network timeout in seconds (default: $PIP_TIMEOUT)
  -h, --help                Show this help message
USAGE
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --py-wine-prefix)
        [[ $# -ge 2 ]] || die "--py-wine-prefix requires a value"
        PY_WINE_PREFIX="$2"
        shift 2
        ;;
      --mt5-wine-prefix)
        [[ $# -ge 2 ]] || die "--mt5-wine-prefix requires a value"
        MT5_WINE_PREFIX="$2"
        shift 2
        ;;
      --win-python)
        [[ $# -ge 2 ]] || die "--win-python requires a value"
        WIN_PYTHON="$2"
        shift 2
        ;;
      --terminal)
        [[ $# -ge 2 ]] || die "--terminal requires a value"
        MT5_TERMINAL="$2"
        shift 2
        ;;
      --bridge-backend)
        [[ $# -ge 2 ]] || die "--bridge-backend requires a value"
        BRIDGE_BACKEND="$2"
        BRIDGE_BACKEND_EXPLICIT="1"
        shift 2
        ;;
      --host)
        [[ $# -ge 2 ]] || die "--host requires a value"
        MT5LINUX_HOST="$2"
        shift 2
        ;;
      --port)
        [[ $# -ge 2 ]] || die "--port requires a value"
        MT5LINUX_PORT="$2"
        shift 2
        ;;
      --server-dir)
        [[ $# -ge 2 ]] || die "--server-dir requires a value"
        MT5LINUX_SERVER_DIR="$2"
        shift 2
        ;;
      --timeout-ms)
        [[ $# -ge 2 ]] || die "--timeout-ms requires a value"
        BRIDGE_TIMEOUT_MS="$2"
        shift 2
        ;;
      --pip-timeout)
        [[ $# -ge 2 ]] || die "--pip-timeout requires a value"
        PIP_TIMEOUT="$2"
        shift 2
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      --)
        shift
        break
        ;;
      *)
        die "Unknown argument: $1"
        ;;
    esac
  done
}

require_command() {
  local cmd="$1"
  command -v "$cmd" >/dev/null 2>&1 || die "Required command not found: $cmd"
}

with_wine_env() {
  local prefix="$1"
  shift
  local overrides="ucrtbase=native,builtin;api-ms-win-crt-*=native,builtin"
  WINEDLLOVERRIDES="$overrides" WINEPREFIX="$prefix" "$@"
}

run_wine() {
  local prefix="$1"
  shift
  with_wine_env "$prefix" wine "$@"
}

to_windows_path() {
  local prefix="$1" path="$2"
  with_wine_env "$prefix" winepath -w "$path" 2>/dev/null | tr -d '\r'
}

detect_windows_python() {
  local prefix="$1"
  local override="$2"
  if [[ -n "$override" ]]; then
    [[ -f "$override" ]] || die "Windows Python override not found at $override"
    echo "$override"
    return 0
  fi

  local candidate
  local versions=("${MT5_PYTHON_TAG}" 313 312 311 310 39 38)
  for version in "${versions[@]}"; do
    candidate="$prefix/drive_c/Python${version}/python.exe"
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
    candidate="$prefix/drive_c/Program Files/Python${version}/python.exe"
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
    candidate="$prefix/drive_c/Program Files/Python/Python${version}/python.exe"
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
    candidate="$prefix/drive_c/Program Files (x86)/Python/Python${version}/python.exe"
    if [[ -f "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  while IFS= read -r -d '' candidate; do
    [[ -f "$candidate" ]] || continue
    echo "$candidate"
    return 0
  done < <(find "$prefix" -maxdepth 5 -type f -iname 'python.exe' -print0 2>/dev/null)

  die "Unable to locate python.exe inside Wine prefix $prefix"
}

detect_terminal() {
  local prefix="$1"
  local override="$2"
  if [[ -n "$override" ]]; then
    [[ -f "$override" ]] || die "MetaTrader terminal override not found at $override"
    echo "$override"
    return 0
  fi

  local candidate="$prefix/drive_c/Program Files/MetaTrader 5/terminal64.exe"
  if [[ -f "$candidate" ]]; then
    echo "$candidate"
    return 0
  fi

  while IFS= read -r -d '' candidate; do
    [[ -f "$candidate" ]] || continue
    echo "$candidate"
    return 0
  done < <(find "$prefix" -maxdepth 6 -type f -iname 'terminal64.exe' -print0 2>/dev/null)

  die "Unable to locate terminal64.exe inside Wine prefix $prefix"
}

install_windows_packages() {
  log "Ensuring Windows pip is up-to-date..."
  if ! PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" run_wine "$PY_WINE_PREFIX" "$WIN_PYTHON_WINPATH" -m pip install --upgrade pip 1>&2; then
    die "Failed to upgrade pip in Windows environment"
  fi

  local req_source=""
  if [[ -n "${MT5LINUX_WINDOWS_CONSTRAINTS:-}" && -f "$MT5LINUX_WINDOWS_CONSTRAINTS" ]]; then
    req_source="$MT5LINUX_WINDOWS_CONSTRAINTS"
  elif [[ -f "${MT5LINUX_LOCK_FILE:-}" ]]; then
    req_source="$MT5LINUX_LOCK_FILE"
  fi

  if [[ "$BRIDGE_BACKEND" == "mt5linux" && -n "$req_source" ]]; then
    local req_winpath
    req_winpath="$(to_windows_path "$PY_WINE_PREFIX" "$req_source")"
    [[ -n "$req_winpath" ]] || die "Unable to translate mt5linux dependency lock file path"
    log "Pre-installing mt5linux dependency pins from $(basename "$req_source")"
    if ! PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" run_wine "$PY_WINE_PREFIX" "$WIN_PYTHON_WINPATH" -m pip install --upgrade --no-deps --only-binary :all: -r "$req_winpath" 1>&2; then
      log "Binary wheel installation for mt5linux dependency pins failed; retrying without --only-binary"
      PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" run_wine "$PY_WINE_PREFIX" "$WIN_PYTHON_WINPATH" -m pip install --upgrade --no-deps -r "$req_winpath" 1>&2 || die "Failed to install mt5linux dependency pins in Windows environment"
    fi
  elif [[ "$BRIDGE_BACKEND" == "mt5linux" ]]; then
    warn "mt5linux dependency lock file not found; skipping pre-install step"
  fi

  log "Ensuring MetaTrader5 wheel is installed"
  if ! PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" run_wine "$PY_WINE_PREFIX" "$WIN_PYTHON_WINPATH" -m pip install --upgrade --only-binary :all: MetaTrader5 1>&2; then
    log "Binary wheel installation for MetaTrader5 failed; retrying without --only-binary"
    PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" run_wine "$PY_WINE_PREFIX" "$WIN_PYTHON_WINPATH" -m pip install --upgrade MetaTrader5 1>&2 || die "Failed to install MetaTrader5 in Windows environment"
  fi

  if [[ "$BRIDGE_BACKEND" == "mt5linux" ]]; then
    log "Installing ${MT5LINUX_PACKAGE} without dependencies"
    if ! PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" run_wine "$PY_WINE_PREFIX" "$WIN_PYTHON_WINPATH" -m pip install --upgrade --no-deps --only-binary :all: "${MT5LINUX_PACKAGE}" 1>&2; then
      log "Binary wheel installation for ${MT5LINUX_PACKAGE} failed; retrying without --only-binary"
      PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" run_wine "$PY_WINE_PREFIX" "$WIN_PYTHON_WINPATH" -m pip install --upgrade --no-deps "${MT5LINUX_PACKAGE}" 1>&2 || die "Failed to install ${MT5LINUX_PACKAGE} in Windows environment"
    fi
  else
    log "Installing mt5 bridge helper and rpyc in Windows environment"
    if ! PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" run_wine "$PY_WINE_PREFIX" "$WIN_PYTHON_WINPATH" -m pip install --upgrade mt5 rpyc 1>&2; then
      die "Failed to install mt5 bridge requirements in Windows environment"
    fi
  fi
}

install_linux_mt5linux() {
  log "Preparing mt5linux auxiliary environment at ${MT5LINUX_VENV_PATH}"
  if ! refresh_mt5linux_venv "${MT5LINUX_BOOTSTRAP_PYTHON}"; then
    die "Failed to prepare mt5linux auxiliary environment"
  fi

  if ! MT5LINUX_PYTHON="$(mt5linux_env_python_path)"; then
    die "mt5linux auxiliary python interpreter missing from ${MT5LINUX_VENV_PATH}"
  fi
}

stage_bridge_assets() {
  mkdir -p "$MT5LINUX_SERVER_DIR"

  if [[ "$BRIDGE_BACKEND" == "mt5" ]]; then
    local server_src="${PROJECT_ROOT}/mt5_bridge_files/mt5_rpyc_server.py"
    if [[ ! -f "$server_src" ]]; then
      die "Bridge server helper not found at ${server_src}"
    fi
    local server_dst="$MT5LINUX_SERVER_DIR/mt5_rpyc_server.py"
    if ! cp "$server_src" "$server_dst"; then
      die "Failed to copy bridge server helper to ${server_dst}"
    fi
  fi
}

stop_mt5linux_server() {
  local pid_file="$MT5LINUX_SERVER_DIR/mt5linux.pid"
  if [[ -f "$pid_file" ]]; then
    local pid
    pid="$(cat "$pid_file" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" >/dev/null 2>&1; then
      log "Stopping existing mt5linux server (pid $pid)"
      kill "$pid" >/dev/null 2>&1 || true
      sleep 2
    fi
    rm -f "$pid_file"
  fi
}

launch_mt5linux_server() {
  mkdir -p "$MT5LINUX_SERVER_DIR"
  stop_mt5linux_server

  local log_file="$MT5LINUX_SERVER_DIR/mt5linux.log"
  local -a launch_cmd

  if [[ "$BRIDGE_BACKEND" == "mt5linux" ]]; then
    local python_arg_winpath="${WIN_PYTHON_WINPATH}"

    if [[ -n "${MT5LINUX_BOOTSTRAP_PYTHON:-}" ]]; then
      local bootstrap_winpath
      bootstrap_winpath="$(to_windows_path "$PY_WINE_PREFIX" "$MT5LINUX_BOOTSTRAP_PYTHON" 2>/dev/null || true)"
      if [[ -n "$bootstrap_winpath" ]]; then
        python_arg_winpath="$bootstrap_winpath"
      else
        warn "Unable to translate MT5LINUX_BOOTSTRAP_PYTHON path '${MT5LINUX_BOOTSTRAP_PYTHON}' to a Windows path; falling back to $WIN_PYTHON_WINPATH"
      fi
    fi

    log "Launching mt5linux RPyC server at ${MT5LINUX_HOST}:${MT5LINUX_PORT}"
    launch_cmd=(
      nohup
      wine
      "$WIN_PYTHON_WINPATH"
      -m
      mt5linux
      --host
      "$MT5LINUX_HOST"
      --port
      "$MT5LINUX_PORT"
      --server
      "$MT5LINUX_SERVER_DIR"
      "$python_arg_winpath"
    )
  else
    local server_script="$MT5LINUX_SERVER_DIR/mt5_rpyc_server.py"
    local server_winpath
    server_winpath="$(to_windows_path "$PY_WINE_PREFIX" "$server_script")"
    [[ -n "$server_winpath" ]] || die "Unable to translate bridge server helper path"

    local timeout_seconds
    timeout_seconds="$(python3 - <<'PY' "$BRIDGE_TIMEOUT_MS"
import sys
try:
    ms = int(sys.argv[1])
except Exception:  # noqa: BLE001
    ms = 90000
print(max(5.0, ms / 1000.0))
PY
)"

    log "Launching mt5 RPyC helper at ${MT5LINUX_HOST}:${MT5LINUX_PORT}"
    launch_cmd=(
      nohup
      wine
      "$WIN_PYTHON_WINPATH"
      "$server_winpath"
      --host
      "$MT5LINUX_HOST"
      --port
      "$MT5LINUX_PORT"
      --timeout
      "$timeout_seconds"
    )
  fi

  if ! with_wine_env "$PY_WINE_PREFIX" "${launch_cmd[@]}" >>"$log_file" 2>&1 & then
    error "Failed to launch ${BRIDGE_BACKEND} server via Wine"
    return 1
  fi
  local server_pid=$!
  echo "$server_pid" >"$MT5LINUX_SERVER_DIR/mt5linux.pid"
  sleep 5
}

apply_appdll_overrides() {
  local prefix="$1" app="$2"
  local key="HKCU\\Software\\Wine\\AppDefaults\\${app}\\DllOverrides"
  local overrides=("ucrtbase" "vcruntime140" "msvcp140" "api-ms-win-crt-*")
  local value="native,builtin"

  log "Recording native CRT DLL overrides for ${app} in prefix ${prefix}"
  for dll in "${overrides[@]}"; do
    if ! with_wine_env "$prefix" wine reg add "$key" /v "$dll" /t REG_SZ /d "$value" /f >/dev/null 2>&1; then
      warn "Failed to register ${dll} override for ${app} in ${prefix}"
    fi
  done
}

apply_terminal_overrides_if_needed() {
  local prefix="$1" exe_path="$2"

  if [[ -z "$exe_path" ]]; then
    return 0
  fi

  local exe_name="${exe_path##*/}"
  local exe_lower
  exe_lower="$(printf '%s' "$exe_name" | tr '[:upper:]' '[:lower:]')"

  if [[ "$exe_lower" == "terminal64.exe" ]]; then
    apply_appdll_overrides "$prefix" "terminal64.exe"
  fi
}

ensure_native_ucrtbase() {
  local prefix="$1"
  local dll_path="$prefix/drive_c/windows/system32/ucrtbase.dll"

  if [[ -f "$dll_path" ]]; then
    apply_appdll_overrides "$prefix" "python.exe"
    return 0
  fi

  warn "Native ucrtbase.dll not found at ${dll_path}. Attempting winetricks -q vcrun2022..."

  if ! command -v winetricks >/dev/null 2>&1; then
    die "winetricks command not available; install it or run 'WINEPREFIX=\"${prefix}\" winetricks -q vcrun2022' to provision the native UCRT runtime."
  fi

  if ! with_wine_env "$prefix" winetricks -q vcrun2022; then
    die "winetricks -q vcrun2022 failed; rerun manually using 'WINEPREFIX=\"${prefix}\" winetricks -q vcrun2022' before continuing."
  fi

  if [[ ! -f "$dll_path" ]]; then
    die "Native ucrtbase.dll still missing at ${dll_path} after winetricks. Rerun 'WINEPREFIX=\"${prefix}\" winetricks -q vcrun2022' and verify the DLL is present."
  fi

  apply_appdll_overrides "$prefix" "python.exe"
}

run_probe() {
  log "Validating ${BRIDGE_BACKEND} bridge connectivity from Linux Python..."
  local host="$MT5LINUX_HOST" port="$MT5LINUX_PORT" timeout_ms="$BRIDGE_TIMEOUT_MS"
  local probe_python="${MT5LINUX_PYTHON:-}"
  if [[ -z "$probe_python" ]]; then
    die "Probe interpreter not configured (mt5linux auxiliary environment missing?)"
  fi

  if ! "$probe_python" - "$host" "$port" "$timeout_ms" "$BRIDGE_BACKEND" <<'PY'
import sys
import time

host = sys.argv[1]
port = int(sys.argv[2])
timeout_ms = int(sys.argv[3]) if len(sys.argv) > 3 else 90000
backend_name = sys.argv[4] if len(sys.argv) > 4 else "mt5linux"
timeout = max(5.0, timeout_ms / 1000.0)

try:
    import rpyc
except Exception as exc:  # noqa: BLE001
    raise SystemExit(f"rpyc import failed: {exc}") from exc

last_error = None
for attempt in range(1, 11):
    try:
        conn = rpyc.classic.connect(host, port=port, config={
            "sync_request_timeout": timeout,
            "connection_timeout": timeout,
            "allow_public_attrs": True,
        })
        try:
            version = conn.eval("import MetaTrader5 as mt5; getattr(mt5, '__version__', '')")
        finally:
            conn.close()
    except Exception as exc:  # noqa: BLE001
        last_error = exc
        time.sleep(2)
    else:
        print(version or "ok")
        sys.exit(0)

raise SystemExit(f"{backend_name} bridge connection failed: {last_error}")
PY
  then
    error "${BRIDGE_BACKEND} bridge validation failed"
    return 1
  fi
}

smoke_check_environment() {
  if [[ -f "${SCRIPT_DIR}/utils/environment.py" ]]; then
    log "Running python -m utils.environment smoke check"
    if ! (cd "$SCRIPT_DIR" && PYTHONPATH="$SCRIPT_DIR" python3 -m utils.environment); then
      warn "python -m utils.environment reported issues; review the diagnostics above."
    fi
  else
    warn "utils.environment module not found; skipping environment smoke check."
  fi
}

configure_bridge_backend() {
  local backend="$1"
  BRIDGE_BACKEND="$backend"

  log "Using bridge backend: $BRIDGE_BACKEND"
  log "Using mt5linux server directory: $MT5LINUX_SERVER_DIR (winpath: $MT5LINUX_SERVER_WINPATH)"

  install_windows_packages
  install_linux_mt5linux
  ensure_native_ucrtbase "$PY_WINE_PREFIX"
  apply_terminal_overrides_if_needed "$MT5_WINE_PREFIX" "$MT5_TERMINAL"
  stage_bridge_assets

  if ! launch_mt5linux_server; then
    return 1
  fi

  if ! run_probe; then
    return 1
  fi

  smoke_check_environment

  printf 'WIN_PYTHON_WINPATH=%s\n' "$WIN_PYTHON_WINPATH"
  printf 'MT5_TERMINAL_WINPATH=%s\n' "$MT5_TERMINAL_WINPATH"
  printf 'MT5LINUX_HOST=%s\n' "$MT5LINUX_HOST"
  printf 'MT5LINUX_PORT=%s\n' "$MT5LINUX_PORT"
  printf 'MT5LINUX_SERVER_DIR=%s\n' "$MT5LINUX_SERVER_DIR"
}

main() {
  parse_args "$@"

  BRIDGE_BACKEND="${BRIDGE_BACKEND,,}"
  case "$BRIDGE_BACKEND" in
    mt5linux|mt5) ;;
    *)
      die "Unsupported bridge backend '${BRIDGE_BACKEND}'. Choose 'mt5linux' or 'mt5'."
      ;;
  esac

  require_command wine
  require_command winepath
  require_command python3

  [[ -d "$PY_WINE_PREFIX" ]] || die "Python Wine prefix not found: $PY_WINE_PREFIX"
  [[ -d "$MT5_WINE_PREFIX" ]] || die "MetaTrader Wine prefix not found: $MT5_WINE_PREFIX"

  [[ -n "$MT5LINUX_HOST" ]] || die "mt5linux host must not be empty"
  if ! [[ "$MT5LINUX_PORT" =~ ^[0-9]+$ ]]; then
    die "mt5linux port must be an integer"
  fi

  WIN_PYTHON="$(detect_windows_python "$PY_WINE_PREFIX" "$WIN_PYTHON")"
  MT5_TERMINAL="$(detect_terminal "$MT5_WINE_PREFIX" "$MT5_TERMINAL")"

  WIN_PYTHON_WINPATH="$(to_windows_path "$PY_WINE_PREFIX" "$WIN_PYTHON")"
  [[ -n "$WIN_PYTHON_WINPATH" ]] || die "Unable to translate Windows Python path"

  MT5_TERMINAL_WINPATH="$(to_windows_path "$PY_WINE_PREFIX" "$MT5_TERMINAL")"
  [[ -n "$MT5_TERMINAL_WINPATH" ]] || die "Unable to translate MetaTrader terminal path"

  MT5LINUX_SERVER_WINPATH="$(to_windows_path "$PY_WINE_PREFIX" "$MT5LINUX_SERVER_DIR")"
  [[ -n "$MT5LINUX_SERVER_WINPATH" ]] || die "Unable to translate mt5linux server directory path"

  log "Discovered Windows Python: $WIN_PYTHON (winpath: $WIN_PYTHON_WINPATH)"
  log "Discovered MetaTrader terminal: $MT5_TERMINAL (winpath: $MT5_TERMINAL_WINPATH)"

  local requested_backend="$BRIDGE_BACKEND"
  if configure_bridge_backend "$requested_backend"; then
    return 0
  fi

  if [[ "$requested_backend" == "mt5linux" ]]; then
    if [[ "$BRIDGE_BACKEND_EXPLICIT" == "1" ]]; then
      warn "mt5linux backend was explicitly requested but failed; attempting fallback to mt5 library"
    else
      warn "${requested_backend} backend failed; attempting fallback to mt5 library"
    fi
    stop_mt5linux_server
    if configure_bridge_backend "mt5"; then
      return 0
    fi
    die "Fallback to mt5 backend also failed"
  fi

  if [[ "$requested_backend" == "mt5" ]]; then
    die "mt5 backend setup failed"
  fi

  die "Bridge setup failed for backend ${requested_backend}"
}

main "$@"
