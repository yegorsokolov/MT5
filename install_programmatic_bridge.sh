#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_python_version_config.sh
source "${SCRIPT_DIR}/scripts/_python_version_config.sh"

log() { printf '[bridge] %s\n' "$*" >&2; }
error() { printf '[bridge:ERROR] %s\n' "$*" >&2; }
die() { error "$*"; exit 1; }

PY_WINE_PREFIX_DEFAULT="${HOME}/${MT5_PYTHON_PREFIX_NAME}"
MT5_WINE_PREFIX_DEFAULT="${HOME}/.mt5"
DEFAULT_TIMEOUT_MS="90000"
DEFAULT_PIP_TIMEOUT="180"

PY_WINE_PREFIX="${PY_WINE_PREFIX:-${PY_WINE_PREFIX_DEFAULT}}"
MT5_WINE_PREFIX="${MT5_WINE_PREFIX:-${MT5_WINE_PREFIX_DEFAULT}}"
WIN_PYTHON="${WIN_PYTHON:-}"
MT5_TERMINAL="${MT5_TERMINAL:-}"
BRIDGE_TIMEOUT_MS="${MT5_BRIDGE_TIMEOUT_MS:-${DEFAULT_TIMEOUT_MS}}"
PIP_TIMEOUT="${PIP_TIMEOUT:-${DEFAULT_PIP_TIMEOUT}}"

usage() {
  cat <<'USAGE'
Usage: install_programmatic_bridge.sh [options]

Options:
  --py-wine-prefix PATH     Wine prefix containing Windows Python (default: $PY_WINE_PREFIX)
  --mt5-wine-prefix PATH    Wine prefix containing MetaTrader terminal (default: $MT5_WINE_PREFIX)
  --win-python PATH         Override path to Windows python.exe inside the Wine prefix
  --terminal PATH           Override path to terminal64.exe inside the MT5 prefix
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

to_windows_path() {
  local prefix="$1" path="$2"
  WINEPREFIX="$prefix" winepath -w "$path" 2>/dev/null | tr -d '\r'
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
  if ! WINEPREFIX="$PY_WINE_PREFIX" PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" wine "$WIN_PYTHON_WINPATH" -m pip install --upgrade pip 1>&2; then
    die "Failed to upgrade pip in Windows environment"
  fi

  local packages=(MetaTrader5 pymt5linux)
  log "Installing required packages: ${packages[*]}"
  if ! WINEPREFIX="$PY_WINE_PREFIX" PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" wine "$WIN_PYTHON_WINPATH" -m pip install --upgrade --only-binary :all: "${packages[@]}" 1>&2; then
    log "Binary wheel installation failed; retrying without --only-binary"
    WINEPREFIX="$PY_WINE_PREFIX" PIP_DEFAULT_TIMEOUT="$PIP_TIMEOUT" wine "$WIN_PYTHON_WINPATH" -m pip install --upgrade "${packages[@]}" 1>&2 || die "Failed to install Windows MetaTrader dependencies"
  fi
}

write_probe_script() {
  local path="$1"
  cat <<'PY' >"$path"
import json
import os
import sys
import time

import MetaTrader5 as mt5

TIMEOUT = int(os.environ.get("MT5_BRIDGE_TIMEOUT_MS", "90000"))
TERMINAL_PATH = os.environ.get("MT5_TERMINAL_PATH")

if not TERMINAL_PATH:
    print(json.dumps({"status": "fail", "error": "missing terminal path"}))
    sys.exit(1)

if not mt5.initialize(path=TERMINAL_PATH, timeout=TIMEOUT):
    code, message = mt5.last_error()
    print(json.dumps({"status": "fail", "error": [code, message]}))
    sys.exit(1)

time.sleep(1)
mt5.shutdown()
print(json.dumps({"status": "ok"}))
PY
}

run_probe() {
  local probe_host="$1"
  local probe_win="$2"

  log "Running MetaTrader bridge probe via Windows Python..."
  local output
  if ! output=$(WINEPREFIX="$PY_WINE_PREFIX" MT5_TERMINAL_PATH="$MT5_TERMINAL_WINPATH" MT5_BRIDGE_TIMEOUT_MS="$BRIDGE_TIMEOUT_MS" wine "$WIN_PYTHON_WINPATH" "$probe_win" 2>&1); then
    printf '%s\n' "$output" >&2
    die "Bridge probe execution failed"
  fi

  output="$(printf '%s' "$output" | tr -d '\r')"
  log "Probe output: $output"
  if ! python3 - "$output" <<'PY'; then
import json
import sys
try:
    data = json.loads(sys.argv[1])
except Exception as exc:  # noqa: BLE001
    raise SystemExit("invalid JSON from probe: %s" % exc)
if data.get("status") != "ok":
    raise SystemExit("probe reported failure: %s" % data)
PY
    die "Bridge probe validation failed"
  fi
}

main() {
  parse_args "$@"

  require_command wine
  require_command winepath
  require_command python3

  [[ -d "$PY_WINE_PREFIX" ]] || die "Python Wine prefix not found: $PY_WINE_PREFIX"
  [[ -d "$MT5_WINE_PREFIX" ]] || die "MetaTrader Wine prefix not found: $MT5_WINE_PREFIX"

  WIN_PYTHON="$(detect_windows_python "$PY_WINE_PREFIX" "$WIN_PYTHON")"
  MT5_TERMINAL="$(detect_terminal "$MT5_WINE_PREFIX" "$MT5_TERMINAL")"

  WIN_PYTHON_WINPATH="$(to_windows_path "$PY_WINE_PREFIX" "$WIN_PYTHON")"
  [[ -n "$WIN_PYTHON_WINPATH" ]] || die "Unable to translate Windows Python path"

  MT5_TERMINAL_WINPATH="$(to_windows_path "$PY_WINE_PREFIX" "$MT5_TERMINAL")"
  [[ -n "$MT5_TERMINAL_WINPATH" ]] || die "Unable to translate MetaTrader terminal path"

  log "Discovered Windows Python: $WIN_PYTHON (winpath: $WIN_PYTHON_WINPATH)"
  log "Discovered MetaTrader terminal: $MT5_TERMINAL (winpath: $MT5_TERMINAL_WINPATH)"

  install_windows_packages

  local probe_host="$PY_WINE_PREFIX/drive_c/mt5_bridge_probe.py"
  write_probe_script "$probe_host"
  trap 'rm -f "$probe_host"' EXIT
  local probe_win
  probe_win="$(to_windows_path "$PY_WINE_PREFIX" "$probe_host")"
  [[ -n "$probe_win" ]] || die "Unable to translate probe script path"

  run_probe "$probe_host" "$probe_win"

  rm -f "$probe_host" || true

  printf 'WIN_PYTHON_WINPATH=%s\n' "$WIN_PYTHON_WINPATH"
  printf 'MT5_TERMINAL_WINPATH=%s\n' "$MT5_TERMINAL_WINPATH"
}

main "$@"
