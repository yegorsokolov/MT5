#!/usr/bin/env bash
set -euo pipefail

# Defaults match the layout produced by the programmatic bridge step.
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/_python_version_config.sh
source "${SCRIPT_ROOT}/scripts/_python_version_config.sh"

PY_WINE_PREFIX="${PY_WINE_PREFIX:-$MT5_PYTHON_PREFIX}"
MT5_WINE_PREFIX="${MT5_WINE_PREFIX:-$HOME/.mt5}"
WIN_PYTHON="${WIN_PYTHON:-$PY_WINE_PREFIX/drive_c/Python${MT5_PYTHON_TAG}/python.exe}"
MT5_TERMINAL="${MT5_TERMINAL:-$MT5_WINE_PREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe}"
ENV_FILE="${ENV_FILE:-.env}"
WRITER_SCRIPT="${WRITER_SCRIPT:-./write_mt5_env.sh}"

error() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

command -v winepath >/dev/null 2>&1 || error "winepath not found; install Wine or adjust PATH."
[[ -x "$WRITER_SCRIPT" ]] || error "Cannot execute ${WRITER_SCRIPT}; confirm the script exists and is chmod +x."
[[ -f "$WIN_PYTHON" ]] || error "Windows python executable not found at ${WIN_PYTHON}."
[[ -f "$MT5_TERMINAL" ]] || error "MetaTrader terminal not found at ${MT5_TERMINAL}."

win_path() {
  local prefix="$1" path="$2"
  WINEPREFIX="$prefix" winepath -w "$path" 2>/dev/null | tr -d '\r'
}

WIN_PY_WINPATH="$(win_path "$PY_WINE_PREFIX" "$WIN_PYTHON")"
MT5_WINPATH="$(win_path "$PY_WINE_PREFIX" "$MT5_TERMINAL")"

[[ -n "$WIN_PY_WINPATH" ]] || error "Failed to convert ${WIN_PYTHON} to a Windows path."
[[ -n "$MT5_WINPATH" ]] || error "Failed to convert ${MT5_TERMINAL} to a Windows path."

export PYMT5LINUX_PYTHON="$WIN_PY_WINPATH"
export WINE_PYTHON="$WIN_PY_WINPATH"
export PYMT5LINUX_WINEPREFIX="$MT5_WINE_PREFIX"
export WINEPREFIX="$MT5_WINE_PREFIX"
export MT5_TERMINAL_PATH="$MT5_WINPATH"

"$WRITER_SCRIPT" \
  --env-file "$ENV_FILE" \
  --win-python "$WIN_PY_WINPATH" \
  --wine-prefix "$MT5_WINE_PREFIX" \
  --terminal "$MT5_WINPATH"

