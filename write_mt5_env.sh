#!/usr/bin/env bash
set -euo pipefail

# write_mt5_env.sh
# Upserts bridge-related keys into .env (defaults taken from environment variables).

usage() {
  cat <<'EOF'
Usage: ./write_mt5_env.sh [--env-file PATH] [--win-python PATH] [--wine-prefix PATH] [--terminal PATH]

  --env-file     Target .env file (default: .env in the current directory)
  --win-python   Windows-style python.exe path (e.g. Z:\opt\mt5\Python313\python.exe)
  --wine-prefix  Wine prefix directory hosting MT5 (e.g. /home/user/.mt5)
  --terminal     Windows-style path to terminal64.exe (optional; adds MT5_TERMINAL_PATH)

Values default to the environment variables
PYMT5LINUX_PYTHON / WINE_PYTHON, PYMT5LINUX_WINEPREFIX / WINEPREFIX,
and MT5_TERMINAL_PATH / MT5_WINPATH.
EOF
}

ENV_FILE=".env"
WIN_PYTHON=""
WINE_PREFIX=""
TERMINAL_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-file) ENV_FILE="$2"; shift 2 ;;
    --win-python) WIN_PYTHON="$2"; shift 2 ;;
    --wine-prefix) WINE_PREFIX="$2"; shift 2 ;;
    --terminal) TERMINAL_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage; exit 1 ;;
  esac
done

WIN_PYTHON="${WIN_PYTHON:-${PYMT5LINUX_PYTHON:-${WINE_PYTHON:-}}}"
WINE_PREFIX="${WINE_PREFIX:-${PYMT5LINUX_WINEPREFIX:-${WINEPREFIX:-}}}"
TERMINAL_PATH="${TERMINAL_PATH:-${MT5_TERMINAL_PATH:-${MT5_WINPATH:-}}}"

if [[ -z "$WIN_PYTHON" ]]; then
  echo "ERROR: Windows python path is required." >&2
  exit 1
fi

if [[ -z "$WINE_PREFIX" ]]; then
  echo "ERROR: Wine prefix is required." >&2
  exit 1
fi

declare -A kv=(
  [PYMT5LINUX_PYTHON]="$WIN_PYTHON"
  [PYMT5LINUX_WINDOWS_PYTHON]="$WIN_PYTHON"
  [WINE_PYTHON]="$WIN_PYTHON"
  [PYMT5LINUX_WINEPREFIX]="$WINE_PREFIX"
  [WIN_PY_WINE_PREFIX]="$WINE_PREFIX"
  [WINEPREFIX]="$WINE_PREFIX"
)

if [[ -n "$TERMINAL_PATH" ]]; then
  kv[MT5_TERMINAL_PATH]="$TERMINAL_PATH"
fi

escape() {
  local value="$1"
  value="${value//$'\r'/\\r}"
  value="${value//$'\n'/\\n}"
  printf '%s' "$value"
}

touch "$ENV_FILE"
tmp="$(mktemp)"
trap 'rm -f "$tmp"' EXIT

cp "$ENV_FILE" "$tmp"
for key in "${!kv[@]}"; do
  escaped_key="$(printf '%s' "$key" | sed 's/[.[\*^$]/\\&/g')"
  grep -v -E "^${escaped_key}=" "$tmp" > "${tmp}.new" || true
  mv "${tmp}.new" "$tmp"
  printf '%s=%s\n' "$key" "$(escape "${kv[$key]}")" >> "$tmp"
done

mv "$tmp" "$ENV_FILE"
trap - EXIT
echo "Updated ${ENV_FILE} with MT5 bridge settings:"
for key in "${!kv[@]}"; do
  printf '  %s\n' "$key"
done
