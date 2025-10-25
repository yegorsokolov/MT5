#!/usr/bin/env bash
set -euo pipefail

# ------------------------------------------------------------
# MT5 on Ubuntu (Wine) full setup — idempotent & chatty
# ------------------------------------------------------------

# --- Config you probably want to keep as defaults
WINEPREFIX_DEFAULT="${HOME}/.wine-py311"
WIN_PYTHON_DEFAULT="C:\\Python311\\python.exe"    # we'll auto-fix to /home/... if .env already has it
MT5_EXE_WIN_DEFAULT="C:\\Program Files\\MetaTrader 5\\terminal64.exe"
MT5_WIN_SERVER_PORT_DEFAULT="8765"

# --- Repo root detection
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Helpers
say() { printf "\033[1;36m[setup]\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m[warn]\033[0m %s\n" "$*"; }
die() { printf "\033[1;31m[fail]\033[0m %s\n" "$*"; exit 1; }

require_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

# --- Preconditions
require_cmd python
require_cmd pip
require_cmd wine
require_cmd ss

# If a venv is used, keep using it:
PY="$(command -v python)"

say "Python: $("$PY" -V)  at $(command -v python)"
say "Repo:   $REPO_ROOT"

# --- Ensure .env exists and normalize key exports
ENV_FILE="$REPO_ROOT/.env"
touch "$ENV_FILE"

# Load any existing values first (so we can keep your login/server)
set -a; source "$ENV_FILE" || true; set +a

# Required creds (user provided earlier). If not set, abort with instructions.
: "${MT5_LOGIN:?MT5_LOGIN not set in .env. Example: MT5_LOGIN=1107832}"
: "${MT5_PASSWORD:?MT5_PASSWORD not set in .env. Example: MT5_PASSWORD=your_password}"
: "${MT5_SERVER:?MT5_SERVER not set in .env. Example: MT5_SERVER=YourBroker-ServerName}"

# Derive defaults (don’t override if already exported in the shell)
WINEPREFIX="${WINEPREFIX:-$WINEPREFIX_DEFAULT}"
MT5_EXE_WIN="${MT5_EXE_WIN:-$MT5_EXE_WIN_DEFAULT}"
MT5_WIN_SERVER_PORT="${MT5_WIN_SERVER_PORT:-$MT5_WIN_SERVER_PORT_DEFAULT}"

# We try to discover Wine’s Python path if not provided
if [[ -z "${WIN_PYTHON:-}" ]]; then
  # Prefer actual prefix install if present
  CAND_UNIX="${WINEPREFIX}/drive_c/Python311/python.exe"
  if [[ -f "$CAND_UNIX" ]]; then
    WIN_PYTHON="$CAND_UNIX"
  else
    # Fall back to Windows style (works fine when passed to `wine`)
    WIN_PYTHON="$WIN_PYTHON_DEFAULT"
  fi
fi

# Compute the Linux path to MT5 terminal (quoted because of spaces)
MT5_TERMINAL_PATH="${MT5_TERMINAL_PATH:-${WINEPREFIX}/drive_c/Program Files/MetaTrader 5/terminal64.exe}"

say "Writing canonical exports into .env (single block, quoted where needed)"
# Remove ALL prior occurrences to avoid parser weirdness
sed -i '/^\s*\(export\s\+\)\?\(MT5_PASSWORD\|MT5_SERVER\|MT5_BRIDGE_BACKEND\|MT5_EXE_WIN\|MT5_TERMINAL_PATH\|WIN_PYTHON\|WINEPREFIX\|MT5_WIN_SERVER_PORT\)=/d' "$ENV_FILE"

cat >> "$ENV_FILE" <<ENV
export MT5_BRIDGE_BACKEND=wine
export MT5_EXE_WIN="$(printf '%s' "$MT5_EXE_WIN")"
export MT5_TERMINAL_PATH="$(printf '%s' "$MT5_TERMINAL_PATH")"
export WIN_PYTHON=$(printf '%s' "$WIN_PYTHON")
export WINEPREFIX=$(printf '%s' "$WINEPREFIX")
export MT5_WIN_SERVER_PORT=$(printf '%s' "$MT5_WIN_SERVER_PORT")
# creds (kept as you had them)
export MT5_LOGIN=$(printf '%s' "$MT5_LOGIN")
export MT5_PASSWORD=$(printf '%s' "$MT5_PASSWORD")
export MT5_SERVER=$(printf '%s' "$MT5_SERVER")
ENV

# Reload into THIS shell session
set -a; source "$ENV_FILE"; set +a

say "Effective env:"
printf '  %-22s %s\n' \
  "WINEPREFIX" "$WINEPREFIX" \
  "WIN_PYTHON" "$WIN_PYTHON" \
  "MT5_EXE_WIN" "$MT5_EXE_WIN" \
  "MT5_TERMINAL_PATH" "$MT5_TERMINAL_PATH" \
  "MT5_WIN_SERVER_PORT" "$MT5_WIN_SERVER_PORT" \
  "MT5_LOGIN" "$MT5_LOGIN" \
  "MT5_SERVER" "$MT5_SERVER"

# --- Linux venv deps: ensure RPyC 5.3.1 (signature includes config kwarg)
say "Pinning RPyC to 5.3.1 in the active interpreter to avoid kwarg mismatch…"
"$PY" -m pip install --upgrade 'rpyc==5.3.1' >/dev/null

# --- Safety-net shim for processes that might still import an older RPyC
# (harmless if not needed)
SITE_CUSTOM="$REPO_ROOT/sitecustomize.py"
if ! grep -q "rpyc.connect" "$SITE_CUSTOM" 2>/dev/null; then
  say "Installing sitecustomize.py safety shim"
  cat > "$SITE_CUSTOM" <<'PY'
import inspect
try:
    import rpyc
    # If signature does NOT accept "config", wrap to ignore unknown kwarg
    if "config" not in inspect.signature(rpyc.connect).parameters:
        _orig = rpyc.connect
        def _wrap(host, port, *args, **kwargs):
            kwargs.pop("config", None)
            return _orig(host, port, *args, **kwargs)
        rpyc.connect = _wrap
except Exception:
    pass
PY
fi

# --- Headless X
if ! pgrep -x Xvfb >/dev/null 2>&1; then
  say "Starting Xvfb :1"
  Xvfb :1 -screen 0 1280x800x24 >/tmp/xvfb.log 2>&1 &
  sleep 1
fi
export DISPLAY=:1
export WINEDEBUG=-all WINEDLLOVERRIDES="winemenubuilder.exe=d"

# --- Clean out any stragglers
say "Stopping any old MT5 instances in prefix ${WINEPREFIX}"
pkill -f "C:\\\\Program Files\\\\MetaTrader 5\\\\terminal64.exe" 2>/dev/null || true
wineserver -k 2>/dev/null || true
sleep 1

# --- Launch terminal with /portable (so Wine-Python & terminal share data)
say "Launching terminal with /portable"
nohup wine "$MT5_TERMINAL_PATH" /portable >/tmp/mt5.out 2>&1 & disown
sleep 4
if ! pgrep -fa terminal64 >/dev/null; then
  warn "terminal64 not visible yet; recent build may launch via updater — checking logs"
  tail -n 120 /tmp/mt5.out || true
fi

# --- Ensure MetaTrader5 is available in Wine Python and do programmatic login
say "Ensuring MetaTrader5 package in Wine Python"
wine "$WIN_PYTHON" -m pip install --upgrade pip >/dev/null || true
wine "$WIN_PYTHON" -m pip install --upgrade MetaTrader5 >/dev/null

say "Programmatic initialize/login via Wine Python"
wine "$WIN_PYTHON" - <<'PY'
import os, time, MetaTrader5 as mt5

TIMEOUT_MS = 120_000
ok = mt5.initialize(timeout=TIMEOUT_MS)
print("initialize:", ok, mt5.last_error())
if not ok:
    raise SystemExit(1)

login  = int(os.environ["MT5_LOGIN"])
pwd    = os.environ["MT5_PASSWORD"]
server = os.environ["MT5_SERVER"]

if not mt5.login(login, pwd, server):
    print("login failed:", mt5.last_error()); raise SystemExit(2)

# Confirm attached
for _ in range(60):
    ti = mt5.terminal_info()
    ai = mt5.account_info()
    if getattr(ti, "connected", False) and getattr(ai, "login", 0):
        print("OK:", mt5.version(), getattr(ti,"connected",None), getattr(ai,"login",None))
        break
    time.sleep(1)
# Keep session open
PY

# --- Start the Wine-side RPC server (if you use the bridge)
if ! ss -lntp | grep -q ":${MT5_WIN_SERVER_PORT} "; then
  say "Starting mt5_win_server.py RPC on 127.0.0.1:${MT5_WIN_SERVER_PORT}"
  nohup wine "$WIN_PYTHON" "Z:\\$(printf '%s' "$REPO_ROOT" | sed 's|/|\\|g')\\utils\\mt5_win_server.py" \
    --host 127.0.0.1 --port "${MT5_WIN_SERVER_PORT}" --terminal "$MT5_EXE_WIN" \
    >/tmp/mt5-bridge.out 2>&1 & disown
  sleep 2
fi

say "RPC listen check:"
ss -lntp | grep ":${MT5_WIN_SERVER_PORT} " || warn "RPC not listening — bridge may not be required if you only use Wine Python"

# --- Final repo checker
say "Running repo environment checker"
"$PY" -m utils.environment || true

say "Done."
