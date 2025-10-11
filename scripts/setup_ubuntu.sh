#!/usr/bin/env bash
# setup_ubuntu.sh — bring-up script for MT5 Wine bridge + env checks
# Safe to re-run. Assumes you run it from the repo root (~/MT5).
set -euo pipefail

# --- Helpers ---------------------------------------------------------------
here="$(cd "$(dirname "$0")" && pwd)"
cd "$here"

say() { printf "\n\033[1;32m==> %s\033[0m\n" "$*"; }
warn() { printf "\n\033[1;33mWARN:\033[0m %s\n" "$*" >&2; }
die() { printf "\n\033[1;31mERROR:\033[0m %s\n" "$*" >&2; exit 1; }

# --- Packages --------------------------------------------------------------
say "Installing required OS packages (ripgrep, Wine runtime)"
sudo apt-get update -y
sudo apt-get install -y --no-install-recommends \
  ripgrep wine winbind cabextract unzip p7zip-full

# --- Canonical paths & constants ------------------------------------------
WINEPREFIX="${WINEPREFIX:-$HOME/.wine-mt5}"
WIN_PYTHON_DEFAULT="$WINEPREFIX/drive_c/Python311/python.exe"
MT5_EXE_LINUX="$WINEPREFIX/drive_c/Program Files/MetaTrader 5/terminal64.exe"
MT5_EXE_WIN_ESC='C:\\Program Files\\MetaTrader 5\\terminal64.exe'  # double-backslash for .env checker

mkdir -p utils

# --- Ensure Wine prefix exists --------------------------------------------
say "Ensuring Wine prefix at $WINEPREFIX"
export WINEPREFIX
winecfg >/dev/null 2>&1 || true

# --- Ensure Windows Python in the SAME prefix as MT5 ----------------------
say "Ensuring Windows Python within $WINEPREFIX"
if [[ ! -x "$WIN_PYTHON_DEFAULT" ]]; then
  # If you already have a Python 3.11 in a sibling prefix, copy it
  if [[ -x "$HOME/.wine-py311/drive_c/Python311/python.exe" ]]; then
    mkdir -p "$WINEPREFIX/drive_c"
    cp -a "$HOME/.wine-py311/drive_c/Python311" "$WINEPREFIX/drive_c/"
  else
    warn "No Windows Python found at $WIN_PYTHON_DEFAULT and no ~/.wine-py311 to copy from."
    warn "Install Python 3.11 for Windows into this prefix or copy an existing one:"
    warn "  mkdir -p \"$WINEPREFIX/drive_c\" && cp -a ~/.wine-py311/drive_c/Python311 \"$WINEPREFIX/drive_c/\""
  fi
fi

[[ -x "$WIN_PYTHON_DEFAULT" ]] || warn "WIN_PYTHON not found yet at $WIN_PYTHON_DEFAULT (script will still proceed)."

# --- Verify MT5 terminal exists -------------------------------------------
if [[ ! -f "$MT5_EXE_LINUX" ]]; then
  warn "MetaTrader 5 terminal not found at:"
  warn "  $MT5_EXE_LINUX"
  warn "Install MT5 into this prefix so the terminal lives there, e.g."
  warn "  WINEPREFIX=\"$WINEPREFIX\" wine mt5setup.exe"
fi

# --- Write Windows-side JSON-RPC server -----------------------------------
say "Writing utils/mt5_win_server.py"
cat > utils/mt5_win_server.py <<'PY'
import json, socket, sys
from threading import Thread

try:
    import MetaTrader5 as mt5
except Exception as e:
    print("MetaTrader5 import failed:", e, file=sys.stderr)
    sys.exit(1)

HOST = "127.0.0.1"
PORT = 8765

_last_err = (0, "OK")

def _r():
    global _last_err
    try:
        _last_err = mt5.last_error()
    except Exception:
        pass
    return _last_err

def _j(obj):
    return (json.dumps(obj) + "\n").encode("utf-8")

def handle_cmd(cmd: dict) -> dict:
    name = cmd.get("cmd")

    if name == "ping":
        return {"ok": True, "pong": True}

    if name == "initialize":
        timeout = cmd.get("timeout")
        path = cmd.get("path")
        try:
            if path:
                ok = mt5.initialize(path, timeout=timeout) if timeout is not None else mt5.initialize(path)
            else:
                ok = mt5.initialize(timeout=timeout) if timeout is not None else mt5.initialize()
            return {"ok": bool(ok), "err": _r()}
        except Exception as e:
            return {"ok": False, "err": (500, f"initialize exception: {e}")}

    if name == "login":
        try:
            login = int(cmd["login"])
            password = cmd["password"]
            server = cmd.get("server")
            ok = mt5.login(login, password=password, server=server) if server else mt5.login(login, password=password)
            return {"ok": bool(ok), "err": _r()}
        except Exception as e:
            return {"ok": False, "err": (500, f"login exception: {e}")}

    if name == "account_info":
        try:
            ai = mt5.account_info()
            err = _r()
            if ai is None:
                return {"ok": False, "err": err, "info": None}
            info = {
                "login": getattr(ai, "login", 0),
                "trade_mode": getattr(ai, "trade_mode", None),
                "leverage": getattr(ai, "leverage", None),
                "balance": getattr(ai, "balance", None),
                "equity": getattr(ai, "equity", None),
                "name": getattr(ai, "name", None),
                "server": getattr(ai, "server", None),
                "currency": getattr(ai, "currency", None),
                "company": getattr(ai, "company", None),
            }
            return {"ok": True, "err": err, "info": info}
        except Exception as e:
            return {"ok": False, "err": (500, f"account_info exception: {e}"), "info": None}

    if name == "version":
        try:
            ver = mt5.version()
            return {"ok": True, "err": _r(), "ver": ver}
        except Exception as e:
            return {"ok": False, "err": (500, f"version exception: {e}"), "ver": None}

    if name == "terminal_info":
        try:
            ti = mt5.terminal_info()
            err = _r()
            if ti is None:
                return {"ok": False, "err": err, "info": None}
            info = {
                "community_account": getattr(ti, "community_account", None),
                "community_connection": getattr(ti, "community_connection", None),
                "connected": getattr(ti, "connected", True),
                "dlls_allowed": getattr(ti, "dlls_allowed", None),
                "ping_last": getattr(ti, "ping_last", None),
                "trade_allowed": getattr(ti, "trade_allowed", None),
                "trade_mode": getattr(ti, "trade_mode", None),
            }
            return {"ok": True, "err": err, "info": info}
        except Exception as e:
            return {"ok": False, "err": (500, f"terminal_info exception: {e}"), "info": None}

    if name == "last_error":
        return {"ok": True, "err": _r()}

    if name == "shutdown":
        try:
            mt5.shutdown()
            return {"ok": True, "err": _r()}
        except Exception as e:
            return {"ok": False, "err": (500, f"shutdown exception: {e}")}

    return {"ok": False, "err": (400, f"unknown cmd: {name}")}

def serve_client(conn):
    with conn:
        buf = b""
        while True:
            data = conn.recv(4096)
            if not data:
                break
            buf += data
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                if not line:
                    continue
                try:
                    cmd = json.loads(line.decode("utf-8"))
                except Exception as e:
                    conn.sendall(_j({"ok": False, "err": (400, f"bad json: {e}")}))
                    continue
                resp = handle_cmd(cmd)
                conn.sendall(_j(resp))

def main():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((HOST, PORT))
    s.listen(5)
    while True:
        c, _ = s.accept()
        Thread(target=serve_client, args=(c,), daemon=True).start()

if __name__ == "__main__":
    main()
PY

# --- Write Linux-side client/proxy ----------------------------------------
say "Writing utils/mt5_win_client.py"
cat > utils/mt5_win_client.py <<'PY'
import json, os, socket

HOST = "127.0.0.1"
PORT = int(os.environ.get("MT5_WIN_SERVER_PORT", "8765"))

def _rpc(cmd: dict) -> dict:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    s.connect((HOST, PORT))
    s.sendall((json.dumps(cmd) + "\n").encode("utf-8"))
    data = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
        if b"\n" in data:
            break
    s.close()
    line = data.split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))

class MetaTrader5Proxy:
    def __init__(self):
        self._last_error = (0, "OK")

    def initialize(self, *args, **kwargs):
        path = None
        timeout = kwargs.get("timeout")
        if len(args) == 1 and isinstance(args[0], (int, float)):
            timeout = args[0]
        elif len(args) >= 1 and isinstance(args[0], str):
            path = args[0]
            if len(args) >= 2:
                timeout = args[1]
        resp = _rpc({"cmd": "initialize", "path": path, "timeout": timeout})
        self._last_error = tuple(resp.get("err") or (0, "OK"))
        return bool(resp.get("ok"))

    def login(self, login: int, password: str, server: str | None = None):
        resp = _rpc({"cmd": "login", "login": int(login), "password": password, "server": server})
        self._last_error = tuple(resp.get("err") or (0, "OK"))
        return bool(resp.get("ok"))

    def account_info(self):
        resp = _rpc({"cmd": "account_info"})
        self._last_error = tuple(resp.get("err") or (0, "OK"))
        if not resp.get("ok"):
            return None
        class _AI:
            def __init__(self, d): self.__dict__.update(d or {})
            def __repr__(self): return f"AccountInfo(login={self.login}, server={self.server}, balance={self.balance})"
        return _AI(resp.get("info") or {})

    def last_error(self):
        return self._last_error

    def version(self):
        resp = _rpc({"cmd": "version"})
        self._last_error = tuple(resp.get("err") or (0, "OK"))
        return resp.get("ver") or (0, 0, "")

    def terminal_info(self):
        resp = _rpc({"cmd": "terminal_info"})
        self._last_error = tuple(resp.get("err") or (0, "OK"))
        if not resp.get("ok"):
            return None
        class _TI:
            def __init__(self, d): self.__dict__.update(d or {})
            def __repr__(self): return f"TerminalInfo(connected={self.connected})"
        return _TI(resp.get("info") or {})

    def shutdown(self):
        resp = _rpc({"cmd": "shutdown"})
        self._last_error = tuple(resp.get("err") or (0, "OK"))
        return bool(resp.get("ok"))
PY

# --- Point the bridge to our proxy loader ---------------------------------
say "Patching utils/mt5_bridge.py to prefer the Windows-side proxy"
if [[ -f utils/mt5_bridge.py && ! -f utils/mt5_bridge.py.bak ]]; then
  cp -a utils/mt5_bridge.py utils/mt5_bridge.py.bak
fi
python - <<'PY' || true
import io, os, re, sys
p="utils/mt5_bridge.py"
src=open(p,"r",encoding="utf-8").read() if os.path.exists(p) else ""
block = """

# --- appended: prefer persistent Windows-side server proxy ---
def get_configured_backend():
    import os as _os
    return _os.environ.get("MT5_BRIDGE_BACKEND", "native").strip().lower()

def load_mt5_module():
    backend = get_configured_backend()
    if backend == "mt5linux":
        import importlib as _importlib
        try:
            return _importlib.import_module("mt5linux")
        except Exception as e:
            raise RuntimeError(f"mt5linux import failed: {e}") from e
    try:
        from utils.mt5_win_client import MetaTrader5Proxy
        return MetaTrader5Proxy()
    except Exception as e_proxy:
        raise RuntimeError(f"mt5 windows proxy init failed: {e_proxy}") from e_proxy
"""
if "mt5_win_client" not in src:
    with open(p,"a",encoding="utf-8") as f: f.write(block)
print("mt5_bridge patched")
PY

# --- Optional: thin direct broker shim used by some code paths ------------
say "Writing brokers/mt5_direct.py"
mkdir -p brokers
cat > brokers/mt5_direct.py <<'PY'
from utils.mt5_bridge import load_mt5_module

def initialize(timeout=90000, path=None):
    mt5 = load_mt5_module()
    return mt5.initialize(timeout=timeout) if path is None else mt5.initialize(path, timeout=timeout)

def is_terminal_logged_in():
    mt5 = load_mt5_module()
    info = mt5.account_info()
    return bool(info and getattr(info, "login", 0))
PY

# --- Normalize local package name (avoid shadowing vendor MetaTrader5) ----
# If repo previously used a local "mt5" package, rename it to mt5_app and update imports
if [[ -d mt5 && ! -d mt5_app ]]; then
  say "Renaming local package mt5 -> mt5_app to avoid import shadowing"
  git mv mt5 mt5_app 2>/dev/null || mv mt5 mt5_app
  say "Rewriting imports from 'mt5' to 'mt5_app' (excluding .venv and .git)"
  rg -l -e '\bfrom\s+mt5(\b|\.)' -e '\bimport\s+mt5(\b|\.)' --hidden --glob '!.venv/*' --glob '!.git/*' . \
    | xargs -r sed -i -E 's/\bfrom\s+mt5\b/from mt5_app/g; s/\bimport\s+mt5\b/import mt5_app/g; s/\bimport\s+mt5\./import mt5_app./g'
fi

# --- Write a clean, checker-friendly .env ---------------------------------
say "Writing canonical .env entries (idempotent)"
touch .env
# Remove any prior duplicates of these keys
sed -i '/^WINEPREFIX=/d; /^WIN_PYTHON=/d; /^MT5_TERMINAL_PATH=/d; /^MT5_EXE_WIN=/d; /^MT5_LOGIN=/d; /^MT5_PASSWORD=/d; /^MT5_SERVER=/d' .env
cat >> .env <<ENV
# --- MT5 bridge + terminal (canonical values) ---
WINEPREFIX=$WINEPREFIX
WIN_PYTHON=$WIN_PYTHON_DEFAULT
MT5_TERMINAL_PATH="$MT5_EXE_LINUX"
MT5_EXE_WIN=$MT5_EXE_WIN_ESC

# --- MT5 credentials (demo/sample) ---
MT5_LOGIN=${MT5_LOGIN:-1107832}
MT5_PASSWORD=${MT5_PASSWORD:-ox@66EPeremogy}
MT5_SERVER=${MT5_SERVER:-OxSecurities-Demo}
ENV

# --- Export to current shell so checker matches the file -------------------
say "Exporting .env into current shell"
set -a; source .env; set +a

# --- Launch terminal + Windows-side server --------------------------------
say "Launching MT5 terminal (detached) in prefix"
WINEDEBUG=-all WINEPREFIX="$WINEPREFIX" wine start 'C:\Program Files\MetaTrader 5\terminal64.exe' /portable || true
sleep 3

say "Starting Windows-side mt5_win_server"
if command -v winepath >/dev/null 2>&1; then
  WIN_SERVER_PATH="$(winepath -w "$PWD/utils/mt5_win_server.py")"
else
  WIN_SERVER_PATH="Z:${PWD//\//\\}\\utils\\mt5_win_server.py"
fi
# Kill any prior servers
wineserver -k || true
# Start server in background
WINEDEBUG=-all WINEPREFIX="$WINEPREFIX" wine "$WIN_PYTHON_DEFAULT" "$WIN_SERVER_PATH" >/dev/null 2>&1 &
sleep 2

# --- Quick connectivity sanity --------------------------------------------
say "Sanity: ping server & initialize/login"
python - <<'PY'
import json, socket, os, time
from utils.mt5_bridge import load_mt5_module
# ping
s=socket.socket(); s.settimeout(2); s.connect(("127.0.0.1", 8765))
s.sendall((json.dumps({"cmd":"ping"})+"\n").encode()); print("ping ->", s.recv(4096).decode().strip()); s.close()
# initialize + optional login
m = load_mt5_module()
print("initialize:", m.initialize(timeout=15000), "err:", m.last_error())
ai = m.account_info()
print("acct before:", ai)
login = os.environ.get("MT5_LOGIN"); pwd=os.environ.get("MT5_PASSWORD"); srv=os.environ.get("MT5_SERVER")
if login and pwd:
    ok = m.login(int(login), pwd, srv)
    print("login:", ok, "err:", m.last_error())
    print("acct after:", m.account_info())
else:
    print("login skipped (no creds in env)")
m.shutdown()
PY

# --- Run the project checker ----------------------------------------------
say "Running utils.environment checker"
python -m utils.environment || true

say "Done. If any checker item failed, re-run this script or check the warnings above."
