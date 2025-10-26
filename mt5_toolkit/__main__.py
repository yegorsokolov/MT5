# mt5_toolkit/__main__.py
"""
CLI for mt5_toolkit that talks to MetaTrader5.
- On native Windows (or if MetaTrader5 is importable), import and use locally.
- Otherwise, fall back to running the MetaTrader5 calls inside Wine's Windows Python.

Usage:
  python -m mt5_toolkit --selftest
  python -m mt5_toolkit --ping     (alias for --selftest)
"""

from __future__ import annotations
import argparse
import os
import shutil
import subprocess
import sys
from typing import Optional


REQUIRED_ENV = ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER")


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v


def _mt5_login_local() -> int:
    """Attempt to init+login using the *current* Python interpreter (works on Windows)."""
    try:
        import MetaTrader5 as mt5  # type: ignore
    except Exception as e:  # not available on Linux wheels
        raise RuntimeError(f"Local MetaTrader5 import failed: {e}") from e

    login = int(_require_env("MT5_LOGIN"))
    password = _require_env("MT5_PASSWORD")
    server = _require_env("MT5_SERVER")

    if not mt5.initialize():
        raise SystemExit(f"initialize() failed: {mt5.last_error()}")
    if not mt5.login(login, password=password, server=server):
        raise SystemExit(f"login() failed: {mt5.last_error()}")

    info = mt5.terminal_info()
    print(
        f"OK: build={getattr(info,'build',None)} "
        f"connected={getattr(info,'connected',None)} "
        f"path={getattr(info,'path',None)}"
    )
    return 0


def _mt5_login_via_wine() -> int:
    """Run a small script inside Wine's Windows Python to init+login and print info."""
    win_python = os.environ.get("WIN_PYTHON")
    if not win_python:
        raise SystemExit("WIN_PYTHON is not set; cannot use Wine fallback.")
    if not shutil.which("wine"):
        raise SystemExit("'wine' not found on PATH; cannot use Wine fallback.")

    # Ensure required env exists
    login = _require_env("MT5_LOGIN")
    password = _require_env("MT5_PASSWORD")
    server = _require_env("MT5_SERVER")

    code = rf"""
import MetaTrader5 as mt5
login = int({login!r})
password = {password!r}
server = {server!r}
if not mt5.initialize():
    raise SystemExit(f"initialize() failed: {{mt5.last_error()}}")
if not mt5.login(login, password=password, server=server):
    raise SystemExit(f"login() failed: {{mt5.last_error()}}")
info = mt5.terminal_info()
print("OK: build=%s connected=%s path=%s" % (
    getattr(info,"build",None),
    getattr(info,"connected",None),
    getattr(info,"path",None),
))
"""

    # Run Wine Windows Python
    proc = subprocess.run(
        ["wine", win_python, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=os.environ.copy(),
        text=True,
    )
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr)
        raise SystemExit(proc.returncode)
    sys.stdout.write(proc.stdout)
    return 0


def _selftest() -> int:
    # Try local import first (works on native Windows), else Wine fallback.
    try:
        return _mt5_login_local()
    except Exception:
        return _mt5_login_via_wine()


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(prog="mt5_toolkit")
    p.add_argument("--selftest", action="store_true", help="init+login to MT5 and print terminal info")
    p.add_argument("--ping", action="store_true", help="alias for --selftest")
    args = p.parse_args(argv)

    if args.selftest or args.ping:
        # Ensure required env is present before we try either path
        for k in REQUIRED_ENV:
            _require_env(k)
        return _selftest()

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
