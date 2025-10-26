# mt5_toolkit/__main__.py
"""
CLI for mt5_toolkit that talks to MetaTrader5 directly (no Win RPC bridge).

Usage:
  python -m mt5_toolkit --selftest      # initialize + login and print terminal info
  python -m mt5_toolkit --ping          # alias for --selftest
"""

from __future__ import annotations
import argparse
import os
import sys


def _require_env(name: str) -> str:
    v = os.environ.get(name)
    if not v:
        raise SystemExit(f"Missing required environment variable: {name}")
    return v


def _mt5_login_via_wine() -> int:
    # Import here so the module can be imported without MT5 installed.
    import MetaTrader5 as mt5  # pip install MetaTrader5

    # Initialize terminal (under Wine this attaches to the running terminal)
    if not mt5.initialize():
        err = mt5.last_error()
        raise SystemExit(f"initialize() failed: {err}")

    # Credentials
    login = int(_require_env("MT5_LOGIN"))
    password = _require_env("MT5_PASSWORD")
    server = _require_env("MT5_SERVER")

    # Login
    if not mt5.login(login, password=password, server=server):
        err = mt5.last_error()
        raise SystemExit(f"login() failed: {err}")

    # Print concise terminal info
    info = mt5.terminal_info()
    print(
        f"OK: build={getattr(info,'build',None)} "
        f"connected={getattr(info,'connected',None)} "
        f"path={getattr(info,'path',None)}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="mt5_toolkit", add_help=True)
    p.add_argument("--selftest", action="store_true", help="init + login using MetaTrader5")
    p.add_argument("--ping", action="store_true", help="alias for --selftest")
    args = p.parse_args(argv)

    if args.selftest or args.ping:
        return _mt5_login_via_wine()

    p.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
