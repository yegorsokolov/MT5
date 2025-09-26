"""Verify MetaTrader 5 connectivity from the Python toolchain."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


try:  # MetaTrader5 is an optional dependency during lint/unit-test runs
    import MetaTrader5 as _mt5  # type: ignore
except Exception:  # pragma: no cover - handled at runtime
    _mt5 = None  # type: ignore

try:  # Reuse the broker helper so we respect retry logic/teardown behaviour
    from brokers import mt5_direct
except Exception:  # pragma: no cover - minimal environments may not ship broker deps
    mt5_direct = None  # type: ignore


@dataclass
class ConnectionResult:
    """Outcome of the MetaTrader 5 connection attempt."""

    success: bool
    account_login: Optional[int] = None
    account_name: Optional[str] = None
    broker: Optional[str] = None
    leverage: Optional[float] = None
    balance: Optional[float] = None
    equity: Optional[float] = None
    error: Optional[str] = None


def _last_error() -> str:
    if _mt5 is None:
        return "MetaTrader5 package unavailable"
    last_error = getattr(_mt5, "last_error", None)
    if callable(last_error):
        try:
            code, message = last_error()
        except Exception:  # pragma: no cover - defensive best-effort
            return "unknown MetaTrader5 error"
        if code or message:
            if message:
                return f"[{code}] {message}" if code else message
            return f"[{code}]"
    return "Unknown MetaTrader5 error"


def _resolve_terminal_path(candidate: Path) -> Optional[Path]:
    """Best-effort resolution of the MT5 terminal executable."""

    candidate = candidate.expanduser().resolve()
    if candidate.is_file():
        return candidate
    if not candidate.exists():
        return None

    # Common executable names on Windows/Wine deployments.
    for name in ("terminal64.exe", "terminal.exe", "terminal"):
        guess = candidate / name
        if guess.exists():
            return guess

    # Fall back to a shallow search – only look a single level deep to avoid
    # traversing entire Wine prefixes.
    for child in candidate.iterdir():
        if child.is_file() and child.name.lower().startswith("terminal"):
            return child
    return None


def _default_mt5_dir() -> Path:
    """Best effort guess at the MetaTrader 5 installation directory."""

    env_override = os.getenv("MT5_TERMINAL_PATH")
    if env_override:
        return Path(env_override).expanduser()

    # Wine deployments default to ``~/.wine-mt5`` in our automation.  When the
    # script executes on Windows we fall back to the standard installation path
    # so ``terminal64.exe`` can be discovered without extra configuration.
    if sys.platform.startswith("win"):
        return Path("C:/Program Files/MetaTrader 5")

    return Path.home() / ".wine-mt5" / "drive_c" / "Program Files" / "MetaTrader 5"


def _copy_heartbeat_script(mt5_dir: Path) -> Optional[Path]:
    """Install the ConnectionHeartbeat script for manual diagnostics."""

    source = Path(__file__).resolve().parent.parent / "mt5" / "mql5" / "ConnectionHeartbeat.mq5"
    if not source.exists():
        return None

    target = mt5_dir / "MQL5" / "Scripts" / "MT5Bridge" / source.name
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    except PermissionError:
        print(
            "Unable to install heartbeat script because the MetaTrader 5 directory is not writable. "
            "Re-run the command with sufficient permissions or point --mt5-dir to a writable location.",
            file=sys.stderr,
        )
        return None
    return target


def attempt_connection(
    *,
    terminal: Optional[Path],
    login: Optional[int],
    password: Optional[str],
    server: Optional[str],
) -> ConnectionResult:
    """Attempt to initialise the MetaTrader5 bridge."""

    if _mt5 is None:
        return ConnectionResult(success=False, error="MetaTrader5 package not installed")
    if mt5_direct is None:
        return ConnectionResult(success=False, error="brokers.mt5_direct unavailable")

    kwargs = {}
    if terminal is not None:
        kwargs["path"] = str(terminal)
    if login is not None:
        kwargs["login"] = login
    if password is not None:
        kwargs["password"] = password
    if server is not None:
        kwargs["server"] = server

    if not mt5_direct.initialize(**kwargs):  # pragma: no cover - requires live terminal
        return ConnectionResult(success=False, error=_last_error())

    try:
        info = _mt5.account_info()
    except Exception as exc:  # pragma: no cover - depends on terminal state
        _mt5.shutdown()
        return ConnectionResult(success=False, error=f"account_info failed: {exc}")

    _mt5.shutdown()

    if info is None:
        return ConnectionResult(success=False, error="No account info returned")

    return ConnectionResult(
        success=True,
        account_login=getattr(info, "login", None),
        account_name=getattr(info, "name", None),
        broker=getattr(info, "company", None),
        leverage=getattr(info, "leverage", None),
        balance=getattr(info, "balance", None),
        equity=getattr(info, "equity", None),
    )


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate that the Python runtime can talk to a logged-in MetaTrader 5 terminal. "
            "Provide login credentials to authenticate headlessly or omit them to reuse the running terminal session."
        )
    )
    parser.add_argument(
        "mt5_dir",
        nargs="?",
        default=str(_default_mt5_dir()),
        help="Path to the MT5 installation directory or terminal executable (default: %(default)s)",
    )
    parser.add_argument("--login", type=int, help="Broker login to authenticate with MetaTrader 5")
    parser.add_argument("--password", help="Broker password for the MT5 account")
    parser.add_argument("--server", help="Broker server name, e.g. 'MetaQuotes-Demo'")
    parser.add_argument(
        "--install-heartbeat",
        action="store_true",
        help="Copy the ConnectionHeartbeat.mq5 script into the terminal for manual diagnostics",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv or sys.argv[1:])

    mt5_dir = Path(args.mt5_dir).expanduser()
    terminal = _resolve_terminal_path(mt5_dir)
    if terminal is None:
        print(f"Could not locate a terminal executable under {mt5_dir}. Pass --mt5-dir pointing to terminal64.exe or set MT5_TERMINAL_PATH.")
    else:
        print(f"Using MetaTrader 5 terminal at {terminal}")

    result = attempt_connection(
        terminal=terminal,
        login=args.login,
        password=args.password,
        server=args.server,
    )

    if result.success:
        print("Successfully connected to MetaTrader 5 via Python.")
        if result.account_login is not None:
            print(f"  Login   : {result.account_login}")
        if result.account_name:
            print(f"  Name    : {result.account_name}")
        if result.broker:
            print(f"  Broker  : {result.broker}")
        if result.leverage is not None:
            print(f"  Leverage: {result.leverage}")
        if result.balance is not None:
            print(f"  Balance : {result.balance}")
        if result.equity is not None:
            print(f"  Equity  : {result.equity}")
    else:
        print("Failed to establish a MetaTrader 5 session via Python.")
        if result.error:
            print(f"  Reason: {result.error}")
            if result.error == "MetaTrader5 package not installed":
                print(
                    "  Hint: On Linux the MetaTrader5 wheel is only available under Wine. "
                    "Run scripts/setup_ubuntu.sh to install the Wine-based toolchain or "
                    "set MT5_TERMINAL_PATH to the Windows terminal and execute this command via Wine."
                )
        print("Ensure the terminal is running and logged in, then retry. If the issue persists, run the ConnectionHeartbeat script inside MetaTrader 5.")

    if args.install_heartbeat:
        script_path = _copy_heartbeat_script(mt5_dir)
        if script_path:
            print(f"Installed heartbeat script to {script_path}")
        else:
            print("Unable to install heartbeat script (source file missing).")

    return 0 if result.success else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
