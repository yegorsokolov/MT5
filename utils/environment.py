# utils/environment.py
from __future__ import annotations

import os
from typing import Any, Dict

# Prefer the local 'mt5' shim (works on Linux via Wine). Fall back to MetaTrader5 if available.
try:
    import mt5 as _mt5  # our shim
except Exception:
    try:
        import MetaTrader5 as _mt5  # type: ignore
    except Exception:
        _mt5 = None  # handled in checks below


def _result(name: str, status: str, detail: str, followup: str | None = None) -> Dict[str, Any]:
    d: Dict[str, Any] = {"name": name, "status": status, "detail": detail}
    if followup:
        d["followup"] = followup
    return d


def _check_mt5_login() -> Dict[str, Any]:
    """Initialize + login to MT5 and report a human-readable result.

    Uses the local 'mt5' shim if available (recommended on Linux), otherwise tries MetaTrader5.
    Requires env: MT5_LOGIN, MT5_PASSWORD, MT5_SERVER.
    """
    name = "MetaTrader 5 login"

    if _mt5 is None:
        return _result(
            name,
            "failed",
            "MetaTrader5/mt5 module not importable in this interpreter.",
            "Ensure the 'mt5' shim is on PYTHONPATH or run via Wine with WIN_PYTHON configured.",
        )

    missing = [k for k in ("MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER") if not os.environ.get(k)]
    if missing:
        return _result(
            name,
            "failed",
            f"Missing environment variables: {', '.join(missing)}",
            "Populate .env/.env.local with MT5 credentials and reload the shell.",
        )

    login = int(os.environ["MT5_LOGIN"])
    password = os.environ["MT5_PASSWORD"]
    server = os.environ["MT5_SERVER"]

    # initialize
    try:
        ok_init = _mt5.initialize()
    except Exception as e:
        err = ""
        try:
            err = str(getattr(_mt5, "last_error", lambda: ("", ""))())
        except Exception:
            pass
        return _result(
            name,
            "failed",
            f"initialize() raised: {e} last_error={err}",
            "Confirm the Windows MT5 terminal is installed and reachable (Wine path correct).",
        )

    if not ok_init:
        err = ""
        try:
            err = str(getattr(_mt5, "last_error", lambda: ("", ""))())
        except Exception:
            pass
        return _result(
            name,
            "failed",
            f"initialize() failed: {err}",
            "Make sure the terminal is launched (or Wine can launch it) and try again.",
        )

    # login
    try:
        ok_login = _mt5.login(login, password=password, server=server)
    except Exception as e:
        err = ""
        try:
            err = str(getattr(_mt5, "last_error", lambda: ("", ""))())
        except Exception:
            pass
        return _result(
            name,
            "failed",
            f"login() raised: {e} last_error={err}",
            "Verify MT5_LOGIN/MT5_PASSWORD/MT5_SERVER are correct.",
        )

    # terminal info
    build = connected = path = None
    try:
        info = _mt5.terminal_info()
        build = getattr(info, "build", None)
        connected = getattr(info, "connected", None)
        path = getattr(info, "path", None)
    except Exception:
        pass

    if ok_login and connected:
        return _result(name, "passed", f"Logged in; build={build}, connected={connected}, path={path}")

    err = ""
    try:
        err = str(getattr(_mt5, "last_error", lambda: ("", ""))())
    except Exception:
        pass
    return _result(
        name,
        "failed",
        f"MetaTrader 5 terminal not logged in. last_error={err}, build={build}, connected={connected}, path={path}",
        "Launch the MT5 terminal and verify credentials; see .env for MT5_* values.",
    )


def _check_mt5_connectivity() -> Dict[str, Any]:
    name = "MetaTrader 5 connectivity"
    if _mt5 is None:
        return _result(name, "failed", "MetaTrader5/mt5 module not importable.", "Ensure the 'mt5' shim is available.")

    # Ensure initialized
    try:
        _mt5.initialize()
    except Exception:
        pass

    # Try a trivial call
    try:
        info = _mt5.terminal_info()
        return _result(
            name,
            "passed",
            f"Terminal responded (build={getattr(info,'build',None)}, connected={getattr(info,'connected',None)})",
        )
    except Exception as e:
        return _result(
            name,
            "failed",
            f"Connectivity check failed: {e}",
            "Confirm terminal is running and reachable.",
        )


if __name__ == "__main__":
    # tiny CLI to run the two checks directly
    import json
    out = [_check_mt5_login(), _check_mt5_connectivity()]
    print(json.dumps(out, indent=2))
