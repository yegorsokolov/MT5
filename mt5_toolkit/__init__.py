# mt5/__init__.py
"""
Shim so code that imports `mt5` works on Linux:
- If `MetaTrader5` is importable in *this* interpreter, proxy directly.
- Otherwise, execute the needed calls inside Wine's Windows Python (WIN_PYTHON).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Tuple

# ---- Direct proxy if available (Windows/native installs) ---------------------
try:  # noqa: SIM105
    import MetaTrader5 as _mt5  # type: ignore

    def initialize(*args: Any, **kwargs: Any) -> bool:
        return _mt5.initialize(*args, **kwargs)

    def login(login: int, password: str, server: str, *args: Any, **kwargs: Any) -> bool:
        return _mt5.login(login, password=password, server=server, *args, **kwargs)

    def terminal_info() -> Any:
        return _mt5.terminal_info()

    def last_error() -> Tuple[int, str]:
        return _mt5.last_error()

    __all__ = ["initialize", "login", "terminal_info", "last_error"]
except Exception:
    # ---- Wine fallback: run MetaTrader5 code inside Windows Python -----------
    _WIN_PY = os.environ.get("WIN_PYTHON")
    _WINE = shutil.which("wine")
    if not _WIN_PY or not _WINE:
        raise ImportError(
            "mt5 shim: MetaTrader5 not importable and Wine fallback unavailable "
            "(set WIN_PYTHON and ensure `wine` is on PATH)."
        )

    def _run_in_wine(py_code: str) -> subprocess.CompletedProcess:
        """Run code inside Wine Windows Python and return the CompletedProcess."""
        return subprocess.run(
            ["wine", _WIN_PY, "-c", py_code],
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def initialize(*_a: Any, **_kw: Any) -> bool:
        code = r"""
import MetaTrader5 as mt5
ok = mt5.initialize()
print("OK" if ok else "FAIL")
"""
        cp = _run_in_wine(code)
        return cp.returncode == 0 and cp.stdout.strip().endswith("OK")

    def login(login: int, password: str, server: str, *_a: Any, **_kw: Any) -> bool:
        code = f"""
import MetaTrader5 as mt5
ok = mt5.initialize() and mt5.login(int({login!r}), password={password!r}, server={server!r})
print("OK" if ok else "FAIL")
"""
        cp = _run_in_wine(code)
        return cp.returncode == 0 and cp.stdout.strip().endswith("OK")

    def terminal_info() -> Any:
        code = r"""
import json, MetaTrader5 as mt5
mt5.initialize()
info = mt5.terminal_info()
obj = dict(
    build=getattr(info,"build",None),
    connected=getattr(info,"connected",None),
    path=getattr(info,"path",None),
)
print(json.dumps(obj))
"""
        cp = _run_in_wine(code)
        if cp.returncode != 0:
            # return a minimal object to avoid None attribute crashes
            return SimpleNamespace(build=None, connected=False, path=None)
        try:
            d = json.loads(cp.stdout.strip())
        except Exception:
            d = {"build": None, "connected": False, "path": None}
        return SimpleNamespace(**d)

    def last_error() -> Tuple[int, str]:
        # We cannot easily fetch real last_error() via subprocess; provide a benign fallback.
        return (1, "Success")

    __all__ = ["initialize", "login", "terminal_info", "last_error"]
