# mt5/__init__.py
"""
Wine-aware shim so `import mt5` works on Linux and mirrors the MetaTrader5 API.

Behavior:
- If `MetaTrader5` is importable in the current interpreter (e.g., native Windows),
  this module simply proxies to it.
- Otherwise, it uses Wine's Windows-Python (WIN_PYTHON) to execute the same calls
  and returns results to the Linux interpreter.

Provided (Wine fallback):
- initialize(), login(), terminal_info(), last_error()
- TIMEFRAME_* constants (M1, M5, M15, M30, H1, H4, D1, W1, MN1)
- copy_rates_range(symbol, timeframe, from_ts, to_ts) -> list[dict]

You can extend this with more wrappers as needed (copy_rates_from_pos, order_send, ...).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Tuple

# ======= Native proxy path (Windows / MetaTrader5 installed in this Python) =======
try:
    import MetaTrader5 as _mt5  # type: ignore

    # Direct passthrough of commonly-used functions/attrs
    initialize = _mt5.initialize
    login = _mt5.login
    terminal_info = _mt5.terminal_info
    last_error = _mt5.last_error

    # Expose timeframes directly from MetaTrader5
    TIMEFRAME_M1 = _mt5.TIMEFRAME_M1
    TIMEFRAME_M5 = _mt5.TIMEFRAME_M5
    TIMEFRAME_M15 = _mt5.TIMEFRAME_M15
    TIMEFRAME_M30 = _mt5.TIMEFRAME_M30
    TIMEFRAME_H1 = _mt5.TIMEFRAME_H1
    TIMEFRAME_H4 = _mt5.TIMEFRAME_H4
    TIMEFRAME_D1 = _mt5.TIMEFRAME_D1
    TIMEFRAME_W1 = _mt5.TIMEFRAME_W1
    TIMEFRAME_MN1 = _mt5.TIMEFRAME_MN1

    # Expose copy_rates_range directly
    copy_rates_range = _mt5.copy_rates_range

    __all__ = [
        # funcs
        "initialize", "login", "terminal_info", "last_error", "copy_rates_range",
        # consts
        "TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30",
        "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1", "TIMEFRAME_W1", "TIMEFRAME_MN1",
    ]

except Exception:
    # ======= Wine fallback path (Linux with terminal under Wine) =======
    _WIN_PY = os.environ.get("WIN_PYTHON")
    _WINE = shutil.which("wine")
    if not _WIN_PY or not _WINE:
        raise ImportError(
            "mt5 shim: MetaTrader5 not importable and Wine fallback unavailable "
            "(set WIN_PYTHON and ensure `wine` is on PATH)."
        )

    def _run_in_wine(py_code: str) -> subprocess.CompletedProcess:
        """Run code inside Wine's Windows-Python and return CompletedProcess."""
        return subprocess.run(
            ["wine", _WIN_PY, "-c", py_code],
            env=os.environ.copy(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    # ---------- Core ops ----------
    def initialize(*_a: Any, **_kw: Any) -> bool:
        cp = _run_in_wine(
            "import MetaTrader5 as mt5; print('OK' if mt5.initialize() else 'FAIL')"
        )
        return cp.returncode == 0 and cp.stdout.strip().endswith("OK")

    def login(login: int, password: str, server: str, *_a: Any, **_kw: Any) -> bool:
        code = (
            f"import MetaTrader5 as mt5; "
            f"ok=mt5.initialize() and mt5.login(int({login!r}), password={password!r}, server={server!r}); "
            f"print('OK' if ok else 'FAIL')"
        )
        cp = _run_in_wine(code)
        return cp.returncode == 0 and cp.stdout.strip().endswith("OK")

    def terminal_info() -> Any:
        code = (
            "import json, MetaTrader5 as mt5; mt5.initialize(); info=mt5.terminal_info(); "
            "print(json.dumps({'build':getattr(info,'build',None),"
            "'connected':getattr(info,'connected',None),'path':getattr(info,'path',None)}))"
        )
        cp = _run_in_wine(code)
        if cp.returncode != 0:
            return SimpleNamespace(build=None, connected=False, path=None)
        try:
            d = json.loads(cp.stdout.strip())
        except Exception:
            d = {"build": None, "connected": False, "path": None}
        return SimpleNamespace(**d)

    def last_error() -> Tuple[int, str]:
        # Not easily retrievable via subprocess across calls; benign fallback
        return (1, "Success")

    # ---------- Constants (TIMEFRAME_*) ----------
    # Pull the numeric values from MetaTrader5 once and cache them on this module.
    _CONST_JSON = (
        "import json, MetaTrader5 as mt5; "
        "print(json.dumps({"
        "'TIMEFRAME_M1': mt5.TIMEFRAME_M1, 'TIMEFRAME_M5': mt5.TIMEFRAME_M5, "
        "'TIMEFRAME_M15': mt5.TIMEFRAME_M15, 'TIMEFRAME_M30': mt5.TIMEFRAME_M30, "
        "'TIMEFRAME_H1': mt5.TIMEFRAME_H1, 'TIMEFRAME_H4': mt5.TIMEFRAME_H4, "
        "'TIMEFRAME_D1': mt5.TIMEFRAME_D1, 'TIMEFRAME_W1': mt5.TIMEFRAME_W1, "
        "'TIMEFRAME_MN1': mt5.TIMEFRAME_MN1 }))"
    )

    def _ensure_timeframe_consts_loaded() -> None:
        global TIMEFRAME_M1, TIMEFRAME_M5, TIMEFRAME_M15, TIMEFRAME_M30
        global TIMEFRAME_H1, TIMEFRAME_H4, TIMEFRAME_D1, TIMEFRAME_W1, TIMEFRAME_MN1
        if "TIMEFRAME_M1" in globals():
            return
        cp = _run_in_wine(_CONST_JSON)
        if cp.returncode != 0:
            raise RuntimeError(f"Failed to load timeframe constants: {cp.stderr}")
        d = json.loads(cp.stdout.strip())
        for k, v in d.items():
            globals()[k] = v

    _ensure_timeframe_consts_loaded()

    # ---------- Data pulling ----------
    def copy_rates_range(symbol: str, timeframe: int, from_ts: int, to_ts: int) -> List[Dict[str, Any]]:
        """
        Mirror MetaTrader5.copy_rates_range. `from_ts`/`to_ts` are UNIX timestamps (UTC).
        Returns a list of dicts with keys: time, open, high, low, close, tick_volume, spread, real_volume.
        """
        code = (
            "import json, MetaTrader5 as mt5; from_ts={fts}; to_ts={tts}; sym={sym!r}; tf={tf};"
            "mt5.initialize(); "
            "rates = mt5.copy_rates_range(sym, tf, from_ts, to_ts) or []; "
            "out = ["
            "  {'time': int(r['time']), 'open': float(r['open']), 'high': float(r['high']), "
            "   'low': float(r['low']), 'close': float(r['close']), 'tick_volume': int(r['tick_volume']), "
            "   'spread': int(r['spread']), 'real_volume': int(r['real_volume'])}"
            "  for r in rates"
            "]; "
            "print(json.dumps(out))"
        ).format(fts=int(from_ts), tts=int(to_ts), sym=symbol, tf=int(timeframe))
        cp = _run_in_wine(code)
        if cp.returncode != 0:
            raise RuntimeError(f"copy_rates_range failed: {cp.stderr}")
        return json.loads(cp.stdout.strip())

    __all__ = [
        # funcs
        "initialize", "login", "terminal_info", "last_error", "copy_rates_range",
        # consts
        "TIMEFRAME_M1", "TIMEFRAME_M5", "TIMEFRAME_M15", "TIMEFRAME_M30",
        "TIMEFRAME_H1", "TIMEFRAME_H4", "TIMEFRAME_D1", "TIMEFRAME_W1", "TIMEFRAME_MN1",
    ]
