import datetime as dt

import sys
import types
from pathlib import Path

import pandas as pd
import pytest


class _Symbol:
    def __init__(self, name: str) -> None:
        self.name = name


class _MT5Mock:
    """Lightweight mock of the MetaTrader5 API."""

    COPY_TICKS_ALL = 0

    def __init__(self, ticks: list[dict], symbols: list[str]) -> None:
        self._ticks = ticks
        self._symbols = symbols
        self.selected: list[tuple[str, bool]] = []

    def initialize(self, **kwargs):
        return True

    def shutdown(self):
        return None

    def symbol_info(self, symbol: str):
        if symbol in self._symbols:
            return _Symbol(symbol)
        return None

    def symbols_get(self):
        return [_Symbol(s) for s in self._symbols]

    def symbol_select(self, symbol: str, enable: bool):
        self.selected.append((symbol, enable))
        return True

    def copy_ticks_range(self, symbol: str, start: int, end: int, flags: int):
        return [t for t in self._ticks if start <= t["time"] < end]


def test_fetch_history_formats_and_stores(monkeypatch, tmp_path):
    ticks = [
        {"time": 1_000_000, "bid": 1.0, "ask": 1.1, "volume": 2},
        {"time": 1_000_010, "bid": 1.2, "ask": 1.3, "volume": 3},
    ]
    mt5 = _MT5Mock(ticks, ["EURUSD"])
    monkeypatch.setitem(sys.modules, "MetaTrader5", types.SimpleNamespace())
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import brokers.mt5_direct as mt5_direct
    monkeypatch.setattr(mt5_direct, "_mt5", mt5)

    start = dt.datetime.utcfromtimestamp(1_000_000)
    end = dt.datetime.utcfromtimestamp(1_000_020)
    df = mt5_direct.fetch_history("EURUSD", start, end)

    assert list(df.columns) == [
        "Timestamp",
        "Bid",
        "Ask",
        "BidVolume",
        "AskVolume",
    ]
    assert len(df) == 2

    pytest.importorskip("pyarrow")
    out = tmp_path / "history.parquet"
    df.to_parquet(out, index=False)
    loaded = pd.read_parquet(out)
    pd.testing.assert_frame_equal(df, loaded)


def test_fetch_history_symbol_variants(monkeypatch):
    ticks = [
        {"time": 1_000_000, "bid": 1.5, "ask": 1.6, "volume": 1, "flags": 0},
    ]
    mt5 = _MT5Mock(ticks, ["EURUSDm"])
    monkeypatch.setitem(sys.modules, "MetaTrader5", types.SimpleNamespace())
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))
    import brokers.mt5_direct as mt5_direct
    monkeypatch.setattr(mt5_direct, "_mt5", mt5)

    start = dt.datetime.utcfromtimestamp(1_000_000)
    end = dt.datetime.utcfromtimestamp(1_000_010)
    df = mt5_direct.fetch_history("EURUSD", start, end)

    assert len(df) == 1
    assert mt5.selected[0][0] == "EURUSDm"
    assert "flags" not in df.columns

