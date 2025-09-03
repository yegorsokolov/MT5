import types
from datetime import datetime, timedelta, timezone

import pandas as pd

import risk_manager as rm_mod
from risk_manager import RiskManager


def _stub_metric(logged):
    def _rec(name, value, tags=None, path=None):
        logged.append((name, value, tags or {}))
    return _rec


def test_currency_specific_blackout(monkeypatch):
    logged = []
    monkeypatch.setattr(rm_mod, "record_metric", _stub_metric(logged))
    monkeypatch.setattr("risk_manager.get_impact", lambda *a, **k: (0.0, 0.0))
    monkeypatch.setattr("risk_manager.decision_logger", types.SimpleNamespace(log=lambda *a, **k: None))

    rm = RiskManager(max_drawdown=100, max_total_drawdown=1000)
    ts = pd.Timestamp("2024-01-01", tz=timezone.utc)
    window = {
        "start": ts - pd.Timedelta(minutes=5),
        "end": ts + pd.Timedelta(minutes=5),
        "currencies": ["USD"],
        "symbols": [],
    }
    rm.set_quiet_windows([window])
    size = rm.adjust_size("EURUSD", 1.0, ts, 1)
    assert size == 0.0
    assert logged and logged[0][0] == "trades_skipped_news"
    assert logged[0][2]["symbol"] == "EURUSD"

    # Non-USD pair should trade normally
    size2 = rm.adjust_size("EURJPY", 1.0, ts, 1)
    assert size2 == 1.0
    assert len(logged) == 1


def test_symbol_specific_blackout(monkeypatch):
    logged = []
    monkeypatch.setattr(rm_mod, "record_metric", _stub_metric(logged))
    monkeypatch.setattr("risk_manager.get_impact", lambda *a, **k: (0.0, 0.0))
    monkeypatch.setattr("risk_manager.decision_logger", types.SimpleNamespace(log=lambda *a, **k: None))

    rm = RiskManager(max_drawdown=100, max_total_drawdown=1000)
    ts = pd.Timestamp("2024-01-01", tz=timezone.utc)
    window = {
        "start": ts - pd.Timedelta(minutes=5),
        "end": ts + pd.Timedelta(minutes=5),
        "currencies": [],
        "symbols": ["AAPL"],
    }
    rm.set_quiet_windows([window])
    size = rm.adjust_size("AAPL", 1.0, ts, 1)
    assert size == 0.0
    assert logged and logged[0][2]["symbol"] == "AAPL"

    # Different symbol should trade
    size2 = rm.adjust_size("MSFT", 1.0, ts, 1)
    assert size2 == 1.0
    assert len(logged) == 1
