import pandas as pd
import pytest
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "trade_analyzer", Path(__file__).resolve().parents[1] / "analytics" / "trade_analyzer.py"
)
trade_analyzer = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trade_analyzer)
TradeAnalyzer = trade_analyzer.TradeAnalyzer


def _sample_trades():
    return [
        {
            "entry_time": pd.Timestamp("2024-01-01 09:00"),
            "exit_time": pd.Timestamp("2024-01-01 10:00"),
            "entry_price": 100,
            "exit_price": 110,
            "volume": 1.0,
            "pnl": 10.0,
        },
        {
            "entry_time": pd.Timestamp("2024-01-01 09:30"),
            "exit_time": pd.Timestamp("2024-01-01 11:30"),
            "entry_price": 200,
            "exit_price": 180,
            "volume": 2.0,
            "pnl": -40.0,
        },
        {
            "entry_time": pd.Timestamp("2024-01-01 12:00"),
            "exit_time": pd.Timestamp("2024-01-01 12:30"),
            "entry_price": 50,
            "exit_price": 55,
            "volume": 1.5,
            "pnl": 7.5,
        },
    ]


def test_trade_analyzer_metrics():
    analyzer = TradeAnalyzer.from_records(_sample_trades())

    assert analyzer.average_hold_time() == pd.Timedelta(minutes=70)

    pnl_by_duration = analyzer.pnl_by_duration()
    assert pnl_by_duration[60] == pytest.approx(10.0)
    assert pnl_by_duration[120] == pytest.approx(-40.0)
    assert pnl_by_duration[30] == pytest.approx(7.5)

    assert analyzer.turnover() == pytest.approx(1127.5)
