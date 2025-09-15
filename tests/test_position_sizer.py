import types
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

# Ensure metrics module exists
sys.modules.setdefault(
    "analytics.metrics_store", types.SimpleNamespace(record_metric=lambda *a, **k: None)
)

from risk.position_sizer import PositionSizer


def test_size_reduced_by_slippage_and_liquidity(monkeypatch):
    calls = []

    def fake_metric(name, value, **kwargs):
        calls.append((name, value))

    monkeypatch.setattr("risk.position_sizer.record_metric", fake_metric)
    sizer = PositionSizer(capital=1000.0, method="kelly")

    base = sizer.size(prob=0.6)
    with_slip = sizer.size(prob=0.6, slippage=0.5)
    with_liq = sizer.size(prob=0.6, liquidity=base / 2)

    assert with_slip < base
    assert with_liq == pytest.approx(base / 2)
    assert any(name == "slip_adj_realized_risk" for name, _ in calls)


def test_split_size_breaks_large_positions():
    sizer = PositionSizer(capital=1000.0, method="kelly")
    sizer._last_size["EURUSD"] = 1.0
    parts = sizer.split_size("EURUSD", 5.0)
    assert sum(parts) == pytest.approx(5.0)
    assert len(parts) > 1
    last = 1.0
    for p in parts:
        assert abs(p) <= last * sizer.max_martingale_multiplier + 1e-9
        last = abs(p)
    assert sizer._last_size["EURUSD"] == pytest.approx(abs(parts[-1]))
