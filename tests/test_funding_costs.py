import types
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

sys.modules.setdefault(
    "analytics.metrics_store", types.SimpleNamespace(record_metric=lambda *a, **k: None)
)

from risk.position_sizer import PositionSizer
from risk.funding_costs import FundingInfo


def test_funding_cost_and_margin(monkeypatch):
    calls = []

    def fake_metric(name, value, **kwargs):
        calls.append((name, value))

    monkeypatch.setattr("risk.position_sizer.record_metric", fake_metric)

    def fake_fetch(symbol: str) -> FundingInfo:
        return FundingInfo(swap_rate=0.1, margin_requirement=1.0, available_margin=50.0)

    monkeypatch.setattr("risk.position_sizer.fetch_funding_info", fake_fetch)

    sizer = PositionSizer(capital=1000.0, method="kelly")
    size = sizer.size(prob=0.6, symbol="EURUSD")

    assert size == pytest.approx(50.0)
    metrics = dict(calls)
    assert metrics["expected_funding_cost"] == pytest.approx(20.0)
    assert metrics["margin_required"] == pytest.approx(50.0)
    assert metrics["margin_available"] == pytest.approx(50.0)
