import pandas as pd
import sys
import types
from pathlib import Path

# ensure repo root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# stub mlflow before importing utils
class _DummyRun:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

mlflow_stub = types.SimpleNamespace(
    set_tracking_uri=lambda *a, **k: None,
    set_experiment=lambda *a, **k: None,
    start_run=lambda *a, **k: _DummyRun(),
    log_dict=lambda *a, **k: None,
)
sys.modules.setdefault("mlflow", mlflow_stub)

class _DummyMetric:
    def __init__(self, *a, **k):
        pass

    def inc(self, *a, **k):
        pass

    def set(self, *a, **k):
        pass

prom_stub = types.SimpleNamespace(Counter=_DummyMetric, Gauge=_DummyMetric)
sys.modules.setdefault("prometheus_client", prom_stub)

import utils
import plugins.spread as spread
import plugins.slippage as slippage
import plugins.atr as atr
import plugins.donchian as donchian


def test_spread_slippage_toggle(monkeypatch):
    cfg = {
        "use_spread_check": True,
        "max_spread": 0.1,
        "use_slippage_check": True,
        "max_slippage": 0.1,
    }
    monkeypatch.setattr(utils, "load_config", lambda: cfg)
    monkeypatch.setattr(spread, "load_config", lambda: cfg)
    tick = {"Bid": 1.0, "Ask": 1.2}
    assert spread.check_spread(tick) is False

    cfg["use_spread_check"] = False
    assert spread.check_spread(tick) is True

    monkeypatch.setattr(slippage, "load_config", lambda: cfg)
    order = {"requested_price": 1.0, "filled_price": 1.2}
    cfg["use_slippage_check"] = True
    assert slippage.check_slippage(order) is False

    cfg["use_slippage_check"] = False
    assert slippage.check_slippage(order) is True


def test_atr_toggle(monkeypatch):
    cfg = {"use_atr": True, "atr_period": 2, "atr_mult": 1}
    monkeypatch.setattr(utils, "load_config", lambda: cfg)
    monkeypatch.setattr(atr, "load_config", lambda: cfg)

    df = pd.DataFrame(
        {
            "Bid": [1.0, 1.1, 1.2],
            "Ask": [1.0001, 1.1001, 1.2001],
        }
    )
    out = atr.add_atr_stops(df.copy())
    assert {"atr_14", "atr_stop_long", "atr_stop_short"}.issubset(out.columns)

    cfg["use_atr"] = False
    out2 = atr.add_atr_stops(df.copy())
    assert "atr_14" not in out2.columns


def test_donchian_toggle(monkeypatch):
    cfg = {"use_donchian": True, "donchian_period": 2}
    monkeypatch.setattr(utils, "load_config", lambda: cfg)
    monkeypatch.setattr(donchian, "load_config", lambda: cfg)

    df = pd.DataFrame(
        {
            "Bid": [1.0, 1.1, 1.2],
            "Ask": [1.0001, 1.1001, 1.2001],
        }
    )
    out = donchian.add_donchian_channels(df.copy())
    assert {
        "donchian_high",
        "donchian_low",
        "donchian_break",
    }.issubset(out.columns)

    cfg["use_donchian"] = False
    out2 = donchian.add_donchian_channels(df.copy())
    assert "donchian_high" not in out2.columns
