import sys
from pathlib import Path

import pandas as pd
import pytest
import types

sys.modules["prometheus_client"] = types.SimpleNamespace(
    Counter=lambda *a, **k: None,
    Gauge=lambda *a, **k: None,
)
sys.modules["crypto_utils"] = types.SimpleNamespace(
    _load_key=lambda *a, **k: b"", encrypt=lambda x, *a, **k: x, decrypt=lambda x, *a, **k: x
)
sys.modules["analysis.extreme_value"] = types.SimpleNamespace(
    estimate_tail_probability=lambda *a, **k: (0.0, {}),
    log_evt_result=lambda *a, **k: None,
)
sys.modules["news"] = types.SimpleNamespace()
sys.modules["news.impact_model"] = types.SimpleNamespace(
    get_impact=lambda *a, **k: (0.0, 0.0)
)

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def test_net_exposure_limits_and_logging(tmp_path, monkeypatch):
    import analytics.metrics_store as ms

    logged: list[tuple[str, float, dict | None]] = []
    monkeypatch.setattr(
        ms, "record_metric", lambda name, value, tags=None, path=None: logged.append((name, value, tags))
    )

    from risk_manager import RiskManager

    rm = RiskManager(
        max_drawdown=1e9,
        max_var=1e9,
        max_long_exposure=100.0,
        max_short_exposure=80.0,
    )

    ts = pd.Timestamp("2024-01-01")
    s1 = rm.adjust_size("AAA", 60, ts, direction=1)
    assert s1 == 60
    rm.update("b1", 0.0, s1, symbol="AAA")

    s2 = rm.adjust_size("BBB", 50, ts, direction=1)
    assert s2 == 40
    rm.update("b2", 0.0, s2, symbol="BBB")

    s3 = rm.adjust_size("CCC", 10, ts, direction=1)
    assert s3 == 0

    totals = rm.net_exposure.totals()
    assert totals["long"] == pytest.approx(100.0)
    assert totals["short"] == pytest.approx(0.0)

    s4 = rm.adjust_size("AAA", 30, ts, direction=-1)
    assert s4 == 30
    rm.update("b3", 0.0, -s4, symbol="AAA")

    s5 = rm.adjust_size("BBB", 100, ts, direction=-1)
    assert s5 == 50
    rm.update("b4", 0.0, -s5, symbol="BBB")

    s6 = rm.adjust_size("DDD", 10, ts, direction=-1)
    assert s6 == 0

    totals = rm.net_exposure.totals()
    assert totals["long"] == pytest.approx(100.0)
    assert totals["short"] == pytest.approx(80.0)

    long_val = [v for n, v, _ in logged if n == "long_exposure"][-1]
    short_val = [v for n, v, _ in logged if n == "short_exposure"][-1]
    assert long_val == pytest.approx(100.0)
    assert short_val == pytest.approx(80.0)
