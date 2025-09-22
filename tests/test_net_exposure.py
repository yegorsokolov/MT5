import sys
from pathlib import Path

import pytest
import types
import importlib

sys.modules.pop("pandas", None)
pd = importlib.import_module("pandas")

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
sys.modules["requests"] = types.SimpleNamespace(Session=lambda *a, **k: None)

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))


def test_net_exposure_limits_and_logging(tmp_path, monkeypatch):
    import analytics.metrics_store as ms

    logged: list[tuple[str, float, dict | None]] = []
    monkeypatch.setattr(
        ms, "record_metric", lambda name, value, tags=None, path=None: logged.append((name, value, tags))
    )
    monkeypatch.chdir(tmp_path)
from mt5.risk_manager import RiskManager

    rm = RiskManager(
        max_drawdown=1e9,
        max_var=1e9,
        max_long_exposure=100.0,
        max_short_exposure=80.0,
    )

    for i in range(1, 6):
        rm.net_exposure.record_returns({"AAA": 0.01 * i, "BBB": 0.01 * i, "CCC": 0.01 * i, "DDD": 0.01 * i})

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


def test_correlation_limiter(tmp_path, monkeypatch):
    import analytics.metrics_store as ms

    logged: list[tuple[str, float]] = []
    monkeypatch.setattr(
        ms, "record_metric", lambda name, value, tags=None, path=None: logged.append((name, value))
    )
    monkeypatch.chdir(tmp_path)

    import importlib
    from risk import net_exposure as ne_mod
    importlib.reload(ne_mod)
    NetExposure = ne_mod.NetExposure

    ne = NetExposure(max_long=100.0, window=3)
    ne.record_returns({"AAA": 0.01, "BBB": 0.02})
    ne.record_returns({"AAA": 0.02, "BBB": 0.01})
    ne.record_returns({"AAA": 0.03, "BBB": 0.02})
    ne.update("AAA", 60)
    uncorr_allowed = ne.limit("BBB", 70)
    for r in [0.01, 0.02, 0.03]:
        ne.record_returns({"AAA": r, "BBB": r})
    corr_allowed = ne.limit("BBB", 70)
    assert corr_allowed < uncorr_allowed
    assert ne.corr.loc["AAA", "BBB"] > 0.9
    avg = [v for n, v in logged if n == "avg_correlation"][-1]
    assert avg > 0.9


def test_net_exposure_thread_safe(tmp_path, monkeypatch):
    import analytics.metrics_store as ms

    monkeypatch.setattr(
        ms, "record_metric", lambda *a, **k: None
    )
    monkeypatch.chdir(tmp_path)

    from risk.net_exposure import NetExposure

    ne = NetExposure()

    import threading

    def worker():
        for _ in range(100):
            ne.update("AAA", 1)

    threads = [threading.Thread(target=worker) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    totals = ne.totals()
    assert totals["long"] == pytest.approx(1000.0)


def test_net_exposure_persistence_round_trip(tmp_path, monkeypatch):
    import analytics.metrics_store as ms

    monkeypatch.setattr(
        ms, "record_metric", lambda *a, **k: None
    )
    monkeypatch.chdir(tmp_path)

    from risk.net_exposure import NetExposure

    ne = NetExposure()
    ne.update("AAA", 10)
    ne.update("BBB", -5)

    ne2 = NetExposure()
    assert ne2.long["AAA"] == pytest.approx(10.0)
    assert ne2.short["BBB"] == pytest.approx(5.0)
    assert ne2.totals() == ne.totals()
