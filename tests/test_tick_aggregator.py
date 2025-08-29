import asyncio
import contextlib
import time

import pandas as pd
from pathlib import Path
import importlib.util
import types
import sys

analytics_pkg = types.ModuleType("analytics")
analytics_pkg.metrics_store = types.SimpleNamespace(
    record_metric=lambda *a, **k: None
)
analytics_pkg.decision_logger = types.SimpleNamespace(log=lambda *a, **k: None)
sys.modules.setdefault("analytics", analytics_pkg)
sys.modules.setdefault("analytics.metrics_store", analytics_pkg.metrics_store)
sys.modules.setdefault("analytics.decision_logger", analytics_pkg.decision_logger)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

news_pkg = types.ModuleType("news")
news_pkg.impact_model = types.SimpleNamespace(get_impact=lambda *a, **k: (0.0, 0.0))
sys.modules.setdefault("news", news_pkg)
sys.modules.setdefault("news.impact_model", news_pkg.impact_model)

spec = importlib.util.spec_from_file_location(
    "data.tick_aggregator", Path(__file__).resolve().parents[1] / "data" / "tick_aggregator.py"
)
ta = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ta
spec.loader.exec_module(ta)

import risk_manager as rm_mod


class FastBroker:
    COPY_TICKS_ALL = 0

    def copy_ticks_from(self, *args):
        # simple deterministic tick
        return [{"time": args[1], "bid": 1.0, "ask": 2.0, "volume": 1}]


class SlowBroker:
    COPY_TICKS_ALL = 0

    def copy_ticks_from(self, *args):
        time.sleep(0.01)
        return [{"time": args[1], "bid": 1.1, "ask": 2.1, "volume": 1}]


class FailingBroker:
    COPY_TICKS_ALL = 0

    def copy_ticks_from(self, *args):
        raise RuntimeError("fail")


class FlakyBroker:
    COPY_TICKS_ALL = 0

    def __init__(self):
        self.aligned = False

    def copy_ticks_from(self, *args):
        price = 1.5 if not self.aligned else 1.0
        return [{"time": args[1], "bid": price, "ask": price + 1.0, "volume": 1}]


def test_conflict_resolution(monkeypatch):
    metrics = []
    monkeypatch.setattr(ta, "record_metric", lambda *a, **k: metrics.append((a, k)))
    ta.init(FastBroker(), SlowBroker())
    df = asyncio.run(ta.fetch_ticks("EURUSD", 1))
    assert not df.empty
    # fast broker should win due to lower latency
    assert df.iloc[0]["Bid"] == 1.0
    # two latency metrics and one divergence metric recorded
    assert sum(1 for m in metrics if m[0][0] == "tick_source_latency") == 2
    assert any(m[0][0] == "tick_source_divergence" for m in metrics)


def test_failover(monkeypatch):
    metrics = []
    monkeypatch.setattr(ta, "record_metric", lambda *a, **k: metrics.append((a, k)))
    ta.init(FailingBroker(), FastBroker())
    df = asyncio.run(ta.fetch_ticks("EURUSD", 1))
    assert not df.empty
    # only fast broker provided data
    assert df.iloc[0]["Bid"] == 1.0
    # latency metrics recorded for both sources despite failure
    assert sum(1 for m in metrics if m[0][0] == "tick_source_latency") == 2


def test_divergence_alert_blocks_trades(tmp_path, monkeypatch):
    alerts: list[str] = []
    monkeypatch.setattr(ta, "record_metric", lambda *a, **k: None)
    monkeypatch.setattr(ta, "send_alert", lambda msg: alerts.append(msg))
    monkeypatch.setattr(rm_mod, "record_metric", lambda *a, **k: None)
    monkeypatch.setattr(rm_mod, "send_alert", lambda msg: alerts.append(msg))
    monkeypatch.setattr(rm_mod.decision_logger, "log", lambda *a, **k: None)
    monkeypatch.setattr(ta, "_DIVERGENCE_THRESHOLD", 0.05)
    log_path = tmp_path / "anoms.csv"
    monkeypatch.setattr(ta, "_LOG_PATH", log_path)
    primary = FastBroker()
    flaky = FlakyBroker()
    ta.init(primary, flaky)
    rm = rm_mod.RiskManager(max_drawdown=100)

    async def run():
        task = rm_mod.subscribe_to_broker_alerts(rm)
        await ta.fetch_ticks("EURUSD", 1)
        await asyncio.sleep(0)
        assert rm.metrics.trading_halted is True
        assert log_path.exists()
        df = pd.read_csv(log_path)
        assert len(df) == 1
        size = rm.adjust_size("EURUSD", 1.0, pd.Timestamp.utcnow(), 1)
        assert size == 0.0
        flaky.aligned = True
        await ta.fetch_ticks("EURUSD", 1)
        await asyncio.sleep(0)
        assert rm.metrics.trading_halted is False
        size = rm.adjust_size("EURUSD", 1.0, pd.Timestamp.utcnow(), 1)
        assert size == 1.0
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    asyncio.run(run())
