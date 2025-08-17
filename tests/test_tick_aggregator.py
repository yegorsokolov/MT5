import asyncio
import time

import pandas as pd

from data import tick_aggregator as ta


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
