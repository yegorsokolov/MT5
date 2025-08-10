import asyncio
import time
import types
import sys
from pathlib import Path
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))

# stub git module for realtime_train
sys.modules['git'] = types.SimpleNamespace(Repo=lambda *args, **kwargs: None)
sys.modules['yaml'] = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})
import importlib.machinery
sys.modules['mlflow'] = types.SimpleNamespace(
    __loader__=True, __spec__=importlib.machinery.ModuleSpec('mlflow', loader=None)
)
sys.modules['prometheus_client'] = types.SimpleNamespace(Counter=lambda *a, **k: None, Gauge=lambda *a, **k: None)


# Create fake MetaTrader5 module before importing realtime_train

def _fake_copy(symbol, start, n, mode):
    time.sleep(0.1)
    return [{"time": start, "bid": 1.0, "ask": 1.1, "volume": 1}]

def _fake_init():
    return True

fake_mt5 = types.SimpleNamespace(COPY_TICKS_ALL=0, copy_ticks_from=_fake_copy, initialize=_fake_init)
sys.modules['MetaTrader5'] = fake_mt5

import realtime_train as rt


class FakeQueue:
    def __init__(self):
        self.published = []

    def publish_dataframe(self, df):
        time.sleep(0.1)
        self.published.extend(df.to_dict('records'))


def fake_make_features(df):
    time.sleep(0.1)
    return df


@pytest.mark.asyncio
async def test_pipeline_parallel(monkeypatch):
    monkeypatch.setattr(rt, 'make_features', fake_make_features)
    queue = FakeQueue()

    async def pipeline(sym: str):
        ticks = await rt.fetch_ticks(sym, 1)
        if ticks.empty:
            return
        ticks['Symbol'] = sym
        feats = await rt.generate_features(ticks)
        await rt.dispatch_signals(queue, feats)

    start = time.perf_counter()
    await asyncio.gather(pipeline('EURUSD'), pipeline('GBPUSD'))
    elapsed = time.perf_counter() - start
    # sequential would be about 0.6s; concurrent should be under 0.5s
    assert elapsed < 0.5
    assert len(queue.published) == 2

