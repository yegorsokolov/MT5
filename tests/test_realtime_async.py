import asyncio
import time
import types
import sys
from pathlib import Path
import os
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DISABLE_ENV_CHECK", "1")

# stub git module for realtime_train
sys.modules['git'] = types.SimpleNamespace(Repo=lambda *args, **kwargs: None)
sys.modules['yaml'] = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})
sys.modules['utils'] = types.SimpleNamespace(load_config=lambda: {})
data_pkg = types.ModuleType('data')
data_pkg.features = types.SimpleNamespace(make_features=lambda df: df)
data_pkg.sanitize = types.SimpleNamespace(sanitize_ticks=lambda df: df)
data_pkg.trade_log = types.SimpleNamespace(
    TradeLog=lambda *a, **k: types.SimpleNamespace(
        record_order=lambda *a, **k: None,
        record_fill=lambda *a, **k: None,
        sync_mt5_positions=lambda *a, **k: None,
    )
)
sys.modules['data'] = data_pkg
sys.modules['data.features'] = data_pkg.features
sys.modules['data.sanitize'] = data_pkg.sanitize
sys.modules['data.trade_log'] = data_pkg.trade_log
sys.modules['signal_queue'] = types.SimpleNamespace(get_signal_backend=lambda cfg: None)
sys.modules['metrics'] = types.SimpleNamespace(
    RECONNECT_COUNT=types.SimpleNamespace(inc=lambda: None),
    ERROR_COUNT=types.SimpleNamespace(inc=lambda: None),
    TRADE_COUNT=types.SimpleNamespace(inc=lambda: None),
    FEATURE_ANOMALIES=types.SimpleNamespace(inc=lambda: None),
    RESOURCE_RESTARTS=types.SimpleNamespace(inc=lambda: None),
    QUEUE_DEPTH=types.SimpleNamespace(set=lambda v: None),
    BATCH_LATENCY=types.SimpleNamespace(set=lambda v: None),
)
sys.modules['models'] = types.SimpleNamespace(model_store=types.SimpleNamespace(load_model=lambda *a, **k: (None, None)))
import importlib.machinery
sys.modules['mlflow'] = types.SimpleNamespace(
    __loader__=True, __spec__=importlib.machinery.ModuleSpec('mlflow', loader=None)
)
sys.modules['prometheus_client'] = types.SimpleNamespace(Counter=lambda *a, **k: None, Gauge=lambda *a, **k: None)
sys.modules['analysis.anomaly_detector'] = types.SimpleNamespace(
    detect_anomalies=lambda df, quarantine_path=None, counter=None: (df, [])
)


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


def test_pipeline_parallel(monkeypatch):
    monkeypatch.setattr(rt, 'make_features', fake_make_features)
    queue = FakeQueue()

    async def pipeline(sym: str):
        ticks = await rt.fetch_ticks(sym, 1)
        if ticks.empty:
            return
        ticks['Symbol'] = sym
        feats = await rt.generate_features(ticks)
        await rt.dispatch_signals(queue, feats)

    async def run():
        await asyncio.gather(pipeline('EURUSD'), pipeline('GBPUSD'))

    start = time.perf_counter()
    asyncio.run(run())
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5
    assert len(queue.published) == 2

