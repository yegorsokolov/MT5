import asyncio
import sys
import types
from pathlib import Path
import os
import pandas as pd
import pytest

# stub heavy modules before importing realtime_train
sys.path.append(str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("DISABLE_ENV_CHECK", "1")

brokers_pkg = types.ModuleType("brokers")
brokers_pkg.connection_manager = types.SimpleNamespace(
    init=lambda *a, **k: None,
    get_active_broker=lambda: None,
    failover=lambda: None,
)
sys.modules["brokers"] = brokers_pkg
sys.modules["brokers.connection_manager"] = brokers_pkg.connection_manager

sys.modules["MetaTrader5"] = types.SimpleNamespace(COPY_TICKS_ALL=0)

sys.modules["git"] = types.SimpleNamespace(Repo=lambda *a, **k: None)
sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *a, **k: {})

utils_pkg = types.ModuleType("utils")
utils_pkg.load_config = lambda: {}
utils_pkg.resource_monitor = types.SimpleNamespace(ResourceMonitor=lambda *a, **k: types.SimpleNamespace(start=lambda: None))
sys.modules["utils"] = utils_pkg
sys.modules["utils.resource_monitor"] = utils_pkg.resource_monitor

signal_pkg = types.ModuleType("signal_queue")
signal_pkg.get_signal_backend = lambda cfg: None
sys.modules["signal_queue"] = signal_pkg

latency_values: list = []

metrics_pkg = types.SimpleNamespace(
    RECONNECT_COUNT=types.SimpleNamespace(inc=lambda: None),
    FEATURE_ANOMALIES=types.SimpleNamespace(inc=lambda: None),
    RESOURCE_RESTARTS=types.SimpleNamespace(inc=lambda: None),
    QUEUE_DEPTH=types.SimpleNamespace(set=lambda v: None),
    BATCH_LATENCY=types.SimpleNamespace(set=lambda v: latency_values.append(v)),
)

from data import features as real_features
from data import sanitize as real_sanitize

sys.modules["data.features"] = types.SimpleNamespace(make_features=real_features.make_features)
sys.modules["data.sanitize"] = types.SimpleNamespace(sanitize_ticks=real_sanitize.sanitize_ticks)

def test_queue_throttles_under_load(monkeypatch):
    async def fake_fetch(symbol: str, n: int = 1000, retries: int = 3):
        return pd.DataFrame(
            {
                "Timestamp": pd.date_range("2020", periods=10, freq="s"),
                "Bid": [1.0] * 10,
                "Ask": [1.1] * 10,
                "BidVolume": [1.0] * 10,
                "AskVolume": [1.0] * 10,
            }
        )

    async def slow_process(df: pd.DataFrame):
        await asyncio.sleep(0.2)

    import importlib
    monkeypatch.setitem(sys.modules, "metrics", metrics_pkg)
    sys.modules.pop("realtime_train", None)
    rt = importlib.import_module("realtime_train")
    monkeypatch.setattr(rt, "fetch_ticks", fake_fetch)

    q = asyncio.Queue()

    async def run_test():
        producer = asyncio.create_task(rt.tick_producer(["EURUSD"], q, fake_fetch, throttle_threshold=5))
        worker = asyncio.create_task(rt.tick_worker(q, slow_process, target_latency=0.05))
        await asyncio.sleep(1)
        assert latency_values
        producer.cancel()
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await producer
        with pytest.raises(asyncio.CancelledError):
            await worker

    asyncio.run(run_test())
    assert q.qsize() <= 6
