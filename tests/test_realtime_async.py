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
sys.modules["git"] = types.SimpleNamespace(Repo=lambda *args, **kwargs: None)
sys.modules["yaml"] = types.SimpleNamespace(safe_load=lambda *args, **kwargs: {})
utils_mod = types.ModuleType("utils")
utils_mod.load_config = lambda: {}
utils_mod.__path__ = []
sys.modules["utils"] = utils_mod
res_mod = types.ModuleType("utils.resource_monitor")
res_mod.monitor = lambda *a, **k: None
sys.modules["utils.resource_monitor"] = res_mod
utils_mod.resource_monitor = res_mod
sys.modules["utils.graceful_exit"] = types.SimpleNamespace(graceful_exit=lambda: None)
sys.modules["requests"] = types.SimpleNamespace(get=lambda *a, **k: None)
sys.modules["crypto_utils"] = types.SimpleNamespace(
    _load_key=lambda: b"", encrypt=lambda *a, **k: b"", decrypt=lambda *a, **k: b""
)
data_pkg = types.ModuleType("data")
data_pkg.features = types.SimpleNamespace(make_features=lambda df: df)
data_pkg.sanitize = types.SimpleNamespace(sanitize_ticks=lambda df: df)
data_pkg.trade_log = types.SimpleNamespace(
    TradeLog=lambda *a, **k: types.SimpleNamespace(
        record_order=lambda *a, **k: None,
        record_fill=lambda *a, **k: None,
        sync_mt5_positions=lambda *a, **k: None,
    )
)
data_pkg.feature_scaler = types.SimpleNamespace(FeatureScaler=lambda: None)
sys.modules["data"] = data_pkg
sys.modules["data.features"] = data_pkg.features
sys.modules["data.sanitize"] = data_pkg.sanitize
sys.modules["data.trade_log"] = data_pkg.trade_log
sys.modules["data.feature_scaler"] = data_pkg.feature_scaler
sys.modules["data.live_recorder"] = types.SimpleNamespace(
    LiveRecorder=lambda *a, **k: None
)


async def _publish(bus, df):
    bus.publish_dataframe(df)


sys.modules["signal_queue"] = types.SimpleNamespace(
    get_signal_backend=lambda cfg: None,
    publish_dataframe_async=_publish,
)
sys.modules["metrics"] = types.SimpleNamespace(
    RECONNECT_COUNT=types.SimpleNamespace(inc=lambda: None),
    ERROR_COUNT=types.SimpleNamespace(inc=lambda: None),
    TRADE_COUNT=types.SimpleNamespace(inc=lambda: None),
    FEATURE_ANOMALIES=types.SimpleNamespace(inc=lambda: None),
    RESOURCE_RESTARTS=types.SimpleNamespace(inc=lambda: None),
    QUEUE_DEPTH=types.SimpleNamespace(set=lambda v: None),
    BATCH_LATENCY=types.SimpleNamespace(set=lambda v: None),
    BROKER_LATENCY_MS=types.SimpleNamespace(
        labels=lambda **k: types.SimpleNamespace(set=lambda v: None)
    ),
    BROKER_FAILURES=types.SimpleNamespace(
        labels=lambda **k: types.SimpleNamespace(inc=lambda: None)
    ),
    SLIPPAGE_BPS=types.SimpleNamespace(set=lambda v: None),
    REALIZED_SLIPPAGE_BPS=types.SimpleNamespace(set=lambda v: None),
    PRED_CACHE_HIT=types.SimpleNamespace(inc=lambda: None),
    PRED_CACHE_HIT_RATIO=types.SimpleNamespace(set=lambda v: None),
    PIPELINE_ANOMALY_TOTAL=types.SimpleNamespace(inc=lambda *a, **k: None),
    PIPELINE_ANOMALY_RATE=types.SimpleNamespace(set=lambda v: None),
)
sys.modules["models"] = types.SimpleNamespace(
    model_store=types.SimpleNamespace(load_model=lambda *a, **k: (None, None)),
    residual_learner=None,
)
sys.modules["model_registry"] = types.SimpleNamespace(
    register_policy=lambda *a, **k: None,
    get_policy_path=lambda *a, **k: None,
    save_model=lambda *a, **k: None,
    ModelRegistry=types.SimpleNamespace,
)
sys.modules["train_online"] = types.SimpleNamespace(train_online=lambda *a, **k: None)
sys.modules["train_rl"] = types.SimpleNamespace(launch=lambda *a, **k: 0)
sys.modules["user_risk_inputs"] = types.SimpleNamespace(
    configure_user_risk=lambda *a, **k: None
)
import importlib.machinery

sys.modules["mlflow"] = types.SimpleNamespace(
    __loader__=True, __spec__=importlib.machinery.ModuleSpec("mlflow", loader=None)
)
sys.modules["prometheus_client"] = types.SimpleNamespace(
    Counter=lambda *a, **k: None, Gauge=lambda *a, **k: None
)
sys.modules["analysis.anomaly_detector"] = types.SimpleNamespace(
    detect_anomalies=lambda df, quarantine_path=None, counter=None: (df, [])
)
sys.modules["analysis.data_quality"] = types.SimpleNamespace(
    check_recency=lambda *a, **k: True,
    apply_quality_checks=lambda df: (df, {}),
)


# Create fake MetaTrader5 module before importing realtime_train


def _fake_copy(symbol, start, n, mode):
    time.sleep(0.1)
    return [{"time": start, "bid": 1.0, "ask": 1.1, "volume": 1}]


def _fake_init():
    return True


fake_mt5 = types.SimpleNamespace(
    COPY_TICKS_ALL=0, copy_ticks_from=_fake_copy, initialize=_fake_init
)
sys.modules["MetaTrader5"] = fake_mt5
conn_mgr_stub = types.SimpleNamespace(_manager=None)


def _conn_init(mods):
    conn_mgr_stub._manager = object()


conn_mgr_stub.init = _conn_init
conn_mgr_stub.get_active_broker = lambda: fake_mt5
conn_mgr_stub.failover = lambda: None
sys.modules["brokers.connection_manager"] = conn_mgr_stub

import realtime_train as rt


class FakeQueue:
    def __init__(self):
        self.published = []

    def publish_dataframe(self, df):
        time.sleep(0.1)
        self.published.extend(df.to_dict("records"))


def fake_make_features(df):
    time.sleep(0.1)
    return df


def test_pipeline_parallel(monkeypatch):
    monkeypatch.setattr(rt, "make_features", fake_make_features)
    queue = FakeQueue()

    async def pipeline(sym: str):
        ticks = await rt.fetch_ticks(sym, 1)
        if ticks.empty:
            return
        ticks["Symbol"] = sym
        feats = await rt.generate_features(ticks)
        await rt.dispatch_signals(queue, feats)

    async def run():
        await asyncio.gather(pipeline("EURUSD"), pipeline("GBPUSD"))

    start = time.perf_counter()
    asyncio.run(run())
    elapsed = time.perf_counter() - start
    assert elapsed < 0.5
    assert len(queue.published) == 2
