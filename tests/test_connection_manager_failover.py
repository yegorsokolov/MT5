import asyncio
import sys
import types
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# stub heavy dependencies so realtime_train imports without installing them
sys.modules['git'] = types.SimpleNamespace(Repo=lambda *a, **k: None)
sys.modules['yaml'] = types.SimpleNamespace(safe_load=lambda *a, **k: {})
utils_mod = types.ModuleType('utils')
utils_mod.load_config = lambda: {}
sys.modules['utils'] = utils_mod
sys.modules['utils.resource_monitor'] = types.SimpleNamespace(monitor=lambda *a, **k: None)
sys.modules['utils.graceful_exit'] = types.SimpleNamespace(graceful_exit=lambda *a, **k: None)
sys.modules['requests'] = types.SimpleNamespace(get=lambda *a, **k: None)
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
data_pkg.feature_scaler = types.SimpleNamespace(
    FeatureScaler=types.SimpleNamespace(
        load=lambda *a, **k: types.SimpleNamespace(
            partial_fit=lambda *a, **k: None,
            transform=lambda x: x,
            save=lambda *a, **k: None,
        )
    )
)
sys.modules['data'] = data_pkg
sys.modules['data.features'] = data_pkg.features
sys.modules['data.sanitize'] = data_pkg.sanitize
sys.modules['data.trade_log'] = data_pkg.trade_log
sys.modules['data.feature_scaler'] = data_pkg.feature_scaler
sys.modules['signal_queue'] = types.SimpleNamespace(
    get_signal_backend=lambda cfg: None,
    publish_dataframe_async=lambda *a, **k: None,
)
sys.modules['metrics'] = types.SimpleNamespace(
    RECONNECT_COUNT=types.SimpleNamespace(inc=lambda: None),
    ERROR_COUNT=types.SimpleNamespace(inc=lambda: None),
    TRADE_COUNT=types.SimpleNamespace(inc=lambda: None),
    FEATURE_ANOMALIES=types.SimpleNamespace(inc=lambda: None),
    RESOURCE_RESTARTS=types.SimpleNamespace(inc=lambda: None),
    QUEUE_DEPTH=types.SimpleNamespace(set=lambda v: None),
    BATCH_LATENCY=types.SimpleNamespace(set=lambda v: None),
    BROKER_LATENCY_MS=types.SimpleNamespace(labels=lambda **k: types.SimpleNamespace(set=lambda v: None)),
    BROKER_FAILURES=types.SimpleNamespace(labels=lambda **k: types.SimpleNamespace(inc=lambda: None)),
    SLIPPAGE_BPS=types.SimpleNamespace(set=lambda v: None),
    REALIZED_SLIPPAGE_BPS=types.SimpleNamespace(set=lambda v: None),
    PIPELINE_ANOMALY_TOTAL=types.SimpleNamespace(inc=lambda *a, **k: None),
    PIPELINE_ANOMALY_RATE=types.SimpleNamespace(set=lambda v: None),
)
sys.modules['models'] = types.SimpleNamespace(model_store=types.SimpleNamespace(load_model=lambda *a, **k: (None, None)))
sys.modules['prometheus_client'] = types.SimpleNamespace(Counter=lambda *a, **k: None, Gauge=lambda *a, **k: None)
sys.modules['analysis.anomaly_detector'] = types.SimpleNamespace(
    detect_anomalies=lambda df, quarantine_path=None, counter=None: (df, [])
)
analytics_pkg = types.ModuleType('analytics')
analytics_pkg.metrics_store = types.SimpleNamespace(
    record_metric=lambda *a, **k: None,
    TS_PATH=Path('metrics.parquet'),
)
sys.modules['analytics'] = analytics_pkg
sys.modules['analytics.metrics_store'] = analytics_pkg.metrics_store
sys.modules['user_risk_inputs'] = types.SimpleNamespace(configure_user_risk=lambda *a, **k: None)
sys.modules['state_manager'] = types.SimpleNamespace()
sys.modules['crypto_utils'] = types.SimpleNamespace(
    _load_key=lambda *a, **k: None,
    encrypt=lambda *a, **k: None,
    decrypt=lambda *a, **k: None,
)


class _DummySpan:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        pass


class _DummyTracer:
    def start_as_current_span(self, name):
        return _DummySpan()


class _DummyCounter:
    def add(self, value):
        pass


class _DummyHistogram:
    def record(self, value):
        pass


class _DummyMeter:
    def create_counter(self, *a, **k):
        return _DummyCounter()

    def create_histogram(self, *a, **k):
        return _DummyHistogram()


sys.modules['telemetry'] = types.SimpleNamespace(
    get_tracer=lambda name=None: _DummyTracer(),
    get_meter=lambda name=None: _DummyMeter(),
)

# Stub MetaTrader5 so realtime_train imports cleanly
sys.modules['MetaTrader5'] = types.SimpleNamespace(
    initialize=lambda: True,
    COPY_TICKS_ALL=0,
    ORDER_TYPE_BUY=0,
    ORDER_TYPE_SELL=1,
    TRADE_ACTION_DEAL=0,
    symbol_select=lambda *a, **k: True,
    symbol_info_tick=lambda s: types.SimpleNamespace(bid=1.0, ask=1.1),
    copy_ticks_from=lambda *a, **k: [],
    positions_get=lambda **k: [],
    order_send=lambda *a, **k: True,
)

from brokers import connection_manager as cm
import realtime_train as rt


class FailingBroker:
    COPY_TICKS_ALL = 0

    def initialize(self):
        return True

    def copy_ticks_from(self, *args):
        return None


class WorkingBroker:
    COPY_TICKS_ALL = 0

    def initialize(self):
        return True

    def copy_ticks_from(self, *args):
        return [{"time": args[1], "bid": 1.0, "ask": 1.1, "volume": 1}]


def test_failover(monkeypatch, caplog):
    caplog.set_level(logging.WARNING)
    cm.init([FailingBroker(), WorkingBroker()])
    df = asyncio.run(rt.fetch_ticks("EURUSD", 1))
    assert not df.empty
    assert isinstance(cm.get_active_broker(), WorkingBroker)
    assert any("Failover" in r.message for r in caplog.records)
    sys.modules.pop('MetaTrader5', None)
