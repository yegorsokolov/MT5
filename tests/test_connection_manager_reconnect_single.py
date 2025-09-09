import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import pandas as pd  # ensure real DataFrame for query_metrics stub

# Stub metrics and analytics before importing connection manager
class _Gauge:
    def labels(self, **labels):
        return types.SimpleNamespace(set=lambda v: None)

class _Counter:
    def labels(self, **labels):
        return types.SimpleNamespace(inc=lambda: None)

sys.modules['metrics'] = types.SimpleNamespace(
    BROKER_FAILURES=_Counter(),
    BROKER_LATENCY_MS=_Gauge(),
)

# Stub analytics metrics_store used by the connection manager
analytics_pkg = types.ModuleType('analytics')
analytics_pkg.metrics_store = types.SimpleNamespace(
    query_metrics=lambda *a, **k: pd.DataFrame()
)
sys.modules['analytics'] = analytics_pkg
sys.modules['analytics.metrics_store'] = analytics_pkg.metrics_store

from brokers import connection_manager as cm


class DummyBroker:
    def __init__(self):
        self.calls = 0

    def initialize(self):
        self.calls += 1
        return True


def test_reconnect_single_broker():
    broker = DummyBroker()
    cm.init([broker])
    assert broker.calls == 1
    cm.failover()
    assert broker.calls == 2
