import asyncio
import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub metrics before importing connection manager
class _Gauge:
    def __init__(self):
        self.values = {}
    def labels(self, **labels):
        broker = labels.get("broker")
        def set_value(v):
            self.values[broker] = v
        return types.SimpleNamespace(set=set_value)

class _Counter:
    def __init__(self):
        self.counts = {}
    def labels(self, **labels):
        broker = labels.get("broker")
        def inc():
            self.counts[broker] = self.counts.get(broker, 0) + 1
        return types.SimpleNamespace(inc=inc)

metrics_stub = types.SimpleNamespace(
    RECONNECT_COUNT=types.SimpleNamespace(inc=lambda: None),
    ERROR_COUNT=types.SimpleNamespace(inc=lambda: None),
    TRADE_COUNT=types.SimpleNamespace(inc=lambda: None),
    FEATURE_ANOMALIES=types.SimpleNamespace(inc=lambda: None),
    RESOURCE_RESTARTS=types.SimpleNamespace(inc=lambda: None),
    BROKER_LATENCY_MS=_Gauge(),
    BROKER_FAILURES=_Counter(),
)
sys.modules['metrics'] = metrics_stub

from brokers import connection_manager as cm

class SlowBroker:
    def initialize(self):
        return True
    async def ping(self):
        await asyncio.sleep(0.2)

class FastBroker:
    def initialize(self):
        return True
    async def ping(self):
        return True

import pytest


def test_watchdog_failover():
    async def run():
        cm.init([SlowBroker(), FastBroker()])
        task = asyncio.create_task(
            cm.watchdog(interval=0.05, timeout=0.05, latency_threshold_ms=10, failure_threshold=1)
        )
        await asyncio.sleep(0.15)
        assert isinstance(cm.get_active_broker(), FastBroker)
        task.cancel()
        await task
    asyncio.run(run())
