import importlib
import sys
import types
from pathlib import Path

import pytest

try:
    from prometheus_client import Counter, Gauge, CollectorRegistry
except ModuleNotFoundError:
    class _Value:
        def __init__(self) -> None:
            self._total = 0.0

        def inc(self, amount: float = 1.0) -> None:
            self._total += amount

        def set(self, value: float) -> None:
            self._total = value

        def get(self) -> float:
            return self._total

    class Counter:
        def __init__(self, *args, **kwargs) -> None:
            self._value = _Value()

        def inc(self, amount: float = 1.0) -> None:
            self._value.inc(amount)

    class Gauge:
        def __init__(self, *args, **kwargs) -> None:
            self._value = _Value()

        def set(self, value: float) -> None:
            self._value.set(value)

        def inc(self, amount: float = 1.0) -> None:
            self._value.inc(amount)

    class CollectorRegistry:
        def __init__(self, *args, **kwargs) -> None:
            return

    prometheus_stub = types.SimpleNamespace(
        Counter=Counter,
        Gauge=Gauge,
        CollectorRegistry=CollectorRegistry,
    )
    sys.modules.setdefault("prometheus_client", prometheus_stub)
    from prometheus_client import Counter, Gauge, CollectorRegistry

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mt5 import metrics


def test_metrics_counters(monkeypatch):
    reg = CollectorRegistry()
    q = Gauge("q_test", "queue", registry=reg)
    t = Counter("t_test", "trade", registry=reg)
    e = Counter("e_test", "error", registry=reg)
    monkeypatch.setattr(metrics, "QUEUE_DEPTH", q)
    monkeypatch.setattr(metrics, "TRADE_COUNT", t)
    monkeypatch.setattr(metrics, "ERROR_COUNT", e)

    metrics.QUEUE_DEPTH.set(3)
    metrics.TRADE_COUNT.inc()
    metrics.ERROR_COUNT.inc(2)

    assert metrics.QUEUE_DEPTH._value.get() == 3
    assert metrics.TRADE_COUNT._value.get() == 1
    assert metrics.ERROR_COUNT._value.get() == 2


def test_telemetry_uninitialised_by_default():
    module = importlib.import_module("telemetry")
    telemetry = importlib.reload(module)
    try:
        status = telemetry.telemetry_status()
        assert status["initialized"] is False
    finally:
        importlib.reload(module)

