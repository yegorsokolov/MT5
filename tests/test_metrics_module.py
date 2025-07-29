from prometheus_client import Counter, Gauge, CollectorRegistry
import metrics


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


