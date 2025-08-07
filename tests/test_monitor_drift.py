import asyncio
import sys
import types

import pandas as pd

class _Metric:
    def __init__(self):
        self.val = 0

    def inc(self, amount: int = 1):
        self.val += amount

    def set(self, value: int):
        self.val = value

prom_stub = types.SimpleNamespace(Counter=lambda *a, **k: _Metric(), Gauge=lambda *a, **k: _Metric())
sys.modules.setdefault("prometheus_client", prom_stub)

import importlib.util
from pathlib import Path

metric_spec = importlib.util.spec_from_file_location(
    "metrics", Path(__file__).resolve().parents[1] / "metrics.py"
)
metrics = importlib.util.module_from_spec(metric_spec)
sys.modules["metrics"] = metrics
metric_spec.loader.exec_module(metrics)

spec = importlib.util.spec_from_file_location(
    "monitor_drift", Path(__file__).resolve().parents[1] / "monitor_drift.py"
)
monitor_drift = importlib.util.module_from_spec(spec)
sys.modules["monitor_drift"] = monitor_drift
spec.loader.exec_module(monitor_drift)


def test_drift_detection(tmp_path, monkeypatch, caplog):
    # cancel global task from module import to avoid side effects
    if monitor_drift.monitor._task is not None:
        monitor_drift.monitor._task.cancel()
        asyncio.get_event_loop().run_until_complete(asyncio.sleep(0))

    baseline = pd.DataFrame({"f": [0] * 100, "prediction": [0] * 100})
    baseline_path = tmp_path / "baseline.parquet"
    baseline.to_parquet(baseline_path, index=False)

    dm = monitor_drift.DriftMonitor(
        baseline_path=baseline_path, store_path=tmp_path / "curr.parquet", threshold=0.05
    )

    features = pd.DataFrame({"f": [1] * 100})
    preds = pd.Series([1] * 100)
    dm.record(features, preds)

    counter = _Metric()
    monkeypatch.setattr(monitor_drift.metrics, "DRIFT_EVENTS", counter, raising=False)

    with caplog.at_level("WARNING"):
        dm.compare()

    assert counter.val == 2
    assert "Data drift detected" in caplog.text
