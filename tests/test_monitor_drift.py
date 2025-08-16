import asyncio
import asyncio
import sys
import pandas as pd

import importlib.util
from pathlib import Path


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

    calls = []
    def fake_record(name, value, tags=None):
        calls.append((name, value))
    monkeypatch.setattr(monitor_drift, "record_metric", fake_record)

    with caplog.at_level("WARNING"):
        dm.compare()

    assert sum(v for n, v in calls if n == "drift_events") == 2
    assert "Data drift detected" in caplog.text
