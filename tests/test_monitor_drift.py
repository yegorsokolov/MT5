import asyncio
import sys
from pathlib import Path

import pandas as pd

import importlib.util


root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

spec = importlib.util.spec_from_file_location(
    "monitor_drift", root / "monitor_drift.py"
)
monitor_drift = importlib.util.module_from_spec(spec)
sys.modules["monitor_drift"] = monitor_drift
spec.loader.exec_module(monitor_drift)


def test_drift_detection(tmp_path, monkeypatch, caplog):
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


def test_monitor_lifecycle_cancels_task():
    async def runner():
        task = monitor_drift.start_monitoring()
        await asyncio.sleep(0)
        assert task is not None and not task.done()
        await monitor_drift.stop_monitoring()
        assert task.cancelled()
        assert monitor_drift.monitor._task is None

    asyncio.run(runner())
