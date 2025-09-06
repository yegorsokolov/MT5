import time
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.worker_manager import WorkerManager

import types, importlib.util

class _Resp:
    def __init__(self, preds=None):
        self._preds = preds or []

    def raise_for_status(self):
        return None

    def json(self):
        return {"predictions": self._preds}


def _post(url, json, timeout):
    # Echo back one prediction per feature to simulate server behaviour
    feats = json["features"]
    return _Resp([0.0] * len(feats))


requests_stub = types.SimpleNamespace(post=_post)
sys.modules.setdefault("requests", requests_stub)

_remote_spec = importlib.util.spec_from_file_location(
    "remote_client", Path(__file__).resolve().parents[1] / "models" / "remote_client.py"
)
remote_client = importlib.util.module_from_spec(_remote_spec)
assert _remote_spec and _remote_spec.loader
_remote_spec.loader.exec_module(remote_client)  # type: ignore[arg-type]
predict_remote = remote_client.predict_remote


def test_worker_manager_scales_on_load(monkeypatch):
    # Capture metrics instead of writing to disk
    metrics: list[tuple[str, float, dict | None]] = []

    def fake_record(name, value, tags=None, path=None):
        metrics.append((name, value, tags))

    monkeypatch.setattr(
        "services.worker_manager.record_metric", fake_record
    )

    # Speed up scaling for the test
    manager = WorkerManager(window=2.0, high_rps=1.0, low_rps=0.1)
    monkeypatch.setattr("services.worker_manager._manager", manager, raising=False)

    # Generate a burst of requests from remote_client and simulated feature_store
    for _ in range(3):
        predict_remote("m", pd.DataFrame({"a": [1, 2, 3, 4]}), batch_size=2)
        manager.record_request("feature_store", 0.01, batch_size=1)

    assert manager.worker_count > 1

    # After quiet period the manager should scale back down
    time.sleep(2.1)
    for _ in range(5):
        manager._scale()
    assert manager.worker_count == 1

    # Metrics should have been recorded
    assert any(m[0] == "worker_count" for m in metrics)
    assert any(
        m[0] == "queue_latency" and m[2]["source"] == "remote_client" for m in metrics
    )
    assert any(
        m[0] == "queue_latency" and m[2]["source"] == "feature_store" for m in metrics
    )
    # Batching metrics should also be recorded
    assert any(
        m[0] == "batch_throughput" and m[2]["source"] == "remote_client"
        for m in metrics
    )
