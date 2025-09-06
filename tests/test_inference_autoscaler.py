import importlib.util
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from services.inference_autoscaler import InferenceAutoscaler


def _load_remote_server():
    spec = importlib.util.spec_from_file_location(
        "remote_server", Path(__file__).resolve().parents[1] / "models" / "remote_server.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


def test_autoscaler_scales_on_load(monkeypatch):
    remote_server = _load_remote_server()

    # Capture metrics instead of writing to disk
    metrics: list[tuple[str, float, dict | None]] = []

    def fake_record(name, value, tags=None, path=None):
        metrics.append((name, value, tags))

    monkeypatch.setattr("services.inference_autoscaler.record_metric", fake_record)
    monkeypatch.setattr("services.inference_autoscaler.remote_server", remote_server)

    scaler = InferenceAutoscaler(
        window=1.0, high_rps=2.0, low_rps=0.5, high_gpu=0.8, low_gpu=0.2
    )

    # Simulate burst of requests and high GPU utilisation
    for _ in range(3):
        remote_server.record_request()
    monkeypatch.setattr(remote_server, "get_gpu_utilization", lambda: 0.9)

    scaler.check()
    assert scaler.worker_count > 1
    assert len(scaler.registry()) == scaler.worker_count

    # Quiet period with low GPU utilisation
    remote_server._REQUEST_TIMES.clear()
    monkeypatch.setattr(remote_server, "get_gpu_utilization", lambda: 0.1)

    scaler.check()
    assert scaler.worker_count == scaler.min_workers

    # Metrics for scaling events should be recorded
    assert any(m[0] == "autoscale_up" for m in metrics)
    assert any(m[0] == "autoscale_down" for m in metrics)

