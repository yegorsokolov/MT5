import types
from pathlib import Path

from fastapi.testclient import TestClient

import sys
import contextlib

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules["telemetry"] = types.SimpleNamespace(
    get_tracer=lambda name: types.SimpleNamespace(
        start_as_current_span=lambda *a, **k: contextlib.nullcontext()
    ),
    get_meter=lambda name: types.SimpleNamespace(
        create_counter=lambda *a, **k: types.SimpleNamespace(add=lambda *a, **k: None)
    ),
)

from model_registry import ModelRegistry, ResourceCapabilities
from services.inference_server import app
import importlib.util
import requests

# Import inference_client without triggering heavy imports from ``models``
ic_spec = importlib.util.spec_from_file_location(
    "inference_client", Path(__file__).resolve().parents[1] / "models" / "inference_client.py"
)
ic = importlib.util.module_from_spec(ic_spec)
assert ic_spec.loader
ic_spec.loader.exec_module(ic)  # type: ignore


class DummyMonitor:
    def __init__(self, caps: ResourceCapabilities):
        self.capabilities = caps
        self.capability_tier = caps.capability_tier()

    def start(self) -> None:  # pragma: no cover - no background tasks
        pass


class DummyModel:
    def predict(self, df):
        return [0.42] * len(df)


def test_remote_prediction_path(monkeypatch):
    server_model = DummyModel()
    monkeypatch.setattr(
        "services.inference_server._load_model", lambda name: server_model
    )
    test_client = TestClient(app)
    
    def fake_post(url, json, timeout):
        resp = test_client.post("/predict", json=json)
        return types.SimpleNamespace(
            status_code=resp.status_code,
            json=resp.json,
            raise_for_status=lambda: None,
        )

    def fake_get(url, timeout):
        resp = test_client.get("/health")
        return types.SimpleNamespace(
            status_code=resp.status_code,
            json=resp.json,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(ic.requests, "post", fake_post)
    monkeypatch.setattr(ic.requests, "get", fake_get)
    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(
        "services.worker_manager.get_worker_manager",
        lambda: types.SimpleNamespace(record_request=lambda *a, **k: None),
    )
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=1, memory_gb=1, has_gpu=False, gpu_count=0)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    preds = registry.predict("sentiment", [{"x": 1}], loader=lambda name: server_model)
    assert preds == [0.42]
    assert registry.requires_remote("sentiment") is True


def test_local_prediction_path():
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=32, memory_gb=128, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)

    class LocalModel:
        def predict(self, df):
            return [1.0] * len(df)

    preds = registry.predict("sentiment", [{"x": 1}], loader=lambda name: LocalModel())
    assert preds == [1.0]
    assert registry.requires_remote("sentiment") is False


def test_inference_client_retries(monkeypatch):
    calls = []

    def failing_post(url, json, timeout):
        calls.append(1)
        if len(calls) < 2:
            raise ic.requests.RequestException("boom")
        return types.SimpleNamespace(
            json=lambda: {"predictions": [0.1]},
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(ic.requests, "post", failing_post)
    client = ic.InferenceClient(base_url="http://test", retries=2, backoff=0)
    preds = client.predict("m", [{"x": 1}])
    assert preds == [0.1]
    assert len(calls) == 2
