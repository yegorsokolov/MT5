import contextlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

from fastapi.testclient import TestClient
import requests

# Add repository root to path and stub telemetry
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
from prediction_cache import PredictionCache
from services.inference_server import app
import metrics

# Ensure prediction cache metrics are available without requiring the full
# prometheus client implementation.
metrics.PRED_CACHE_HIT = types.SimpleNamespace(inc=lambda: None)


@dataclass
class DummyMonitor:
    capabilities: ResourceCapabilities

    def __post_init__(self) -> None:
        self.capability_tier = self.capabilities.capability_tier()

    def start(self) -> None:  # pragma: no cover - no background tasks
        pass


class DummyModel:
    def predict(self, df):
        return [0.9] * len(df)


def test_remote_fallback_and_cache(monkeypatch):
    server_model = DummyModel()
    monkeypatch.setattr(
        "services.inference_server._load_model", lambda name: server_model
    )
    test_client = TestClient(app)

    calls = {"post": 0}

    def fake_post(url, json, timeout):
        calls["post"] += 1
        resp = test_client.post("/predict", json=json)
        return types.SimpleNamespace(
            status_code=resp.status_code,
            json=resp.json,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(
        "services.worker_manager.get_worker_manager",
        lambda: types.SimpleNamespace(record_request=lambda *a, **k: None),
    )

    monitor = DummyMonitor(
        ResourceCapabilities(cpus=1, memory_gb=1, has_gpu=False, gpu_count=0)
    )
    cache = PredictionCache(maxsize=8)
    registry = ModelRegistry(monitor, auto_refresh=False, cache=cache)

    feats = [{"x": 1}]
    preds1 = registry.predict("sentiment", feats)
    key = registry._feature_hash(feats)
    # cache populated after first prediction
    assert key in registry.cache._data
    preds2 = registry.predict("sentiment", feats)
    assert preds1 == preds2 == [0.9]


class LocalModel:
    def predict(self, df):
        return [0.1] * len(df)


def test_local_path_no_remote_call(monkeypatch):
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=32, memory_gb=128, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False, cache=PredictionCache())

    calls = {"post": 0}

    def failing_post(url, json, timeout):
        calls["post"] += 1
        raise AssertionError("remote client should not be used")

    monkeypatch.setattr(requests, "post", failing_post)

    feats = [{"x": 1}]
    preds = registry.predict("sentiment", feats, loader=lambda name: LocalModel())
    assert preds == [0.1]
    assert calls["post"] == 0


def test_batched_remote_vs_local(monkeypatch):
    server_model = DummyModel()
    monkeypatch.setattr(
        "services.inference_server._load_model", lambda name: server_model
    )
    test_client = TestClient(app)
    calls = {"post": 0}

    def fake_post(url, json, timeout):
        calls["post"] += 1
        resp = test_client.post("/predict", json=json)
        return types.SimpleNamespace(
            status_code=resp.status_code,
            json=resp.json,
            raise_for_status=lambda: None,
        )

    monkeypatch.setattr(requests, "post", fake_post)
    monkeypatch.setattr(
        "services.worker_manager.get_worker_manager",
        lambda: types.SimpleNamespace(record_request=lambda *a, **k: None),
    )

    monitor = DummyMonitor(
        ResourceCapabilities(cpus=1, memory_gb=1, has_gpu=False, gpu_count=0)
    )
    registry = ModelRegistry(monitor, auto_refresh=False, cache=PredictionCache(maxsize=8))

    feats = [{"x": i} for i in range(5)]
    preds = registry.predict("sentiment", feats, batch_size=2)
    assert preds == [0.9] * 5
    # 5 features with batch_size=2 -> 3 remote calls
    assert calls["post"] == 3

    # Local model should also support batching
    monitor_local = DummyMonitor(
        ResourceCapabilities(cpus=32, memory_gb=128, has_gpu=True, gpu_count=1)
    )
    registry_local = ModelRegistry(monitor_local, auto_refresh=False, cache=PredictionCache())
    preds_local = registry_local.predict(
        "sentiment", feats, loader=lambda name: LocalModel(), batch_size=2
    )
    assert preds_local == [0.1] * 5
