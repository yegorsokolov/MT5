import contextlib
import importlib.util
from dataclasses import dataclass
from pathlib import Path
import sys
import time
import types

# Stub telemetry module for lightweight testing
sys.modules["telemetry"] = types.SimpleNamespace(
    get_tracer=lambda name: types.SimpleNamespace(
        start_as_current_span=lambda *a, **k: contextlib.nullcontext()
    ),
    get_meter=lambda name: types.SimpleNamespace(
        create_counter=lambda *a, **k: types.SimpleNamespace(add=lambda *a, **k: None)
    ),
)

# Capture metric names recorded by the registry
_metric_calls = []
metrics_stub = types.SimpleNamespace(
    record_metric=lambda name, value, tags=None: _metric_calls.append(name),
    model_cache_hit=lambda: _metric_calls.append("model_cache_hits"),
    model_unload=lambda: _metric_calls.append("model_unloads"),
)
sys.modules["analytics.metrics_store"] = metrics_stub

# Provide a lightweight joblib stub for model loading
import joblib as _joblib_real

class _StubModel:
    pass

sys.modules["joblib"] = types.SimpleNamespace(
    load=lambda *a, **k: _StubModel(),
    dump=lambda *a, **k: None,
)

# Ensure repository root on path and load model_registry module
sys.path.append(str(Path(__file__).resolve().parents[1]))
spec = importlib.util.spec_from_file_location(
    "model_registry", Path(__file__).resolve().parents[1] / "model_registry.py"
)
mr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mr)
sys.modules["joblib"] = _joblib_real

ModelRegistry = mr.ModelRegistry
ResourceCapabilities = mr.ResourceCapabilities


@dataclass
class DummyMonitor:
    capabilities: ResourceCapabilities

    def __post_init__(self) -> None:
        self.capability_tier = self.capabilities.capability_tier()

    def start(self) -> None:  # pragma: no cover - no background tasks in tests
        pass


def test_ttl_eviction_and_metrics(monkeypatch):
    _metric_calls.clear()
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False, cfg={"model_cache_ttl": 1})

    t0 = time.time()
    monkeypatch.setattr(mr.time, "time", lambda: t0)
    registry._load_model("sentiment_large")
    registry._load_model("sentiment_large")
    assert "model_cache_hits" in _metric_calls

    monkeypatch.setattr(mr.time, "time", lambda: t0 + 2)
    registry._purge_unused()
    assert "sentiment_large" not in registry._models
    assert "model_unloads" in _metric_calls


def test_downgrade_unloads_and_fallback():
    _metric_calls.clear()
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    registry._load_model("sentiment_large")
    assert "sentiment_large" in registry._models

    monitor.capabilities = ResourceCapabilities(cpus=2, memory_gb=4, has_gpu=False, gpu_count=0)
    monitor.capability_tier = monitor.capabilities.capability_tier()
    registry.refresh()
    assert registry.get("sentiment") == "sentiment_small_quantized"
    assert "sentiment_large" not in registry._models
    assert "model_unloads" in _metric_calls
