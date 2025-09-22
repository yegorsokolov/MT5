import contextlib
import gc
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pickle
import psutil

sys.modules.setdefault(
    "telemetry",
    types.SimpleNamespace(
        get_tracer=lambda name: types.SimpleNamespace(
            start_as_current_span=lambda *a, **k: contextlib.nullcontext()
        ),
        get_meter=lambda name: types.SimpleNamespace(
            create_counter=lambda *a, **k: types.SimpleNamespace(
                add=lambda *a, **k: None
            )
        ),
    ),
)

sys.modules.setdefault(
    "joblib",
    types.SimpleNamespace(
        dump=lambda obj, f: open(f, "wb").write(pickle.dumps(obj)),
        load=lambda f: pickle.loads(open(f, "rb").read()),
    ),
)
import joblib  # type: ignore

sys.modules.setdefault(
    "analytics.metrics_store",
    types.SimpleNamespace(record_metric=lambda *a, **k: None),
)

sys.path.append(str(Path(__file__).resolve().parents[1]))
from mt5.model_registry import ModelRegistry, ModelVariant, ResourceCapabilities, MODEL_REGISTRY


@dataclass
class DummyMonitor:
    capabilities: ResourceCapabilities

    def __post_init__(self) -> None:
        self.capability_tier = self.capabilities.capability_tier()

    def start(self) -> None:  # pragma: no cover - no background loop in tests
        pass


class DummyModel:
    def __init__(self, size: int) -> None:
        # allocate ``size`` bytes
        self.data = bytearray(size)

    def predict(self, features):
        return [0] * len(features)


def test_lazy_loading_and_unload(tmp_path, monkeypatch):
    heavy_path = tmp_path / "heavy.pkl"
    light_path = tmp_path / "light.pkl"
    joblib.dump(DummyModel(2_000_000), heavy_path)
    joblib.dump(DummyModel(10), light_path)

    monkeypatch.setitem(
        MODEL_REGISTRY,
        "dummy",
        [
            ModelVariant("heavy", ResourceCapabilities(4, 4, False, gpu_count=0), weights=heavy_path),
            ModelVariant("light", ResourceCapabilities(1, 1, False, gpu_count=0), weights=light_path),
        ],
    )

    monitor = DummyMonitor(ResourceCapabilities(4, 4, False, gpu_count=0))
    registry = ModelRegistry(monitor, auto_refresh=False)

    assert registry._models == {}
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    registry.predict("dummy", [1, 2, 3])
    assert "heavy" in registry._models
    rss_loaded = proc.memory_info().rss
    assert rss_loaded > rss_before

    monitor.capabilities = ResourceCapabilities(1, 1, False, gpu_count=0)
    monitor.capability_tier = monitor.capabilities.capability_tier()
    registry.refresh()
    gc.collect()
    rss_after = proc.memory_info().rss
    assert "heavy" not in registry._models
    assert rss_after <= rss_loaded
