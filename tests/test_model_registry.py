from dataclasses import dataclass
import sys
import types
import contextlib
from pathlib import Path

sys.modules["telemetry"] = types.SimpleNamespace(
    get_tracer=lambda name: types.SimpleNamespace(
        start_as_current_span=lambda *a, **k: contextlib.nullcontext()
    ),
    get_meter=lambda name: types.SimpleNamespace(
        create_counter=lambda *a, **k: types.SimpleNamespace(
            add=lambda *a, **k: None
        )
    ),
)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_registry import ModelRegistry, ResourceCapabilities
import model_registry as mr
import time
from analysis.inference_latency import InferenceLatency


@dataclass
class DummyMonitor:
    capabilities: ResourceCapabilities

    def __post_init__(self) -> None:
        self.capability_tier = self.capabilities.capability_tier()

    def start(self) -> None:  # pragma: no cover - no background tasks in tests
        pass


def test_initial_selection() -> None:
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    assert registry.get("sentiment") == "sentiment_large"
    assert registry.get("rl_policy") == "rl_medium"
    assert registry.get("trade_exit") == "exit_transformer"


def test_fallback_on_refresh() -> None:
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    monitor.capabilities = ResourceCapabilities(
        cpus=2, memory_gb=4, has_gpu=False, gpu_count=0
    )
    monitor.capability_tier = monitor.capabilities.capability_tier()
    registry.refresh()
    assert registry.get("sentiment") == "sentiment_small_quantized"
    assert registry.get("rl_policy") == "rl_small_quantized"
    assert registry.get("trade_exit") == "exit_gbm_quantized"


def test_baseline_on_resource_loss_and_recovery(caplog) -> None:
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    monitor.capabilities = ResourceCapabilities(
        cpus=1, memory_gb=1, has_gpu=False, gpu_count=0
    )
    monitor.capability_tier = monitor.capabilities.capability_tier()
    caplog.set_level("WARNING")
    registry.refresh()
    assert registry.get("rl_policy") == "baseline"
    assert registry.get("trade_exit") == "exit_baseline"
    assert any("baseline" in rec.message for rec in caplog.records)
    caplog.clear()
    monitor.capabilities = ResourceCapabilities(
        cpus=8, memory_gb=32, has_gpu=True, gpu_count=1
    )
    monitor.capability_tier = monitor.capabilities.capability_tier()
    caplog.set_level("INFO")
    registry.refresh()
    assert registry.get("rl_policy") == "rl_medium"
    assert registry.get("trade_exit") == "exit_transformer"
    assert any("restored" in rec.message.lower() for rec in caplog.records)


def test_baseline_on_model_crash(caplog) -> None:
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    caplog.set_level("ERROR")
    registry.report_failure("rl_policy")
    assert registry.get("rl_policy") == "baseline"
    assert any("crashed" in rec.message for rec in caplog.records)
    registry.refresh()
    assert registry.get("rl_policy") == "rl_medium"


def test_latency_watchdog_downgrades_and_recovers(monkeypatch, caplog) -> None:
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    registry.latency = InferenceLatency(window=1)
    registry.breach_checks = 1
    registry.recovery_checks = 1
    monkeypatch.setitem(mr.LATENCY_THRESHOLDS, "gpu", 0.01)
    registry._remote_variant = lambda task: None

    def slow_loader(name):
        class SlowModel:
            def predict(self, features):
                time.sleep(0.05)
                return 0

        return SlowModel()

    caplog.set_level("WARNING")
    registry.predict("rl_policy", None, loader=slow_loader)
    assert registry.get("rl_policy") == "rl_small"
    assert any("Latency high" in rec.message for rec in caplog.records)

    def fast_loader(name):
        class FastModel:
            def predict(self, features):
                return 0

        return FastModel()

    caplog.clear()
    caplog.set_level("INFO")
    registry.predict("rl_policy", None, loader=fast_loader)
    assert registry.get("rl_policy") == "rl_medium"
    assert any("Latency normalised" in rec.message for rec in caplog.records)

