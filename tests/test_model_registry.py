from dataclasses import dataclass
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from model_registry import ModelRegistry, ResourceCapabilities


@dataclass
class DummyMonitor:
    capabilities: ResourceCapabilities

    def start(self) -> None:  # pragma: no cover - no background tasks in tests
        pass


def test_initial_selection() -> None:
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    assert registry.get("sentiment") == "sentiment_large"
    assert registry.get("rl_policy") == "rl_medium"


def test_fallback_on_refresh() -> None:
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    monitor.capabilities = ResourceCapabilities(
        cpus=2, memory_gb=4, has_gpu=False, gpu_count=0
    )
    registry.refresh()
    assert registry.get("sentiment") == "sentiment_small"
    assert registry.get("rl_policy") == "rl_small"


def test_baseline_on_resource_loss_and_recovery(caplog) -> None:
    monitor = DummyMonitor(
        ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1)
    )
    registry = ModelRegistry(monitor, auto_refresh=False)
    monitor.capabilities = ResourceCapabilities(
        cpus=1, memory_gb=1, has_gpu=False, gpu_count=0
    )
    caplog.set_level("WARNING")
    registry.refresh()
    assert registry.get("rl_policy") == "baseline"
    assert any("baseline" in rec.message for rec in caplog.records)
    caplog.clear()
    monitor.capabilities = ResourceCapabilities(
        cpus=8, memory_gb=32, has_gpu=True, gpu_count=1
    )
    caplog.set_level("INFO")
    registry.refresh()
    assert registry.get("rl_policy") == "rl_medium"
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
