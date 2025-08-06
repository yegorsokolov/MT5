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
    monitor = DummyMonitor(ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True))
    registry = ModelRegistry(monitor, auto_refresh=False)
    assert registry.get("sentiment") == "sentiment_large"
    assert registry.get("rl_policy") == "rl_medium"


def test_fallback_on_refresh() -> None:
    monitor = DummyMonitor(ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True))
    registry = ModelRegistry(monitor, auto_refresh=False)
    monitor.capabilities = ResourceCapabilities(cpus=2, memory_gb=4, has_gpu=False)
    registry.refresh()
    assert registry.get("sentiment") == "sentiment_small"
    assert registry.get("rl_policy") == "rl_small"
