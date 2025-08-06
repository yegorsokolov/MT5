"""Dynamic model selection based on system capabilities."""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

import importlib.util
from pathlib import Path

# ``utils.resource_monitor`` lives inside a package whose ``__init__`` pulls in
# heavy optional dependencies. Import the module directly from its file path to
# avoid importing ``utils`` itself and keep tests lightweight.
_spec = importlib.util.spec_from_file_location(
    "resource_monitor", Path(__file__).with_name("utils") / "resource_monitor.py"
)
_rm = importlib.util.module_from_spec(_spec)
assert _spec and _spec.loader
_spec.loader.exec_module(_rm)  # type: ignore

ResourceCapabilities = _rm.ResourceCapabilities
ResourceMonitor = _rm.ResourceMonitor
monitor = _rm.monitor


@dataclass
class ModelVariant:
    """A specific model implementation and its resource needs."""

    name: str
    requirements: ResourceCapabilities

    def is_supported(self, capabilities: ResourceCapabilities) -> bool:
        """Return True if system capabilities meet this variant's needs."""
        return (
            capabilities.cpus >= self.requirements.cpus
            and capabilities.memory_gb >= self.requirements.memory_gb
            and (not self.requirements.has_gpu or capabilities.has_gpu)
        )


# Registry of task -> model variants ordered from heaviest to lightest
MODEL_REGISTRY: Dict[str, List[ModelVariant]] = {
    "sentiment": [
        ModelVariant("sentiment_large", ResourceCapabilities(8, 32, True)),
        ModelVariant("sentiment_medium", ResourceCapabilities(4, 16, True)),
        ModelVariant("sentiment_small", ResourceCapabilities(2, 4, False)),
    ],
    "rl_policy": [
        ModelVariant("rl_large", ResourceCapabilities(16, 64, True)),
        ModelVariant("rl_medium", ResourceCapabilities(8, 32, True)),
        ModelVariant("rl_small", ResourceCapabilities(2, 4, False)),
    ],
}


class ModelRegistry:
    """Selects appropriate model variants based on available resources."""

    def __init__(self, monitor: ResourceMonitor = monitor, auto_refresh: bool = True) -> None:
        self.monitor = monitor
        self.selected: Dict[str, ModelVariant] = {}
        self._task: Optional[asyncio.Task] = None
        self._pick_models()
        if auto_refresh:
            self.monitor.start()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            self._task = loop.create_task(self._watch())

    def _pick_models(self) -> None:
        caps = self.monitor.capabilities
        for task, variants in MODEL_REGISTRY.items():
            for variant in variants:
                if variant.is_supported(caps):
                    self.selected[task] = variant
                    break
            else:
                self.selected[task] = variants[-1]

    async def _watch(self) -> None:
        """Periodically re-evaluate models in case capabilities change."""
        prev_caps = self.monitor.capabilities
        while True:
            await asyncio.sleep(24 * 60 * 60)
            caps = self.monitor.capabilities
            if caps != prev_caps:
                prev_caps = caps
                self._pick_models()

    def get(self, task: str) -> str:
        """Return the chosen model variant for the given task."""
        return self.selected[task].name

    def refresh(self) -> None:
        """Manually re-run model selection using current capabilities."""
        self._pick_models()
