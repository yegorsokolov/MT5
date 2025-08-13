"""Dynamic model selection based on system capabilities."""

import asyncio
import importlib.util
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

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
        ModelVariant(
            "sentiment_large", ResourceCapabilities(8, 32, True, gpu_count=1)
        ),
        ModelVariant(
            "sentiment_medium", ResourceCapabilities(4, 16, True, gpu_count=1)
        ),
        ModelVariant(
            "sentiment_small", ResourceCapabilities(2, 4, False, gpu_count=0)
        ),
    ],
    "rl_policy": [
        ModelVariant("rl_large", ResourceCapabilities(16, 64, True, gpu_count=1)),
        ModelVariant("rl_medium", ResourceCapabilities(8, 32, True, gpu_count=1)),
        ModelVariant("rl_small", ResourceCapabilities(2, 4, False, gpu_count=0)),
        ModelVariant("baseline", ResourceCapabilities(1, 1, False, gpu_count=0)),
    ],
}


class ModelRegistry:
    """Selects appropriate model variants based on available resources."""

    def __init__(self, monitor: ResourceMonitor = monitor, auto_refresh: bool = True) -> None:
        self.monitor = monitor
        self.selected: Dict[str, ModelVariant] = {}
        self._task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
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
            baseline = variants[-1]
            chosen = baseline
            for variant in variants:
                if variant.is_supported(caps):
                    chosen = variant
                    break
            prev = self.selected.get(task)
            if prev != chosen:
                if chosen is baseline:
                    self.logger.warning("Falling back to baseline for %s", task)
                elif prev and prev.name == baseline.name:
                    self.logger.info("Restored %s for %s", chosen.name, task)
            self.selected[task] = chosen

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

    def report_failure(self, task: str) -> None:
        """Report that the active model for ``task`` has crashed.

        Switch to the baseline variant and log the event.
        """

        variants = MODEL_REGISTRY.get(task)
        if not variants:
            return
        baseline = variants[-1]
        prev = self.selected.get(task)
        if prev != baseline:
            self.logger.error(
                "Model %s for %s crashed; using baseline", prev.name if prev else "unknown", task
            )
            self.selected[task] = baseline
