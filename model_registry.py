"""Dynamic model selection based on system capabilities."""

import asyncio
import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

# ``models.mixture_of_experts`` depends on no heavy libraries but importing the
# ``models`` package would trigger optional imports like ``torch``. Import the
# module directly from its file path to keep tests lightweight.
_moe_spec = importlib.util.spec_from_file_location(
    "mixture_of_experts", Path(__file__).with_name("models") / "mixture_of_experts.py"
)
_moe = importlib.util.module_from_spec(_moe_spec)
assert _moe_spec and _moe_spec.loader
sys.modules["mixture_of_experts"] = _moe
_moe_spec.loader.exec_module(_moe)  # type: ignore

ExpertSpec = _moe.ExpertSpec
GatingNetwork = _moe.GatingNetwork
MacroExpert = _moe.MacroExpert
MeanReversionExpert = _moe.MeanReversionExpert
TrendExpert = _moe.TrendExpert

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

from telemetry import get_tracer, get_meter

tracer = get_tracer(__name__)
meter = get_meter(__name__)
_refresh_counter = meter.create_counter(
    "model_refreshes", description="Number of model refresh operations"
)

TIERS = {"lite": 0, "standard": 1, "gpu": 2, "hpc": 3}


@dataclass
class ModelVariant:
    """A specific model implementation and its resource needs."""

    name: str
    requirements: ResourceCapabilities
    quantized: Optional[str] = None
    remote_only: bool = False

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
            "sentiment_large",
            ResourceCapabilities(8, 32, True, gpu_count=1),
            "sentiment_large_quantized",
            remote_only=True,
        ),
        ModelVariant(
            "sentiment_medium",
            ResourceCapabilities(4, 16, True, gpu_count=1),
            "sentiment_medium_quantized",
        ),
        ModelVariant(
            "sentiment_small",
            ResourceCapabilities(2, 4, False, gpu_count=0),
            "sentiment_small_quantized",
        ),
    ],
    "rl_policy": [
        ModelVariant(
            "rl_large",
            ResourceCapabilities(16, 64, True, gpu_count=1),
            "rl_large_quantized",
            remote_only=True,
        ),
        ModelVariant(
            "rl_medium",
            ResourceCapabilities(8, 32, True, gpu_count=1),
            "rl_medium_quantized",
        ),
        ModelVariant(
            "rl_small",
            ResourceCapabilities(2, 4, False, gpu_count=0),
            "rl_small_quantized",
        ),
        ModelVariant("baseline", ResourceCapabilities(1, 1, False, gpu_count=0)),
    ],
}


class ModelRegistry:
    """Selects appropriate model variants based on available resources."""

    def __init__(self, monitor: ResourceMonitor = monitor, auto_refresh: bool = True) -> None:
        self.monitor = monitor
        self.selected: Dict[str, ModelVariant] = {}
        # Keep track of previous variants so we can rollback if a canary fails
        self._previous: Dict[str, ModelVariant] = {}
        self._task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
        self.moe = GatingNetwork(
            [
                ExpertSpec(TrendExpert(), ResourceCapabilities(2, 4, False, gpu_count=0)),
                ExpertSpec(
                    MeanReversionExpert(), ResourceCapabilities(1, 2, False, gpu_count=0)
                ),
                ExpertSpec(MacroExpert(), ResourceCapabilities(1, 1, False, gpu_count=0)),
            ]
        )
        self._pick_models()
        if auto_refresh:
            self.monitor.start()
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = asyncio.get_event_loop()
            queue = self.monitor.subscribe()
            self._task = loop.create_task(self._watch(queue))

    def _pick_models(self) -> None:
        with tracer.start_as_current_span("pick_models"):
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
            _refresh_counter.add(1)

    async def _watch(self, queue: asyncio.Queue[str]) -> None:
        """Re-evaluate models when capability tier increases."""
        prev = self.monitor.capability_tier
        while True:
            tier = await queue.get()
            if TIERS.get(tier, 0) > TIERS.get(prev, 0):
                self.logger.info("Capability tier upgraded to %s; re-evaluating models", tier)
                self._pick_models()
            prev = tier

    def get(self, task: str) -> str:
        """Return the chosen model variant for the given task."""
        variant = self.selected[task]
        tier = self.monitor.capability_tier
        if TIERS.get(tier, 0) <= TIERS["lite"] and variant.quantized:
            return variant.quantized
        return variant.name

    def refresh(self) -> None:
        """Manually re-run model selection using current capabilities."""
        with tracer.start_as_current_span("refresh_models"):
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

    def requires_remote(self, task: str) -> bool:
        """Return True if ``task`` should be executed remotely."""
        return self._remote_variant(task) is not None

    # Internal ------------------------------------------------------------
    def _remote_variant(self, task: str) -> Optional[ModelVariant]:
        variants = MODEL_REGISTRY.get(task)
        if not variants:
            return None
        top = variants[0]
        if top.remote_only and not top.is_supported(self.monitor.capabilities):
            return top
        return None

    def predict_mixture(self, history: Any, regime: float) -> float:
        """Predict using the mixture-of-experts gating network."""
        with tracer.start_as_current_span("predict_mixture"):
            return self.moe.predict(history, regime, self.monitor.capabilities)

    def predict(self, task: str, features: Any, loader) -> Any:
        """Return predictions for ``task`` using local or remote models.

        Parameters
        ----------
        task:
            Name of the task as defined in :data:`MODEL_REGISTRY`.
        features:
            Feature matrix passed to the model's ``predict`` or ``predict_proba``.
        loader:
            Callable taking a model name and returning the local model instance.
        """
        model_name = self.get(task)
        with tracer.start_as_current_span("predict"):
            remote_variant = self._remote_variant(task)
            if remote_variant is not None:
                import importlib.util

                spec = importlib.util.spec_from_file_location(
                    "inference_client", Path(__file__).with_name("models") / "inference_client.py"
                )
                ic = importlib.util.module_from_spec(spec)
                assert spec and spec.loader
                spec.loader.exec_module(ic)  # type: ignore
                client = ic.InferenceClient()
                return client.predict(remote_variant.name, features)
            model = loader(model_name)
            if hasattr(model, "predict_proba"):
                return model.predict_proba(features)
            return model.predict(features)

    # ------------------------------------------------------------------
    def promote(self, task: str, model_name: str) -> None:
        """Promote a candidate model to production for ``task``."""

        prev = self.selected.get(task)
        if prev:
            self._previous[task] = prev
            requirements = prev.requirements
        else:
            requirements = ResourceCapabilities(0, 0, False, gpu_count=0)
        self.selected[task] = ModelVariant(model_name, requirements)

    # ------------------------------------------------------------------
    def rollback(self, task: str) -> None:
        """Rollback to the previously selected model for ``task``."""

        prev = self._previous.get(task)
        if prev:
            self.logger.info("Reverting to previous model %s for %s", prev.name, task)
            self.selected[task] = prev
            del self._previous[task]
