import logging
import random
from collections import deque, defaultdict
from typing import Any, Callable, Deque, Dict, Tuple

try:  # pragma: no cover - optional dependency
    from analytics.metrics_store import record_metric
except Exception:  # pragma: no cover - fallback when pandas is missing
    def record_metric(*args: Any, **kwargs: Any) -> None:  # type: ignore
        pass
from mt5.model_registry import ModelRegistry


class CanaryManager:
    """Route a fraction of traffic to candidate models and track performance.

    Parameters
    ----------
    registry:
        :class:`~model_registry.ModelRegistry` instance controlling active
        model variants.
    fraction:
        Fraction of calls routed to the canary candidate.  Defaults to ``0.05``
        (5%).
    window:
        Number of recent observations used when comparing performance between
        canary and production models.
    """

    def __init__(self, registry: ModelRegistry, *, fraction: float = 0.05, window: int = 100) -> None:
        self.registry = registry
        self.fraction = fraction
        self.window = window
        self.logger = logging.getLogger(__name__)
        self._candidates: Dict[str, str] = {}
        self._metrics: Dict[str, Dict[str, Deque[float]]] = defaultdict(
            lambda: {"canary": deque(maxlen=window), "production": deque(maxlen=window)}
        )

    # ------------------------------------------------------------------
    def register(self, task: str, model_name: str) -> None:
        """Register a candidate model for ``task``."""

        self.logger.info("Registered canary %s for %s", model_name, task)
        self._candidates[task] = model_name

    # ------------------------------------------------------------------
    def _choose(self, task: str) -> Tuple[str, bool]:
        """Return model name and flag indicating if it is the canary."""

        candidate = self._candidates.get(task)
        if candidate and random.random() < self.fraction:
            return candidate, True
        return self.registry.get(task), False

    # ------------------------------------------------------------------
    def predict(self, task: str, features: Any, loader: Callable[[str], Any]) -> Tuple[Any, bool]:
        """Route prediction requests to production or canary models."""

        model_name, is_canary = self._choose(task)
        model = loader(model_name)
        if hasattr(model, "predict_proba"):
            return model.predict_proba(features), is_canary
        return model.predict(features), is_canary

    # ------------------------------------------------------------------
    def record(self, task: str, metric: float, is_canary: bool) -> None:
        """Record performance for a single prediction."""

        slot = "canary" if is_canary else "production"
        self._metrics[task][slot].append(metric)
        record_metric(
            "model_performance",
            metric,
            {"task": task, "variant": slot},
        )

    # ------------------------------------------------------------------
    def evaluate(self, task: str) -> None:
        """Compare canary and production metrics and promote on success."""

        data = self._metrics.get(task)
        if not data:
            return
        canary = data["canary"]
        prod = data["production"]
        if not canary or not prod:
            return
        avg_canary = sum(canary) / len(canary)
        avg_prod = sum(prod) / len(prod)
        if len(canary) >= self.window and avg_canary > avg_prod:
            candidate = self._candidates.get(task)
            if candidate:
                self.logger.info("Promoting canary %s for %s", candidate, task)
                self.registry.promote(task, candidate)
                del self._candidates[task]
                del self._metrics[task]
        elif len(prod) >= self.window and avg_canary <= avg_prod and task in self._candidates:
            self.logger.info("Rolling back canary for %s", task)
            self.registry.rollback(task)
            del self._candidates[task]
            del self._metrics[task]

    # ------------------------------------------------------------------
    def evaluate_all(self) -> None:
        """Evaluate all tasks with registered canaries."""

        for task in list(self._candidates.keys()):
            self.evaluate(task)
