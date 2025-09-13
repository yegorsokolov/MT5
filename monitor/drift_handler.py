from __future__ import annotations

import time
from typing import Callable, Mapping

from analysis.concept_drift import ConceptDriftMonitor
from scheduler import schedule_retrain


class DriftHandler:
    """Handle concept drift events and trigger mitigation actions.

    Parameters
    ----------
    monitor:
        :class:`~analysis.concept_drift.ConceptDriftMonitor` instance emitting
        drift signals.
    threshold:
        Number of drift detections required before actions are triggered.
    cooldown:
        Seconds to wait before triggering actions again for the same model.
    retrain_cb:
        Callback invoked to schedule model retraining. Defaults to
        :func:`scheduler.schedule_retrain`.
    allocate_cb:
        Optional callback to adjust capital allocation away from drifting
        strategies.
    """

    def __init__(
        self,
        monitor: ConceptDriftMonitor,
        threshold: int = 3,
        cooldown: float = 3600.0,
        retrain_cb: Callable[..., None] = schedule_retrain,
        allocate_cb: Callable[[str], None] | None = None,
    ) -> None:
        self.monitor = monitor
        self.threshold = threshold
        self.cooldown = cooldown
        self.retrain_cb = retrain_cb
        self.allocate_cb = allocate_cb
        self._counts: dict[str, int] = {}
        self._last_action: dict[str, float] = {}

    def update(
        self,
        features: Mapping[str, float],
        prediction: float,
        model: str = "classic",
    ) -> bool:
        """Feed new observation to the monitor and handle drift.

        Returns ``True`` if drift was detected by the underlying monitor.
        """
        drifted = self.monitor.update(features, prediction, model=model)
        if not drifted:
            return False
        cnt = self._counts.get(model, 0) + 1
        self._counts[model] = cnt
        if cnt < self.threshold:
            return True
        now = time.time()
        last = self._last_action.get(model, 0.0)
        if now - last < self.cooldown:
            return True
        self.retrain_cb(model=model)
        if self.allocate_cb:
            try:
                self.allocate_cb(model)
            except TypeError:
                # Allow callbacks accepting (model, fraction)
                self.allocate_cb(model, 0.0)  # type: ignore[misc]
        self._counts[model] = 0
        self._last_action[model] = now
        return True


__all__ = ["DriftHandler"]
