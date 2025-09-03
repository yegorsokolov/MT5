from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Protocol


class _EventStore(Protocol):
    def record(self, event_type: str, payload: dict) -> None:
        ...


class PerformanceMonitor:
    """Monitor PnL and prediction drift and emit retrain events."""

    def __init__(
        self,
        *,
        pnl_threshold: float = -1.0,
        drift_threshold: float = 0.1,
        window: int = 100,
        model: str = "classic",
        store: Optional[_EventStore] = None,
    ) -> None:
        """Initialise the monitor.

        Parameters
        ----------
        pnl_threshold:
            PnL below this value triggers a retrain event.
        drift_threshold:
            Absolute deviation from the rolling mean prediction that triggers
            a retrain event.
        window:
            Number of previous predictions used to compute the rolling mean.
        model:
            Identifier of the model that should be retrained.  Used by the
            scheduler to select the training script.
        store:
            Optional :class:`EventStore` used for emitting retrain events.
        """

        self.pnl_threshold = pnl_threshold
        self.drift_threshold = drift_threshold
        self.model = model
        self.predictions: Deque[float] = deque(maxlen=window)
        if store is None:
            from event_store import EventStore  # imported lazily

            self.store: _EventStore = EventStore()
        else:
            self.store = store

    # ------------------------------------------------------------------
    def record(self, pnl: float, prediction: float) -> None:
        """Record a trade outcome and model prediction.

        A retrain event is emitted when either ``pnl`` falls below
        ``pnl_threshold`` or when the latest prediction deviates from the
        rolling mean by more than ``drift_threshold``.
        """

        if pnl <= self.pnl_threshold:
            self.store.record(
                "retrain",
                {"model": self.model, "reason": "pnl", "pnl": float(pnl)},
            )

        if self.predictions:
            mean_pred = sum(self.predictions) / len(self.predictions)
            drift = abs(prediction - mean_pred)
            if drift > self.drift_threshold:
                self.store.record(
                    "retrain",
                    {
                        "model": self.model,
                        "reason": "prediction_drift",
                        "drift": float(drift),
                    },
                )

        self.predictions.append(prediction)


__all__ = ["PerformanceMonitor"]
