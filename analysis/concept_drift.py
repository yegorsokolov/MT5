from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Mapping

import pandas as pd
from scheduler import schedule_retrain
from analytics import mlflow_client as mlflow

try:  # optional dependency
    from event_store import EventStore  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    EventStore = None  # type: ignore


class PageHinkley:
    """Simple Page-Hinkley change detector."""

    def __init__(self, delta: float = 0.005, threshold: float = 50.0) -> None:
        self.delta = delta
        self.threshold = threshold
        self.mean = 0.0
        self.cumulative = 0.0
        self.min_cum = 0.0
        self.t = 0

    def update(self, x: float) -> bool:
        self.t += 1
        self.mean += (x - self.mean) / self.t
        self.cumulative += x - self.mean - self.delta
        self.min_cum = min(self.cumulative, self.min_cum)
        if self.cumulative - self.min_cum > self.threshold:
            self.reset()
            return True
        return False

    def reset(self) -> None:
        self.mean = 0.0
        self.cumulative = 0.0
        self.min_cum = 0.0
        self.t = 0


class ADWIN:
    """Very small ADWIN-like detector using a sliding window."""

    def __init__(self, delta: float = 0.002, max_window: int = 200) -> None:
        self.delta = delta
        self.max_window = max_window
        self.window: list[float] = []

    def update(self, x: float) -> bool:
        import math

        self.window.append(x)
        drift = False
        while len(self.window) > 10:
            n = len(self.window)
            for cut in range(5, n - 5):
                w0 = self.window[:cut]
                w1 = self.window[cut:]
                diff = abs(sum(w0) / len(w0) - sum(w1) / len(w1))
                eps = math.sqrt(2 * math.log(1 / self.delta) / cut)
                if diff > eps:
                    self.window = w1
                    drift = True
                    break
            else:
                break
        if len(self.window) > self.max_window:
            self.window = self.window[-self.max_window :]
        return drift


class _MemoryStore:
    """Lightweight in-memory event store used when full store is unavailable."""

    def __init__(self) -> None:
        self.events: list[dict[str, object]] = []

    def record(self, event_type: str, payload: dict[str, object]) -> None:
        self.events.append({"timestamp": "", "type": event_type, "payload": payload})

    def iter_events(self, event_type: str | None = None):
        for ev in self.events:
            if event_type is None or ev["type"] == event_type:
                yield ev


class ConceptDriftMonitor:
    """Monitor feature and prediction streams for concept drift.

    Parameters
    ----------
    method:
        Drift detector to use, either ``"adwin"`` or ``"pagehinkley"``.
    delta:
        Sensitivity parameter for ADWIN. Ignored for Page-Hinkley.
    store:
        Event store used to record retrain events.
    log_dir:
        Directory where drift events are logged.
    """

    def __init__(
        self,
        method: str = "adwin",
        delta: float = 0.002,
        store: "EventStore" | None = None,
        log_dir: Path | None = None,
    ) -> None:
        self.method = method.lower()
        self.delta = delta
        if store is not None:
            self.store = store
        elif EventStore is not None:
            self.store = EventStore()
        else:  # fallback to in-memory store
            self.store = _MemoryStore()
        self.log_dir = Path(log_dir or Path("reports") / "drift")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self._feat_detectors: Dict[str, ADWIN | PageHinkley] = {}
        self._pred_detector: ADWIN | PageHinkley = self._make_detector()

    def _make_detector(self) -> ADWIN | PageHinkley:
        if self.method == "pagehinkley":
            return PageHinkley()
        return ADWIN(delta=self.delta)

    def update(
        self,
        features: Mapping[str, float] | pd.Series,
        prediction: float,
        model: str = "classic",
    ) -> bool:
        """Update detectors with new observation.

        Returns ``True`` if any drift was detected.
        """

        feats: Mapping[str, float]
        if isinstance(features, pd.Series):
            feats = features.to_dict()
        else:
            feats = dict(features)

        drifted = False
        for name, val in feats.items():
            det = self._feat_detectors.get(name)
            if det is None:
                det = self._make_detector()
                self._feat_detectors[name] = det
            if det.update(float(val)):
                drifted = True
                self._handle_drift(f"feature:{name}", model)

        if self._pred_detector.update(float(prediction)):
            drifted = True
            self._handle_drift("prediction", model)

        return drifted

    def _handle_drift(self, source: str, model: str) -> None:
        ts = pd.Timestamp.utcnow().isoformat()
        log_path = self.log_dir / f"{ts.replace(':', '-')}_{source}.json"
        with open(log_path, "w") as fh:
            json.dump({"timestamp": ts, "source": source}, fh)
        try:
            mlflow.log_metric("concept_drift", 1.0)
        except Exception:  # pragma: no cover - mlflow optional
            pass
        try:
            schedule_retrain(model=model, update_hyperparams=True, store=self.store)
        except Exception:
            self.logger.exception("Failed scheduling retrain after drift in %s", source)
        self.logger.warning("Concept drift detected in %s", source)


__all__ = ["ConceptDriftMonitor"]
