"""Track model inference latency and compute moving averages."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict


@dataclass
class InferenceLatency:
    """Record per-model inference times and expose a moving average.

    Parameters
    ----------
    window:
        Number of recent samples to consider for the moving average.
    """

    window: int = 50
    _samples: Dict[str, Deque[float]] = field(default_factory=dict)
    _sums: Dict[str, float] = field(default_factory=dict)

    def record(self, model_name: str, duration: float) -> None:
        """Record an inference ``duration`` for ``model_name``."""

        q = self._samples.setdefault(model_name, deque())
        q.append(duration)
        self._sums[model_name] = self._sums.get(model_name, 0.0) + duration
        if len(q) > self.window:
            self._sums[model_name] -= q.popleft()

    def moving_average(self, model_name: str) -> float:
        """Return the moving average latency for ``model_name``.

        Returns ``0.0`` if no samples have been recorded for ``model_name``.
        """

        q = self._samples.get(model_name)
        if not q:
            return 0.0
        return self._sums.get(model_name, 0.0) / len(q)

    def reset(self, model_name: str) -> None:
        """Clear recorded samples for ``model_name``."""

        self._samples.pop(model_name, None)
        self._sums.pop(model_name, None)
