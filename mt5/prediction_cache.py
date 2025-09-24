"""Simple configurable prediction cache."""

from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable


class _NoOpMetric:
    """Fallback metric shim providing ``inc``/``set`` no-ops."""

    def inc(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        return None

    def set(self, *args: Any, **kwargs: Any) -> None:  # pragma: no cover - trivial
        return None


try:  # pragma: no cover - exercised indirectly in tests
    from mt5.metrics import PRED_CACHE_HIT, PRED_CACHE_HIT_RATIO
except Exception:  # ImportError or downstream dependency issues
    PRED_CACHE_HIT = _NoOpMetric()
    PRED_CACHE_HIT_RATIO = _NoOpMetric()


def _call_metric(metric: Any, method: str, *args: Any) -> None:
    """Invoke ``method`` on ``metric`` when available."""

    func: Callable[..., Any] | None = getattr(metric, method, None)
    if callable(func):
        func(*args)


class PredictionCache:
    """A tiny cache for model predictions keyed by feature hashes.

    Parameters
    ----------
    maxsize:
        Maximum number of entries to store. ``0`` disables caching.
    policy:
        Eviction policy, either ``"lru"`` (least recently used) or ``"fifo"``.
    """

    def __init__(self, maxsize: int = 256, policy: str = "lru") -> None:
        self.maxsize = maxsize
        self.policy = policy.lower()
        # Store arbitrary prediction results keyed by integer hashes.  ``Any`` is
        # used for the value type so callers can cache lists, numpy arrays or
        # scalars without type restrictions.
        self._data: "OrderedDict[int, Any]" = OrderedDict()
        self._hits = 0
        self._lookups = 0

    def get(self, key: int) -> Any | None:
        """Return cached value for ``key`` if present."""
        self._lookups += 1
        val = self._data.get(key)
        if val is not None:
            self._hits += 1
            _call_metric(PRED_CACHE_HIT, "inc")
            if self.policy == "lru":
                self._data.move_to_end(key)
        if self._lookups:
            _call_metric(PRED_CACHE_HIT_RATIO, "set", self._hits / self._lookups)
        return val

    def set(self, key: int, value: Any) -> None:
        """Insert ``value`` for ``key`` respecting the cache size."""
        if self.maxsize <= 0:
            return
        if self.policy == "lru" and key in self._data:
            self._data.move_to_end(key)
        self._data[key] = value
        if len(self._data) > self.maxsize:
            self._data.popitem(last=False)


__all__ = ["PredictionCache"]

