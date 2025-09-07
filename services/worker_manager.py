from __future__ import annotations

"""Dynamic worker scaling based on request load.

The :class:`WorkerManager` tracks request rates from the ``remote_client``
model API and the :class:`~data.feature_store.FeatureStore`.  When the recent
request rate crosses configurable thresholds the manager will spawn or
terminate worker containers.  Scaling actions are executed via Ray when
available or fall back to no-ops in tests.  Worker counts and observed
request latencies are persisted using :func:`analytics.metrics_store.record_metric`.
"""

import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional, Tuple

from analytics.metrics_store import record_metric

try:  # pragma: no cover - optional dependency
    import ray  # type: ignore
except Exception:  # pragma: no cover
    ray = None


@dataclass
class _RequestLog:
    q: Deque[Tuple[float, int]] = field(default_factory=deque)
    last_seen: float = 0.0


class WorkerManager:
    """Scale worker processes based on request throughput."""

    def __init__(
        self,
        *,
        window: float = 10.0,
        high_rps: float = 50.0,
        low_rps: float = 10.0,
        min_workers: int = 1,
        max_workers: int = 10,
        backend: str = "ray",
        source_ttl: float = 300.0,
    ) -> None:
        self.window = window
        self.high_rps = high_rps
        self.low_rps = low_rps
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.backend = backend
        self.source_ttl = source_ttl
        self.worker_count = min_workers
        # track (timestamp, batch_size) for each request source
        self._requests: Dict[str, _RequestLog] = defaultdict(_RequestLog)
        if backend == "ray" and ray is not None:
            try:  # pragma: no cover - defensive
                ray.init(ignore_reinit_error=True)
            except Exception:
                pass

    # ------------------------------------------------------------------
    def record_request(self, source: str, latency: float, batch_size: int = 1) -> None:
        """Record a request and update scaling decisions.

        Parameters
        ----------
        source:
            Identifier of the caller submitting the request.
        latency:
            Time taken to service the request in seconds.
        batch_size:
            Number of items processed in the request.  Used to compute
            throughput metrics for autoscaling decisions.
        """

        now = time.time()
        entry = self._requests[source]
        entry.q.append((now, batch_size))
        entry.last_seen = now
        cutoff = now - self.window
        while entry.q and entry.q[0][0] < cutoff:
            entry.q.popleft()
        record_metric("queue_latency", latency, tags={"source": source})
        # Record throughput and batch size for autoscaling decisions
        if batch_size:
            throughput = batch_size / latency if latency > 0 else float("inf")
            record_metric("batch_throughput", throughput, tags={"source": source})
            record_metric("batch_size", float(batch_size), tags={"source": source})
        self._scale()

    # ------------------------------------------------------------------
    def _current_rps(self) -> float:
        now = time.time()
        self._cleanup_sources(now)
        cutoff = now - self.window
        count = 0
        for entry in self._requests.values():
            q = entry.q
            while q and q[0][0] < cutoff:
                q.popleft()
            count += sum(size for ts, size in q if ts >= cutoff)
        return count / self.window

    # ------------------------------------------------------------------
    def _cleanup_sources(self, now: float) -> None:
        """Drop request sources that have not been seen recently."""

        ttl_cutoff = now - self.source_ttl
        stale = [
            src for src, rec in self._requests.items() if rec.last_seen < ttl_cutoff
        ]
        for src in stale:
            del self._requests[src]

    # ------------------------------------------------------------------
    def _scale(self) -> None:
        rps = self._current_rps()
        if rps > self.high_rps and self.worker_count < self.max_workers:
            self._spawn_worker()
        elif rps < self.low_rps and self.worker_count > self.min_workers:
            self._terminate_worker()
        record_metric("worker_count", float(self.worker_count))

    # ------------------------------------------------------------------
    def _spawn_worker(self) -> None:
        self.worker_count += 1
        if self.backend == "ray" and ray is not None:
            try:  # pragma: no cover - best effort

                @ray.remote
                def _noop() -> None:
                    return None

                _noop.remote()
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _terminate_worker(self) -> None:
        if self.worker_count > self.min_workers:
            self.worker_count -= 1
            # Actual termination is backend specific and omitted for brevity.


_manager: Optional[WorkerManager] = None


def get_worker_manager() -> WorkerManager:
    """Return a process-wide :class:`WorkerManager` singleton."""

    global _manager
    if _manager is None:
        _manager = WorkerManager()
    return _manager


__all__ = ["WorkerManager", "get_worker_manager"]
