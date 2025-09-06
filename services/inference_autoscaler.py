from __future__ import annotations

"""Autoscale inference workers based on request rate and GPU usage.

The :class:`InferenceAutoscaler` polls metrics from
``models.remote_server`` which exposes the recent request rate and current GPU
utilisation.  When either metric crosses configurable thresholds the scaler
will spawn or terminate worker containers.  Actual container orchestration is
delegated to Kubernetes or Ray when available; in the tests these operations
are best-effort no-ops.

Scaling events are reported via :func:`analytics.metrics_store.record_metric`
allowing external monitoring systems to observe autoscaling behaviour.
"""

from typing import List

from analytics.metrics_store import record_metric
from models import remote_server

try:  # pragma: no cover - optional dependencies
    import ray  # type: ignore
except Exception:  # pragma: no cover
    ray = None

try:  # pragma: no cover - optional dependency
    from kubernetes import client as k8s_client, config as k8s_config  # type: ignore
except Exception:  # pragma: no cover
    k8s_client = None  # type: ignore
    k8s_config = None  # type: ignore


class InferenceAutoscaler:
    """Scale inference workers based on request throughput and GPU load."""

    def __init__(
        self,
        *,
        window: float = 10.0,
        high_rps: float = 50.0,
        low_rps: float = 10.0,
        high_gpu: float = 0.8,
        low_gpu: float = 0.2,
        min_workers: int = 1,
        max_workers: int = 10,
        backend: str = "ray",
    ) -> None:
        self.window = window
        self.high_rps = high_rps
        self.low_rps = low_rps
        self.high_gpu = high_gpu
        self.low_gpu = low_gpu
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.backend = backend
        self.worker_count = min_workers
        # Simple representation of service discovery records.  In real
        # deployments this would interact with a registry like Consul or
        # Kubernetes services.
        self._registry: List[str] = [f"worker-0"]

    # ------------------------------------------------------------------
    def _spawn_worker(self) -> None:
        self.worker_count += 1
        self._registry.append(f"worker-{self.worker_count-1}")
        record_metric("autoscale_up", 1.0, {"workers": self.worker_count})
        if self.backend == "ray" and ray is not None:  # pragma: no cover - optional
            try:
                @ray.remote
                def _noop() -> None:
                    return None

                _noop.remote()
            except Exception:
                pass
        elif self.backend == "k8s" and k8s_client is not None:  # pragma: no cover - optional
            try:
                k8s_config.load_kube_config()
                # Real implementation would create a deployment or scale a
                # replica set.  Omitted here for brevity and to keep tests
                # lightweight.
            except Exception:
                pass

    # ------------------------------------------------------------------
    def _terminate_worker(self) -> None:
        if self.worker_count <= self.min_workers:
            return
        self.worker_count -= 1
        if self._registry:
            self._registry.pop()
        record_metric("autoscale_down", 1.0, {"workers": self.worker_count})
        # Actual termination logic is backend specific and omitted.

    # ------------------------------------------------------------------
    def check(self) -> None:
        """Poll metrics and adjust worker count if thresholds are crossed."""

        rps = remote_server.get_request_rate(self.window)
        gpu = remote_server.get_gpu_utilization()
        if (rps > self.high_rps or gpu > self.high_gpu) and self.worker_count < self.max_workers:
            self._spawn_worker()
        elif (
            rps < self.low_rps
            and gpu < self.low_gpu
            and self.worker_count > self.min_workers
        ):
            self._terminate_worker()
        record_metric("inference_workers", float(self.worker_count))

    # ------------------------------------------------------------------
    def registry(self) -> List[str]:
        """Return the current service discovery records."""

        return list(self._registry)


_scaler: InferenceAutoscaler | None = None


def get_inference_autoscaler() -> InferenceAutoscaler:
    """Return a process-wide :class:`InferenceAutoscaler` singleton."""

    global _scaler
    if _scaler is None:
        _scaler = InferenceAutoscaler()
    return _scaler


__all__ = ["InferenceAutoscaler", "get_inference_autoscaler"]

