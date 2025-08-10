"""Utilities for working with Ray if available."""
from __future__ import annotations

import types

try:  # pragma: no cover - executed when Ray is installed
    import ray  # type: ignore
    RAY_AVAILABLE = True
except Exception:  # pragma: no cover - handled in tests
    import ray_stub as ray  # type: ignore
    RAY_AVAILABLE = False


def init(address: str | None = None, **kwargs) -> None:
    """Initialize Ray if available."""
    if RAY_AVAILABLE:
        ray.init(address=address, **kwargs)


def shutdown() -> None:
    """Shutdown Ray if it was initialized."""
    if RAY_AVAILABLE:
        ray.shutdown()


def cluster_available() -> bool:
    """Return True if Ray has multiple nodes available."""
    if not RAY_AVAILABLE:
        return False
    try:
        return len(ray.nodes()) > 1
    except Exception:  # pragma: no cover - safeguard
        return False


def submit(func, *args, **kwargs):
    """Execute ``func`` remotely via Ray when possible."""
    if RAY_AVAILABLE and cluster_available():
        remote_func = ray.remote(func)
        return ray.get(remote_func.remote(*args, **kwargs))
    return func(*args, **kwargs)


__all__ = ["ray", "RAY_AVAILABLE", "init", "shutdown", "cluster_available", "submit"]
