"""Utilities for working with Ray if available."""
from __future__ import annotations

try:  # pragma: no cover - executed when Ray is installed
    import ray  # type: ignore
    RAY_AVAILABLE = True
except Exception:  # pragma: no cover - handled in tests
    from mt5 import ray_stub as ray  # type: ignore
    RAY_AVAILABLE = False


def _ray_is_initialized() -> bool:
    """Return ``True`` when Ray has already been initialised."""

    if not RAY_AVAILABLE:
        return False

    checker = getattr(ray, "is_initialized", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:  # pragma: no cover - defensive
            return False

    try:  # pragma: no cover - legacy Ray fallback
        worker = getattr(ray, "worker", None)
        if worker is not None:
            global_worker = getattr(worker, "global_worker", None)
            if global_worker is not None and getattr(global_worker, "connected", False):
                return True
    except Exception:
        return False

    return False


def init(address: str | None = None, **kwargs) -> bool:
    """Initialize Ray if available and return whether this call started it."""
    if not RAY_AVAILABLE:
        return False

    if _ray_is_initialized():
        return False

    ray.init(address=address, **kwargs)
    return True


def shutdown() -> None:
    """Shutdown Ray if it was initialized."""
    if not RAY_AVAILABLE or not _ray_is_initialized():
        return None

    ray.shutdown()
    return None


def cluster_available() -> bool:
    """Return True if Ray has multiple nodes available."""
    if not RAY_AVAILABLE or not _ray_is_initialized():
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
