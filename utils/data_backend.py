from __future__ import annotations

from typing import Any

from .resource_monitor import monitor, ResourceCapabilities


def _choose_module(caps: ResourceCapabilities) -> Any:
    """Return the most suitable dataframe backend for ``caps``."""
    # Prefer cuDF when a GPU is available
    if getattr(caps, "gpu", getattr(caps, "has_gpu", False)):
        try:
            import cudf

            return cudf
        except Exception:  # pragma: no cover - optional dep
            pass
    # Prefer Dask on very capable machines
    if caps.cpus >= 16 or caps.memory_gb >= 32:
        try:
            import dask.dataframe as dd

            return dd
        except Exception:  # pragma: no cover - optional dep
            pass
    # Use Polars when moderate resources are available
    if caps.cpus >= 8 or caps.memory_gb >= 16:
        try:
            import polars as pl

            return pl
        except Exception:  # pragma: no cover - optional dep
            pass
    # Fallback to pandas for lightweight environments
    import pandas as pd

    return pd


def get_dataframe_module() -> Any:
    """Return a pandas-like module for dataframe operations."""
    caps = monitor.capabilities
    return _choose_module(caps)
