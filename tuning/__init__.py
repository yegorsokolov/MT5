"""Hyperparameter tuning utilities."""

from typing import Any

__all__ = ["AutoOptimizer", "EntryExitOptimizer"]


def __getattr__(name: str) -> Any:  # pragma: no cover - trivial
    if name == "AutoOptimizer":
        from .auto_optimizer import AutoOptimizer

        return AutoOptimizer
    if name == "EntryExitOptimizer":
        from .entry_exit_opt import EntryExitOptimizer

        return EntryExitOptimizer
    raise AttributeError(name)
