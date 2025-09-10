from __future__ import annotations

"""Strategy registry providing factory functions for available strategies."""

from typing import Any, Dict, Optional, Type
from importlib.metadata import entry_points

from .baseline_ma import BaselineMovingAverageStrategy

_STRATEGIES: Dict[str, Type[BaselineMovingAverageStrategy]] = {}


def register_strategy(name: str, cls: Type[BaselineMovingAverageStrategy]) -> None:
    """Register a strategy class under ``name``."""

    _STRATEGIES[name] = cls


# Register built-in strategies
register_strategy("baseline_ma", BaselineMovingAverageStrategy)


def get_strategy(name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Return a strategy configuration for ``name``.

    Parameters
    ----------
    name:
        Name of the registered strategy. If ``None``, ``baseline_ma`` is used.
    kwargs:
        Additional keyword arguments passed to the strategy constructor.
    """

    _load_external_strategies()
    strat_cls = _STRATEGIES.get(name or "baseline_ma")
    if strat_cls is None:
        raise KeyError(f"Unknown strategy: {name}")

    strat = strat_cls(**kwargs)
    return {
        "name": name or "baseline_ma",
        "generate_order": strat.generate_order,
        "update": strat.update,
    }


_external_loaded = False


def _load_external_strategies() -> None:
    """Load strategies declared via entry points."""

    global _external_loaded
    if _external_loaded:
        return
    try:
        eps = entry_points(group="mt5.strategies")
    except TypeError:  # pragma: no cover - Python<3.10
        eps = entry_points().get("mt5.strategies", [])
    except Exception:  # pragma: no cover - missing metadata
        eps = []
    for ep in eps:
        try:
            hook = ep.load()
            hook(register_strategy)
        except Exception:  # pragma: no cover - plugin errors shouldn't crash
            pass
    _external_loaded = True


__all__ = ["get_strategy", "register_strategy"]
