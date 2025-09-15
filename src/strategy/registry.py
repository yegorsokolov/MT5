from __future__ import annotations

"""Strategy registry providing factory functions for available strategies."""

from typing import Any, Dict, Optional, Type

from strategies import create_strategy, register_strategy as _register_strategy

from .baseline_ma import BaselineMovingAverageStrategy

_register_strategy(
    "baseline_ma",
    BaselineMovingAverageStrategy,
    description="Simple moving-average crossover baseline",
)


def register_strategy(name: str, cls: Type[BaselineMovingAverageStrategy]) -> None:
    """Register a strategy class under ``name``."""

    _register_strategy(name, cls)


def get_strategy(name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Return a strategy configuration for ``name``.

    Parameters
    ----------
    name:
        Name of the registered strategy. If ``None``, ``baseline_ma`` is used.
    kwargs:
        Additional keyword arguments passed to the strategy constructor.
    """

    strategy_name = name or "baseline_ma"
    strat = create_strategy(strategy_name, **kwargs)
    return {
        "name": strategy_name,
        "generate_order": strat.generate_order,
        "update": getattr(strat, "update", lambda *args, **kwargs: None),
    }


__all__ = ["get_strategy", "register_strategy"]
