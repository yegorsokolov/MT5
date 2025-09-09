from __future__ import annotations

"""Strategy registry providing factory functions for available strategies."""

from typing import Any, Dict, Optional, Type

from .baseline_ma import BaselineMovingAverageStrategy

_STRATEGIES: Dict[str, Type[BaselineMovingAverageStrategy]] = {
    "baseline_ma": BaselineMovingAverageStrategy,
}


def get_strategy(name: Optional[str] = None, **kwargs: Any) -> Dict[str, Any]:
    """Return a strategy configuration for ``name``.

    Parameters
    ----------
    name:
        Name of the registered strategy. If ``None``, ``baseline_ma`` is used.
    kwargs:
        Additional keyword arguments passed to the strategy constructor.
    """

    strat_cls = _STRATEGIES.get(name or "baseline_ma")
    if strat_cls is None:
        raise KeyError(f"Unknown strategy: {name}")

    strat = strat_cls(**kwargs)
    return {
        "approved": True,
        "generate_order": strat.generate_order,
        "update": strat.update,
    }
