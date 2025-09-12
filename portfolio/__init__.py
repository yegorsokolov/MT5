from __future__ import annotations

"""Portfolio optimisation utilities."""

import os

from .optimizer import PortfolioOptimizer, PortfolioRebalancer
from .hrp_optimizer import HRPOptimizer, hrp_weights
from .robust_optimizer import RobustOptimizer

__all__ = [
    "PortfolioOptimizer",
    "PortfolioRebalancer",
    "HRPOptimizer",
    "hrp_weights",
    "RobustOptimizer",
    "optimizer_from_config",
]


def optimizer_from_config(name: str | None = None):
    """Return optimiser selected by ``name`` or ``PORTFOLIO_OPTIMIZER`` env var."""
    method = (name or os.getenv("PORTFOLIO_OPTIMIZER", "robust")).lower()
    if method in {"hrp", "hierarchical"}:
        return HRPOptimizer()
    if method in {"mv", "mean_variance"}:
        return PortfolioOptimizer()
    return RobustOptimizer()
