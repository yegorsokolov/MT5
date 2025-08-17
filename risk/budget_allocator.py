from __future__ import annotations

"""Allocate capital between strategies based on risk metrics."""

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd

from analytics.metrics_store import record_metric


@dataclass
class BudgetAllocator:
    """Compute per-strategy risk budgets.

    Budgets are assigned inversely proportional to the product of historical
    drawdown magnitude and current volatility.  Lower risk strategies therefore
    receive a larger share of ``capital``.
    """

    capital: float
    budgets: Dict[str, float] = field(default_factory=dict)

    def _max_drawdown(self, returns: pd.Series) -> float:
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        return float(drawdown.min())

    def allocate(self, returns: Dict[str, pd.Series]) -> Dict[str, float]:
        """Return new budget allocations for each strategy."""
        scores: Dict[str, float] = {}
        for name, series in returns.items():
            if series.empty:
                continue
            dd = abs(self._max_drawdown(series))
            vol = float(series.std(ddof=1))
            scores[name] = (dd + 1e-6) * (vol + 1e-6)
        if not scores:
            self.budgets = {}
            return {}
        inv = {k: 1.0 / (v + 1e-12) if np.isfinite(v) else 0.0 for k, v in scores.items()}
        total = sum(inv.values()) or 1.0
        self.budgets = {k: self.capital * w / total for k, w in inv.items()}
        for strat, budget in self.budgets.items():
            try:
                record_metric("risk_budget", budget, {"strategy": strat})
            except Exception:
                pass
        return self.budgets

    def fraction(self, strategy: str) -> float:
        """Return capital fraction allocated to ``strategy``."""
        if not self.budgets:
            return 1.0
        return self.budgets.get(strategy, 0.0) / self.capital
