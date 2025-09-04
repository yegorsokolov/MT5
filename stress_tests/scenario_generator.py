from __future__ import annotations

"""Utility to create stressed scenarios by perturbing input series.

The :class:`ScenarioGenerator` applies user defined shock magnitudes to price
paths or macro variable time–series.  Shocks are expressed as percentages for
prices and as additive moves for macro variables.  Generated scenarios can be
fed into :mod:`stress_tests.scenario_runner` for evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class ScenarioGenerator:
    """Generate shocked time–series for stress testing.

    Parameters
    ----------
    price_shocks:
        Mapping of scenario label to percentage shock to apply to price paths
        or PnL series.  A value of ``-0.2`` represents a 20\% drop.
    macro_shocks:
        Optional mapping of macro economic variable names to additive shocks.
        Each entry generates a new dataframe where the column has been
        shifted by the specified amount.
    """

    price_shocks: Dict[str, float] = field(default_factory=dict)
    macro_shocks: Dict[str, float] = field(default_factory=dict)

    # ------------------------------------------------------------------
    def shock_prices(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """Return price paths with the configured shocks applied."""

        scenarios: Dict[str, pd.Series] = {}
        for label, magnitude in self.price_shocks.items():
            scenarios[label] = prices * (1 + magnitude)
        return scenarios

    # ------------------------------------------------------------------
    def shock_macros(self, macros: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Return macro dataframes with additive shocks applied."""

        scenarios: Dict[str, pd.DataFrame] = {}
        for column, magnitude in self.macro_shocks.items():
            if column in macros.columns:
                df = macros.copy()
                df[column] = df[column] + magnitude
                scenarios[column] = df
        return scenarios

    # ------------------------------------------------------------------
    def generate_pnl(self, pnl: pd.Series) -> Dict[str, pd.Series]:
        """Apply price shocks to a PnL series.

        Assumes PnL scales linearly with price moves.  Each configured
        ``price_shock`` returns a new PnL series with the shock applied.
        """

        scenarios: Dict[str, pd.Series] = {}
        for label, magnitude in self.price_shocks.items():
            scenarios[label] = pnl * (1 + magnitude)
        return scenarios

