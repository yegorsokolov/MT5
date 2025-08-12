from __future__ import annotations

"""Compute portfolio exposures to risk factors."""

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class FactorRisk:
    """Estimate factor exposures using linear regression."""

    factor_returns: pd.DataFrame

    def compute_exposures(
        self, asset_returns: Iterable[float] | pd.Series
    ) -> pd.Series:
        """Return estimated exposures of ``asset_returns`` to the factors.

        Parameters
        ----------
        asset_returns:
            Sequence of portfolio returns aligned with ``factor_returns``.
        """
        y = np.asarray(list(asset_returns), dtype=float)
        X = self.factor_returns.to_numpy(dtype=float)
        if X.shape[0] != y.shape[0]:
            raise ValueError("asset_returns and factor_returns must have the same length")
        # Ordinary least squares regression without intercept
        beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        return pd.Series(beta, index=self.factor_returns.columns)

    def factor_contributions(
        self, asset_returns: Iterable[float] | pd.Series
    ) -> pd.Series:
        """Return contribution of each factor to the latest return."""
        exposures = self.compute_exposures(asset_returns)
        latest = self.factor_returns.iloc[-1]
        return exposures * latest
