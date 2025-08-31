from __future__ import annotations

"""Estimate shared risk drivers across instruments.

This module implements a light-weight factor model based on principal
component analysis (PCA).  Given a matrix of asset returns it extracts a small
number of orthogonal factors that explain the majority of the cross-sectional
variance.  The resulting factor returns and asset exposures can then be used by
portfolio construction components to account for correlation structure between
instruments.

The implementation intentionally avoids external dependencies such as
``scikit-learn`` in order to keep the core library lightweight.  Only ``numpy``
and ``pandas`` are required.
"""

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import pandas as pd


@dataclass
class FactorModel:
    """Principal component based factor model."""

    n_factors: int | None = None
    factor_returns_: pd.DataFrame | None = None
    exposures_: pd.DataFrame | None = None

    def fit(self, returns: pd.DataFrame, n_factors: int | None = None) -> "FactorModel":
        """Fit the model to ``returns``.

        Parameters
        ----------
        returns:
            DataFrame of asset returns with shape ``(n_samples, n_assets)``.
        n_factors:
            Optional number of factors to extract.  When omitted the value
            supplied at initialisation is used; if that is also ``None`` the
            minimum of the number of assets and 5 is chosen.
        """

        if returns.empty:
            raise ValueError("returns must contain data")

        self.n_factors = n_factors or self.n_factors
        if self.n_factors is None:
            self.n_factors = min(returns.shape[1], 5)

        # Center data and compute SVD.  Using SVD is numerically stable and
        # provides the principal components directly.
        r = returns - returns.mean()
        U, s, Vt = np.linalg.svd(r.to_numpy(), full_matrices=False)
        k = self.n_factors
        U = U[:, :k]
        s = s[:k]
        Vt = Vt[:k]

        # Factor returns are the principal component scores scaled by singular
        # values.  Exposures (loadings) correspond to the right singular vectors.
        factor_returns = U * s
        self.factor_returns_ = pd.DataFrame(
            factor_returns,
            index=returns.index,
            columns=[f"factor_{i+1}" for i in range(k)],
        )
        self.exposures_ = pd.DataFrame(
            Vt.T,
            index=returns.columns,
            columns=self.factor_returns_.columns,
        )
        return self

    @property
    def factor_names(self) -> List[str]:
        if self.factor_returns_ is None:
            return []
        return list(self.factor_returns_.columns)

    def get_factor_returns(self) -> pd.DataFrame:
        """Return DataFrame of factor returns."""
        if self.factor_returns_ is None:
            raise ValueError("model is not fitted")
        return self.factor_returns_

    def get_exposures(self) -> pd.DataFrame:
        """Return DataFrame of factor loadings per asset."""
        if self.exposures_ is None:
            raise ValueError("model is not fitted")
        return self.exposures_

    def factor_exposures_for(self, asset: str) -> pd.Series:
        """Return exposures for a single ``asset``."""
        if self.exposures_ is None:
            raise ValueError("model is not fitted")
        if asset not in self.exposures_.index:
            raise KeyError(f"unknown asset: {asset}")
        return self.exposures_.loc[asset]
