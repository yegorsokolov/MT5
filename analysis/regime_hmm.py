from __future__ import annotations

"""Simple HMM based market regime labelling."""

from typing import Iterable, Sequence

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - hmmlearn may be missing
    GaussianHMM = None  # type: ignore


def fit_regime_hmm(
    df: pd.DataFrame,
    n_states: int = 3,
    column: str = "regime",
    features: Sequence[str] | None = ("return", "volatility_30"),
) -> pd.DataFrame:
    """Fit a Gaussian HMM on ``df`` and append discrete regimes.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing at least ``return`` and ``volatility_30``.
    n_states : int, default 3
        Number of hidden states for the HMM.
    column : str, default ``"regime"``
        Name of the output column.
    features : sequence of str, optional
        Which columns to use as features for the HMM. Defaults to
        ``("return", "volatility_30")``.
    """

    feats = list(features) if features is not None else ["return", "volatility_30"]
    if GaussianHMM is None or not set(feats).issubset(df.columns):
        out = df.copy()
        out[column] = 0
        return out

    X = df.loc[:, feats].fillna(0).to_numpy()
    model = GaussianHMM(
        n_components=n_states, covariance_type="diag", n_iter=100, random_state=42
    )
    try:
        model.fit(X)
        regimes = model.predict(X)
    except Exception:
        regimes = np.zeros(len(df), dtype=int)

    out = df.copy()
    out[column] = regimes
    return out


__all__ = ["fit_regime_hmm"]
