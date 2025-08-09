"""Tools for labeling market regimes using clustering or HMM."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - hmmlearn may not be installed
    GaussianHMM = None  # type: ignore

from sklearn.cluster import KMeans


def detect_regimes(
    df: pd.DataFrame,
    n_states: int = 3,
    method: str = "hmm",
    columns: Iterable[str] = ("return", "volatility_30"),
) -> pd.Series:
    """Label each row with an estimated market regime.

    Parameters
    ----------
    df: pd.DataFrame
        Historical data including at least ``return`` and ``volatility_30`` columns.
    n_states: int
        Number of regimes to detect.
    method: str
        ``"hmm"`` uses a ``GaussianHMM`` from ``hmmlearn``. When ``hmmlearn`` is not
        available or ``method`` is not ``"hmm"`` a KMeans clustering model is used
        instead.
    columns: Iterable[str]
        Names of columns to use as features for regime detection.

    Returns
    -------
    pd.Series
        A series of integer regime labels aligned to ``df``.
    """
    features = df.loc[:, list(columns)].fillna(0).values
    if method == "hmm" and GaussianHMM is not None:
        model = GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=100, random_state=42
        )
        model.fit(features)
        labels = model.predict(features)
    else:  # fallback to clustering
        model = KMeans(n_clusters=n_states, n_init="auto", random_state=42)
        labels = model.fit_predict(features)
    return pd.Series(labels, index=df.index, name="market_regime")


def periodic_reclassification(
    df: pd.DataFrame,
    step: int = 500,
    **kwargs,
) -> pd.DataFrame:
    """Periodically re-estimate regimes to adapt to structural shifts.

    The data is split into blocks of ``step`` rows. After each block the regimes are
    re-estimated using ``detect_regimes`` on all data up to the end of the block and
    the latest labels are assigned to that block. This mimics an expanding window
    reclassification schedule.

    Parameters
    ----------
    df: pd.DataFrame
        Input market data.
    step: int
        Number of rows between reclassification runs.
    **kwargs:
        Additional arguments forwarded to :func:`detect_regimes`.

    Returns
    -------
    pd.DataFrame
        ``df`` with/updated ``market_regime`` column.
    """
    df = df.copy()
    labels = np.zeros(len(df), dtype=int)
    for end in range(step, len(df) + step, step):
        sub = df.iloc[:end]
        latest = detect_regimes(sub, **kwargs).iloc[-min(step, len(sub)) :]
        labels[end - len(latest) : end] = latest
    df["market_regime"] = labels[: len(df)]
    return df
