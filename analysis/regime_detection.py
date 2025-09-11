from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - hmmlearn may not be installed
    GaussianHMM = None  # type: ignore

from .market_baskets import cluster_market_baskets
from .vae_regime import VAERegime, window_features


def detect_regimes(
    df: pd.DataFrame,
    n_states: int = 3,
    method: str = "hmm",
    columns: Iterable[str] = ("return", "volatility_30"),
    save_path: str | None = None,
    metadata_path: str | None = None,
) -> pd.Series:
    """Label each row with an estimated market regime.

    Parameters
    ----------
    df: pd.DataFrame
        Historical data including at least ``return`` and ``volatility_30`` columns.
    n_states: int
        Number of regimes/baskets to detect when using ``kmeans``.
    method: str
        ``"hmm"`` uses a ``GaussianHMM`` from ``hmmlearn``. ``"kmeans"`` or
        ``"hdbscan"`` cluster the feature vectors into persistent baskets.
    columns: Iterable[str]
        Names of columns to use as features for regime detection.
    save_path: str, optional
        Path to persist basket ids when using clustering methods.
    metadata_path: str, optional
        Path to persist basket metadata for audit/dashboard use.

    Returns
    -------
    pd.Series
        A series of integer regime labels aligned to ``df``.
    """
    if method == "hmm" and GaussianHMM is not None:
        features = df.loc[:, list(columns)].fillna(0).values
        model = GaussianHMM(
            n_components=n_states, covariance_type="diag", n_iter=100, random_state=42
        )
        model.fit(features)
        labels = model.predict(features)
        return pd.Series(labels, index=df.index, name="market_regime")

    labels, _ = cluster_market_baskets(
        df, features=list(columns), n_baskets=n_states, method=method, save_path=save_path, metadata_path=metadata_path
    )
    return pd.Series(labels, index=df.index, name="market_regime")


def periodic_reclassification(
    df: pd.DataFrame,
    step: int = 500,
    vae_window: int = 30,
    **kwargs,
) -> pd.DataFrame:
    """Periodically re-estimate regimes to adapt to structural shifts.

    The data is split into blocks of ``step`` rows. After each block the regimes are
    re-estimated using ``detect_regimes`` on all data up to the end of the block and
    the latest labels are assigned to that block. In parallel a variational
    autoencoder is trained on rolling windows of the same feature set and clustered
    to obtain ``vae_regime`` labels. This mimics an expanding window
    reclassification schedule.

    Parameters
    ----------
    df: pd.DataFrame
        Input market data.
    step: int
        Number of rows between reclassification runs.
    vae_window: int
        Length of the sliding window for the VAE based regime model.
    **kwargs:
        Additional arguments forwarded to :func:`detect_regimes`.

    Returns
    -------
    pd.DataFrame
        ``df`` with/updated ``market_regime`` and ``vae_regime`` columns.
    """
    df = df.copy()
    labels = np.zeros(len(df), dtype=int)
    vae_labels = np.zeros(len(df), dtype=int)
    columns = kwargs.get("columns", ("return", "volatility_30"))
    n_states = kwargs.get("n_states", 3)
    for end in range(step, len(df) + step, step):
        sub = df.iloc[:end]
        latest = detect_regimes(sub, **kwargs).iloc[-min(step, len(sub)) :]
        labels[end - len(latest) : end] = latest

        feats = sub.loc[:, list(columns)].fillna(0).values
        if len(feats) >= vae_window:
            windows = window_features(feats, vae_window)
            vae = VAERegime(windows.shape[1])
            vlabels = vae.fit_predict(windows, n_clusters=n_states)
            aligned = np.concatenate(
                [np.zeros(vae_window - 1, dtype=int), vlabels]
            )
            latest_v = aligned[-min(step, len(sub)) :]
            vae_labels[end - len(latest_v) : end] = latest_v

    df["market_regime"] = labels[: len(df)]
    df["vae_regime"] = vae_labels[: len(df)]
    return df
