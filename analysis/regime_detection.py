from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - hmmlearn may not be installed
    GaussianHMM = None  # type: ignore

try:  # pragma: no cover - optional dependency
    from .market_baskets import cluster_market_baskets
except Exception:  # pragma: no cover - provide lightweight fallback
    def cluster_market_baskets(
        df: pd.DataFrame,
        *,
        features: list[str] | None = None,
        n_baskets: int = 3,
        method: str = "kmeans",
        save_path=None,
        metadata_path=None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        """Return deterministic zero labels when clustering backends are unavailable."""

        labels = np.zeros(len(df), dtype=int)
        meta = {
            "method": method,
            "n_baskets": n_baskets,
            "features": list(features or []),
        }
        return labels, meta

try:  # pragma: no cover - optional dependency
    from .vae_regime import VAERegime, window_features
except Exception:  # pragma: no cover - provide lightweight fallback
    def window_features(arr: np.ndarray, window: int) -> np.ndarray:
        if window <= 0 or window > len(arr):
            raise ValueError("window must be within (0, len(arr)]")
        arr = np.asarray(arr)
        samples = [arr[i - window : i].reshape(-1) for i in range(window, len(arr) + 1)]
        if not samples:
            return np.zeros((0, window * arr.shape[-1]), dtype=arr.dtype)
        return np.vstack(samples)

    class VAERegime:  # type: ignore[override]
        """Simplified stub that avoids heavy ML dependencies during testing."""

        def __init__(self, input_dim: int, *args, **kwargs) -> None:
            self.input_dim = input_dim

        def fit(self, data: np.ndarray, *args, **kwargs) -> "VAERegime":
            return self

        def transform(self, data: np.ndarray, *args, **kwargs) -> np.ndarray:
            return data.astype(float, copy=False)

        def fit_predict(self, data: np.ndarray, n_clusters: int = 3, *args, **kwargs) -> np.ndarray:
            if len(data) == 0:
                return np.zeros(0, dtype=int)
            return np.zeros(len(data), dtype=int)


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
