from __future__ import annotations

"""Market regime detection using clustering into stable market baskets."""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from analysis.market_baskets import cluster_market_baskets

try:  # pragma: no cover - optional dependency
    from utils import load_config
except Exception:  # pragma: no cover - optional dependency
    load_config = lambda: {}  # type: ignore

try:  # pragma: no cover - optional dependency
    from hmmlearn.hmm import GaussianHMM
except Exception:  # pragma: no cover - hmmlearn may be missing
    GaussianHMM = None  # type: ignore


DEFAULT_BASKET_PATH = Path("analysis") / "market_baskets.csv"
DEFAULT_META_PATH = Path("analysis") / "market_baskets_meta.json"


def label_regimes(
    df: pd.DataFrame,
    n_states: int = 3,
    column: str = "market_regime",
    method: str = "kmeans",
    save_path: Path | str = DEFAULT_BASKET_PATH,
    metadata_path: Path | str = DEFAULT_META_PATH,
) -> pd.DataFrame:
    """Label each row with a market regime based on feature baskets.

    Parameters
    ----------
    df : pd.DataFrame
        Data containing at least ``return`` and ``volatility_30`` columns.
    n_states : int, default 3
        Number of baskets for k-means clustering.
    column : str, default ``"market_regime"``
        Name of the output column to store basket identifiers.
    method : str, default ``"kmeans"``
        Clustering algorithm ("kmeans", "hdbscan", or "hmm" for the legacy model).
    save_path : path-like, optional
        Where to persist the basket ids as CSV.
    metadata_path : path-like, optional
        Where to persist basket metadata for audits/visualisation.
    """
    features = ["return", "volatility_30"]
    if set(features).issubset(df.columns) and method != "hmm":
        cfg = load_config() if callable(load_config) else {}
        save_path = cfg.get("basket_id_path", save_path)
        metadata_path = cfg.get("basket_metadata_path", metadata_path)
        labels, meta = cluster_market_baskets(
            df,
            features=features,
            n_baskets=n_states,
            method=method,
            save_path=save_path,
            metadata_path=metadata_path,
        )
        out = df.copy()
        out[column] = labels
        out.attrs["basket_metadata"] = meta
        return out

    if set(features).issubset(df.columns) and GaussianHMM is not None:
        feat_vals = df[features].fillna(0).values
        try:
            model = GaussianHMM(
                n_components=n_states, covariance_type="diag", n_iter=100, random_state=42
            )
            model.fit(feat_vals)
            regimes = model.predict(feat_vals)
        except Exception:
            regimes = np.zeros(len(df), dtype=int)
    else:
        regimes = np.zeros(len(df), dtype=int)

    df[column] = regimes
    return df
