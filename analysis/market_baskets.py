"""Utilities for clustering feature vectors into persistent market baskets."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

try:  # pragma: no cover - optional dependency
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    hdbscan = None  # type: ignore


def cluster_market_baskets(
    df: pd.DataFrame,
    features: Sequence[str],
    n_baskets: int = 5,
    method: str = "kmeans",
    save_path: str | Path | None = None,
    metadata_path: str | Path | None = None,
) -> Tuple[pd.Series, pd.DataFrame]:
    """Cluster feature vectors into a limited set of baskets.

    Parameters
    ----------
    df: pd.DataFrame
        Source data containing feature columns.
    features: Sequence[str]
        Columns from ``df`` to use for clustering.
    n_baskets: int, default 5
        Target number of baskets when using k-means.
    method: str, default "kmeans"
        Clustering algorithm to use ("kmeans" or "hdbscan").
    save_path: path-like, optional
        If provided, basket ids are persisted to this CSV path.
    metadata_path: path-like, optional
        If provided, basket metadata is persisted to this JSON path.

    Returns
    -------
    Tuple[pd.Series, pd.DataFrame]
        Basket id per row and a metadata dataframe describing each basket.
    """
    feat_df = df.loc[:, list(features)].fillna(0)
    X = feat_df.to_numpy()

    if method == "hdbscan" and hdbscan is not None:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=max(2, n_baskets))
        labels = clusterer.fit_predict(X)
        unique_labels = np.unique(labels[labels >= 0])
        centroids = np.array([X[labels == i].mean(axis=0) for i in unique_labels])
    else:
        clusterer = KMeans(n_clusters=n_baskets, n_init="auto", random_state=42)
        labels = clusterer.fit_predict(X)
        centroids = clusterer.cluster_centers_
        unique_labels = range(len(centroids))

    label_series = pd.Series(labels, index=df.index, name="basket_id")

    rows: list[dict[str, float | int | str]] = []
    for idx, cid in enumerate(unique_labels):
        mask = label_series == cid
        if not mask.any():
            continue
        ranges = feat_df.loc[mask].agg(["min", "max"])
        for f_i, f in enumerate(features):
            rows.append(
                {
                    "basket_id": int(cid),
                    "feature": f,
                    "centroid": float(centroids[idx][f_i]),
                    "min": float(ranges.loc["min", f]),
                    "max": float(ranges.loc["max", f]),
                }
            )
    meta = pd.DataFrame(rows)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        label_series.to_csv(save_path, header=True)
    if metadata_path:
        metadata_path = Path(metadata_path)
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        meta.to_json(metadata_path, orient="records")

    return label_series, meta


def load_basket_metadata(path: str | Path) -> pd.DataFrame:
    """Load persisted basket metadata for audit and dashboards."""
    return pd.read_json(path)
