"""Utilities for clustering feature vectors into persistent market baskets."""
from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import csv
import logging
from datetime import datetime

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
    min_count: int = 10,
    merge_log: str | Path = Path("logs") / "basket_merges.csv",
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

    # Track counts and centroids for potential merging of rare baskets
    counts = label_series.value_counts().to_dict()
    centroid_map = {cid: centroids[idx] for idx, cid in enumerate(unique_labels)}

    def _log_merge(source: int, target: int, source_cnt: int, target_before: int, target_after: int) -> None:
        """Persist merge event and notify operators."""
        path = Path(merge_log)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not path.exists()
        with path.open("a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow([
                    "timestamp",
                    "source",
                    "target",
                    "source_count",
                    "target_count_before",
                    "target_count_after",
                ])
            writer.writerow([
                datetime.utcnow().isoformat(),
                source,
                target,
                source_cnt,
                target_before,
                target_after,
            ])
        logging.getLogger(__name__).warning(
            "Merged basket %s (count %s) into basket %s (count %s -> %s)",
            source,
            source_cnt,
            target,
            target_before,
            target_after,
        )

    # Merge baskets with insufficient observations
    while True:
        small = [cid for cid, cnt in counts.items() if cnt < min_count]
        if not small:
            break
        source = small[0]
        source_centroid = centroid_map[source]
        candidates = {cid: c for cid, c in centroid_map.items() if cid != source}
        if not candidates:
            break
        # Select nearest candidate by Euclidean distance
        dists = {cid: np.linalg.norm(source_centroid - c) for cid, c in candidates.items()}
        target = min(dists, key=dists.get)
        target_before = counts[target]
        source_cnt = counts[source]
        # Reassign labels
        label_series[label_series == source] = target
        counts[target] += source_cnt
        del counts[source]
        # Update centroid of target using weighted average
        centroid_map[target] = (
            centroid_map[target] * target_before + source_centroid * source_cnt
        ) / counts[target]
        del centroid_map[source]
        _log_merge(source, target, source_cnt, target_before, counts[target])

    rows: list[dict[str, float | int | str]] = []
    for cid, centroid in centroid_map.items():
        mask = label_series == cid
        if not mask.any():
            continue
        ranges = feat_df.loc[mask].agg(["min", "max"])
        for f_i, f in enumerate(features):
            rows.append(
                {
                    "basket_id": int(cid),
                    "feature": f,
                    "centroid": float(centroid[f_i]),
                    "min": float(ranges.loc["min", f]),
                    "max": float(ranges.loc["max", f]),
                    "count": int(counts[cid]),
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
