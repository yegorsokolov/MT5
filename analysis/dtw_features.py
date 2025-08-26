"""Dynamic Time Warping based features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute DTW distance between two 1D sequences."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = len(a), len(b)
    if n == 0 or m == 0:
        return np.nan
    cost = np.full((n + 1, m + 1), np.inf)
    cost[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diff = a[i - 1] - b[j - 1]
            cost[i, j] = diff * diff + min(cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1])
    return float(np.sqrt(cost[n, m]))


def compute(
    series: pd.Series | np.ndarray,
    window: int = 64,
    motifs: list[np.ndarray] | None = None,
    n_motifs: int = 5,
) -> pd.DataFrame:
    """Compute DTW distances between rolling windows and motif library.

    Parameters
    ----------
    series : pd.Series or np.ndarray
        Input signal.
    window : int, default 64
        Size of the rolling window for comparison.
    motifs : list of np.ndarray, optional
        Predefined motif library.  When ``None`` the first ``n_motifs``
        non-overlapping windows of ``series`` are used.
    n_motifs : int, default 5
        Number of motifs to extract when ``motifs`` is ``None``.

    Returns
    -------
    pd.DataFrame
        DataFrame containing columns ``dtw_dist_<i>`` with DTW distances for
        each motif.  Distances are ``NaN`` for positions before ``window``.
    """
    if isinstance(series, pd.Series):
        index = series.index
        arr = series.to_numpy(dtype=float)
    else:
        arr = np.asarray(series, dtype=float)
        index = None

    n = len(arr)
    if motifs is None:
        motifs = []
        for i in range(n_motifs):
            start = i * window
            if start + window <= n:
                motifs.append(arr[start : start + window].copy())
        if not motifs:
            return pd.DataFrame(index=index)

    dists = [np.full(n, np.nan, dtype=float) for _ in motifs]

    for end in range(window, n + 1):
        segment = arr[end - window : end]
        for k, motif in enumerate(motifs):
            dists[k][end - 1] = _dtw_distance(segment, motif)

    data = {f"dtw_dist_{i}": dists[i] for i in range(len(motifs))}
    return pd.DataFrame(data, index=index)


# DTW features are computationally intensive
compute.min_capability = "hpc"  # type: ignore[attr-defined]
