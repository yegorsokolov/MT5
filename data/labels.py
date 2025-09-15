from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

from analysis.data_lineage import log_lineage


def triple_barrier(
    prices: "pd.Series", pt_mult: float, sl_mult: float, max_horizon: int
) -> "pd.Series":
    """Generate triple barrier labels using a vectorised approach.

    Parameters
    ----------
    prices : pd.Series
        Series of prices indexed by time.
    pt_mult : float
        Multiplier for the profit-taking upper barrier.
    sl_mult : float
        Multiplier for the stop-loss lower barrier.
    max_horizon : int
        Maximum number of steps to look ahead.

    Returns
    -------
    pd.Series
        Labels with values ``1`` (upper barrier hit), ``-1`` (lower barrier
        hit) or ``0`` (no barrier hit within horizon).
    """

    arr = prices.to_numpy()
    n = len(arr)
    horizon = min(max_horizon, n - 1)
    if horizon <= 0:
        labels = np.zeros(n, dtype=np.int8)
    else:
        padded = np.pad(arr, (0, horizon), constant_values=arr[-1])
        windows = sliding_window_view(padded, horizon + 1)
        future = windows[:, 1:]
        p0 = windows[:, [0]]
        upper = p0 * (1 + pt_mult)
        lower = p0 * (1 - sl_mult)

        cummax = np.maximum.accumulate(future, axis=1)
        cummin = np.minimum.accumulate(future, axis=1)
        hit_upper = cummax >= upper
        hit_lower = cummin <= lower

        first_upper = np.where(hit_upper.any(axis=1), hit_upper.argmax(axis=1) + 1, horizon + 1)
        first_lower = np.where(hit_lower.any(axis=1), hit_lower.argmax(axis=1) + 1, horizon + 1)

        labels = np.zeros(n, dtype=np.int8)
        labels[first_upper < first_lower] = 1
        labels[first_lower < first_upper] = -1

    labels = pd.Series(labels, index=prices.index, dtype=int)

    run_id = prices.attrs.get("run_id", "unknown")
    raw_file = prices.attrs.get("source", "unknown")
    log_lineage(run_id, raw_file, "triple_barrier", "label")
    return labels


def multi_horizon_labels(prices: "pd.Series", horizons: list[int]) -> "pd.DataFrame":
    """Generate targets for multiple prediction horizons.

    For each horizon ``h`` the function creates three columns:

    ``label_{h}``
        Binary direction label indicating whether the price after ``h``
        steps is greater than the current price.

    ``abs_return_{h}``
        Absolute percentage return ``|p(t+h)/p(t) - 1|``.

    ``vol_{h}``
        Realised volatility calculated as ``sqrt(sum(r_i^2))`` for the
        ``h`` forward percent returns ``r``.

    The trailing rows where the future is unknown are filled with ``0``.

    Parameters
    ----------
    prices : pd.Series
        Series of prices indexed by time.
    horizons : list[int]
        List of prediction horizons in steps.

    Returns
    -------
    pd.DataFrame
        DataFrame with three columns per horizon.
    """

    data: dict[str, pd.Series] = {}
    pct = prices.pct_change()
    future_pct = pct.shift(-1)

    for h in horizons:
        future = prices.shift(-h)
        data[f"label_{h}"] = (future > prices).astype("int8").fillna(0)
        abs_ret = (future / prices - 1).abs().fillna(0).astype("float32")
        data[f"abs_return_{h}"] = abs_ret
        vol = (
            future_pct.rolling(window=h)
            .apply(lambda arr: np.sqrt(np.square(arr).sum()), raw=True)
            .shift(-h + 1)
        )
        data[f"vol_{h}"] = vol.fillna(0).astype("float32")

    labels = pd.DataFrame(data, index=prices.index)

    run_id = prices.attrs.get("run_id", "unknown")
    raw_file = prices.attrs.get("source", "unknown")
    for h in horizons:
        log_lineage(run_id, raw_file, f"multi_horizon_{h}", "label")
        log_lineage(run_id, raw_file, f"multi_horizon_{h}", "abs_return")
        log_lineage(run_id, raw_file, f"multi_horizon_{h}", "realized_vol")

    return labels


__all__ = ["triple_barrier", "multi_horizon_labels"]
