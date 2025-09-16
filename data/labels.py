from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

try:  # pragma: no cover - optional dependency
    from numba import njit
    _HAS_NUMBA = True
except Exception:  # pragma: no cover
    _HAS_NUMBA = False

    def njit(*args, **kwargs):  # type: ignore
        def wrap(func):
            return func

        return wrap

from analysis.data_lineage import log_lineage


@njit(cache=True)
def _compute_barrier_hits_numba(
    prices: np.ndarray, pt_mult: float, sl_mult: float, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    n = prices.shape[0]
    no_hit = horizon + 1
    upper_hits = np.full(n, no_hit, dtype=np.int64)
    lower_hits = np.full(n, no_hit, dtype=np.int64)

    if horizon <= 0:
        return upper_hits, lower_hits

    pt_levels = prices * (1.0 + pt_mult)
    sl_levels = prices * (1.0 - sl_mult)

    running_max = np.empty(n, dtype=prices.dtype)
    running_min = np.empty(n, dtype=prices.dtype)
    running_max[:] = -np.inf
    running_min[:] = np.inf

    for step in range(1, horizon + 1):
        limit = n - step
        if limit <= 0:
            break

        for i in range(limit):
            future_price = prices[i + step]
            if future_price > running_max[i]:
                running_max[i] = future_price
            if future_price < running_min[i]:
                running_min[i] = future_price

        for i in range(limit):
            if upper_hits[i] == no_hit and running_max[i] >= pt_levels[i]:
                upper_hits[i] = step
            if lower_hits[i] == no_hit and running_min[i] <= sl_levels[i]:
                lower_hits[i] = step

    return upper_hits, lower_hits


def _compute_barrier_hits_numpy(
    prices: np.ndarray, pt_mult: float, sl_mult: float, horizon: int
) -> tuple[np.ndarray, np.ndarray]:
    if horizon <= 0:
        return (
            np.full(prices.shape[0], horizon + 1, dtype=np.int64),
            np.full(prices.shape[0], horizon + 1, dtype=np.int64),
        )

    padded = np.pad(prices, (0, horizon), constant_values=prices[-1])
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
    return first_upper.astype(np.int64), first_lower.astype(np.int64)


def triple_barrier(
    prices: "pd.Series", pt_mult: float, sl_mult: float, max_horizon: int
) -> "pd.Series":
    """Generate triple barrier labels using cumulative extrema arrays.

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

    Notes
    -----
    The implementation maintains arrays of cumulative maxima and minima of
    forward prices.  When :mod:`numba` is available the cumulative updates are
    executed inside a compiled kernel, avoiding Python loops entirely during
    runtime.  A NumPy fallback performs the same cumulative-extrema logic using
    vectorised operations, ensuring identical label semantics regardless of the
    execution path.
    """

    arr = np.asarray(prices.to_numpy(), dtype=np.float64)
    n = len(arr)
    horizon = min(max_horizon, n - 1)
    if horizon <= 0:
        labels = np.zeros(n, dtype=np.int8)
    else:
        if _HAS_NUMBA:
            first_upper, first_lower = _compute_barrier_hits_numba(arr, float(pt_mult), float(sl_mult), horizon)
        else:
            first_upper, first_lower = _compute_barrier_hits_numpy(arr, float(pt_mult), float(sl_mult), horizon)

        labels = np.zeros(n, dtype=np.int8)
        labels[first_upper < first_lower] = 1
        labels[first_lower < first_upper] = -1

    labels = pd.Series(labels, index=prices.index, dtype=int)

    run_id = prices.attrs.get("run_id", "unknown")
    raw_file = prices.attrs.get("source", "unknown")
    log_lineage(run_id, raw_file, "triple_barrier", "label")
    return labels


def multi_horizon_labels(prices: "pd.Series", horizons: list[int]) -> "pd.DataFrame":
    """Generate direction, return and volatility targets for multiple horizons.

    For each horizon ``h`` three targets are produced:

    ``direction_{h}``
        Binary direction label indicating whether ``p(t+h) > p(t)``.

    ``abs_return_{h}``
        Absolute percentage return ``|p(t+h) / p(t) - 1|``.

    ``volatility_{h}``
        Realised volatility over the next ``h`` forward returns computed as
        ``sqrt(sum(r_i^2))`` where ``r_i`` are percentage changes.

    The final ``h`` rows—where future information is unavailable—are filled
    with zeros so that the targets align with the original ``prices`` index.
    """

    if prices.empty:
        return pd.DataFrame(index=prices.index)

    seen: set[int] = set()
    valid_horizons: list[int] = []
    for h in horizons:
        step = int(h)
        if step <= 0 or step in seen:
            continue
        seen.add(step)
        valid_horizons.append(step)

    if not valid_horizons:
        return pd.DataFrame(index=prices.index)

    arr = prices.to_numpy(dtype=np.float64)
    n = len(arr)
    forward_ret = np.zeros(max(n - 1, 0), dtype=np.float64)
    if n > 1:
        base = arr[:-1]
        next_vals = arr[1:]
        with np.errstate(divide="ignore", invalid="ignore"):
            forward = next_vals / base - 1.0
        forward[~np.isfinite(forward)] = 0.0
        forward_ret[: len(forward)] = forward
    sq_forward = forward_ret**2

    data: dict[str, pd.Series] = {}
    for horizon in valid_horizons:
        direction = np.zeros(n, dtype=np.int8)
        abs_return = np.zeros(n, dtype=np.float32)
        volatility = np.zeros(n, dtype=np.float32)

        if horizon < n:
            current = arr[:-horizon]
            future = arr[horizon:]
            with np.errstate(divide="ignore", invalid="ignore"):
                rel = future / current - 1.0
            rel[~np.isfinite(rel)] = 0.0
            direction[:-horizon] = (rel > 0).astype(np.int8)
            abs_return[:-horizon] = np.abs(rel).astype(np.float32)

        if horizon <= len(sq_forward) and len(sq_forward) > 0:
            windows = sliding_window_view(sq_forward, horizon)
            summed = windows.sum(axis=1, dtype=np.float64)
            volatility[:-horizon] = np.sqrt(summed).astype(np.float32)

        data[f"direction_{horizon}"] = pd.Series(
            direction, index=prices.index, dtype="int8"
        )
        data[f"abs_return_{horizon}"] = pd.Series(
            abs_return, index=prices.index, dtype="float32"
        )
        data[f"volatility_{horizon}"] = pd.Series(
            volatility, index=prices.index, dtype="float32"
        )

    labels = pd.DataFrame(data, index=prices.index)

    run_id = prices.attrs.get("run_id", "unknown")
    raw_file = prices.attrs.get("source", "unknown")
    for horizon in valid_horizons:
        log_lineage(run_id, raw_file, f"direction_{horizon}", "direction")
        log_lineage(run_id, raw_file, f"abs_return_{horizon}", "abs_return")
        log_lineage(run_id, raw_file, f"volatility_{horizon}", "realized_vol")

    return labels


__all__ = ["triple_barrier", "multi_horizon_labels"]
