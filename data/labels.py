from __future__ import annotations

import pandas as pd
from analysis.data_lineage import log_lineage


def triple_barrier(
    prices: 'pd.Series', pt_mult: float, sl_mult: float, max_horizon: int
) -> 'pd.Series':
    """Generate triple barrier labels.

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
    labels = pd.Series(0, index=prices.index, dtype=int)
    n = len(prices)
    for i in range(n):
        p0 = prices.iloc[i]
        upper = p0 * (1 + pt_mult)
        lower = p0 * (1 - sl_mult)
        end = min(i + max_horizon, n - 1)
        outcome = 0
        for j in range(i + 1, end + 1):
            p = prices.iloc[j]
            if p >= upper:
                outcome = 1
                break
            if p <= lower:
                outcome = -1
                break
        labels.iloc[i] = outcome

    run_id = prices.attrs.get("run_id", "unknown")
    raw_file = prices.attrs.get("source", "unknown")
    log_lineage(run_id, raw_file, "triple_barrier", "label")
    return labels

__all__ = ["triple_barrier"]
