"""Risk of ruin estimation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_of_ruin(
    returns: pd.Series, equity: float, simulations: int = 10000
) -> float:
    """Estimate the probability that ``equity`` falls to zero.

    Parameters
    ----------
    returns : pd.Series
        Historical returns of the strategy expressed as decimal returns
        (e.g. 0.01 for 1%). The sequence is treated as the empirical
        distribution for bootstrapping.
    equity : float
        Current portfolio equity.  Simulated paths start from this value.
    simulations : int, default 10000
        Number of bootstrap paths to simulate.

    Returns
    -------
    float
        Estimated probability that equity drops to zero or below at any
        point along the simulated paths.
    """
    if returns.empty:
        return 0.0
    if equity <= 0:
        return 1.0

    # Remove NaNs to avoid propagation in simulations
    sample = returns.dropna().to_numpy()
    if sample.size == 0:
        return 0.0

    n = sample.size
    rng = np.random.default_rng(42)  # deterministic for reproducibility
    ruin_count = 0
    for _ in range(simulations):
        path_equity = equity
        path = rng.choice(sample, size=n, replace=True)
        for r in path:
            path_equity *= 1 + r
            if path_equity <= 0:
                ruin_count += 1
                break
    return ruin_count / simulations
