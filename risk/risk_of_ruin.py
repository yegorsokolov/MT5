"""Risk of ruin estimation utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def risk_of_ruin(returns: pd.Series, initial_capital: float, simulations: int = 10000) -> float:
    """Estimate the probability of portfolio ruin.

    Parameters
    ----------
    returns : pd.Series
        Historical returns of the strategy expressed as decimal returns
        (e.g. 0.01 for 1%). The sequence is treated as the empirical
        distribution for bootstrapping.
    initial_capital : float
        Starting capital.
    simulations : int, default 10000
        Number of bootstrap paths to simulate.

    Returns
    -------
    float
        Estimated probability that capital drops to zero or below at any
        point along the simulated paths.
    """
    if returns.empty or initial_capital <= 0:
        return 0.0

    # Remove NaNs to avoid propagation in simulations
    sample = returns.dropna().to_numpy()
    if sample.size == 0:
        return 0.0

    n = sample.size
    rng = np.random.default_rng(42)  # deterministic for reproducibility
    ruin_count = 0
    for _ in range(simulations):
        equity = initial_capital
        path = rng.choice(sample, size=n, replace=True)
        for r in path:
            equity *= 1 + r
            if equity <= 0:
                ruin_count += 1
                break
    return ruin_count / simulations
