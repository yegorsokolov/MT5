"""Monte Carlo analysis utilities.

These helpers perform simple bootstrapping of return series to estimate
variance in strategy performance.  The implementation is intentionally
minimal so it can be executed quickly inside continuous integration
pipelines.
"""
from __future__ import annotations

from typing import Dict
import numpy as np
import pandas as pd


def bootstrap_sharpe(
    returns: pd.Series,
    n_iterations: int = 1000,
    sample_size: int | None = None,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a bootstrap distribution of Sharpe ratios.

    Parameters
    ----------
    returns: pd.Series
        Series of strategy returns.
    n_iterations: int
        Number of bootstrap samples to generate.
    sample_size: int | None
        Number of observations per sample.  Defaults to the length of the
        input series.
    seed: int | None
        Random seed for reproducibility.
    """

    rng = np.random.default_rng(seed)
    data = returns.to_numpy()
    size = sample_size or len(data)
    samples = rng.choice(data, size=(n_iterations, size), replace=True)
    mean = samples.mean(axis=1)
    std = samples.std(axis=1, ddof=0)
    sharpe = np.divide(mean, std, out=np.zeros_like(mean), where=std != 0)
    return sharpe


def analyze_variance(
    returns: pd.Series,
    n_iterations: int = 1000,
    seed: int | None = None,
) -> Dict[str, float]:
    """Evaluate variance of strategy Sharpe ratio via bootstrapping."""

    dist = bootstrap_sharpe(returns, n_iterations=n_iterations, seed=seed)
    return {
        "sharpe_mean": float(dist.mean()),
        "sharpe_std": float(dist.std(ddof=0)),
    }


def is_stable(
    returns: pd.Series,
    threshold: float,
    n_iterations: int = 1000,
    seed: int | None = None,
) -> bool:
    """Determine if a strategy's Sharpe ratio variance is below ``threshold``."""

    stats = analyze_variance(returns, n_iterations=n_iterations, seed=seed)
    return stats["sharpe_std"] < threshold
