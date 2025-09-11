"""Monte Carlo generation of correlated fat-tailed return shocks."""
from __future__ import annotations

import numpy as np
from numpy.random import Generator, default_rng


def generate_correlated_shocks(
    cov: np.ndarray,
    n_steps: int,
    n_paths: int = 1,
    df: float = 5.0,
    rng: Generator | None = None,
) -> np.ndarray:
    """Draw samples from a multivariate Student's t distribution.

    Parameters
    ----------
    cov:
        Positive semi-definite covariance matrix of shape ``(d, d)``.
    n_steps:
        Number of time steps in each path.
    n_paths:
        Number of independent scenario paths to sample.
    df:
        Degrees of freedom controlling tail thickness. Lower values produce
        heavier tails.
    rng:
        Optional ``numpy.random.Generator`` for reproducibility.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_paths, n_steps, d)`` containing return shocks.
    """
    rng = default_rng() if rng is None else rng
    cov = np.asarray(cov, dtype=float)
    L = np.linalg.cholesky(cov)
    d = L.shape[0]
    g = rng.standard_normal((n_paths, n_steps, d))
    chi = rng.chisquare(df, size=(n_paths, n_steps, 1))
    shocks = (g @ L.T) / np.sqrt(chi / df)
    return shocks


__all__ = ["generate_correlated_shocks"]
