from __future__ import annotations

"""ELO based rating for trading algorithms.

This module maintains ELO scores for algorithms based on realised profit and
loss from live trading sessions.  After each session the mapping of algorithm
name to realised PnL is supplied to :func:`update_ratings` which performs
pairwise comparisons between algorithms and updates their ratings accordingly.
Ratings are persisted to ``reports/elo_ratings.parquet`` by default so they can
seed other components such as the :class:`strategy.router.StrategyRouter`.
"""

from pathlib import Path
from typing import Mapping

import pandas as pd

DEFAULT_RATING = 1500.0
K_FACTOR = 32.0


def load_ratings(path: str | Path = "reports/elo_ratings.parquet") -> pd.Series:
    """Load existing ELO ratings from ``path``.

    Parameters
    ----------
    path:
        Location of the parquet file.  If it does not exist an empty series is
        returned.
    """
    p = Path(path)
    if not p.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(p)
    if "algorithm" in df.columns:
        df = df.set_index("algorithm")
    return df["rating"].astype(float)


def save_ratings(
    ratings: pd.Series, path: str | Path = "reports/elo_ratings.parquet"
) -> None:
    """Persist ratings to ``path`` in parquet format."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    df = ratings.rename("rating").to_frame()
    df.index.name = "algorithm"
    df.to_parquet(p)


def expected_score(r_a: float, r_b: float) -> float:
    """Return expected score for player ``a`` against ``b``."""
    return 1.0 / (1.0 + 10 ** ((r_b - r_a) / 400.0))


def update_ratings(
    pnls: Mapping[str, float],
    *,
    path: str | Path = "reports/elo_ratings.parquet",
    k_factor: float = K_FACTOR,
    default: float = DEFAULT_RATING,
) -> pd.Series:
    """Update ELO ratings given realised PnL for a trading session.

    Parameters
    ----------
    pnls:
        Mapping of algorithm name to realised profit and loss for the session.
    path:
        Location where ratings are stored.  Existing ratings are loaded before
        applying the update and saved afterwards.
    k_factor:
        ELO K-factor controlling update magnitude.
    default:
        Starting rating for previously unseen algorithms.

    Returns
    -------
    pandas.Series
        Updated ratings indexed by algorithm name.
    """

    if len(pnls) < 2:
        # Nothing to update if only one algorithm participated
        ratings = load_ratings(path)
        for name in pnls:
            ratings.loc[name] = ratings.get(name, default)
        save_ratings(ratings, path)
        return ratings

    ratings = load_ratings(path)
    for name in pnls:
        if name not in ratings:
            ratings.loc[name] = default

    names = list(pnls.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            r_a = ratings.loc[a]
            r_b = ratings.loc[b]
            pnl_a = pnls[a]
            pnl_b = pnls[b]
            if pnl_a == pnl_b:
                score_a = 0.5
                score_b = 0.5
            elif pnl_a > pnl_b:
                score_a = 1.0
                score_b = 0.0
            else:
                score_a = 0.0
                score_b = 1.0
            exp_a = expected_score(r_a, r_b)
            exp_b = 1.0 - exp_a
            ratings.loc[a] = r_a + k_factor * (score_a - exp_a)
            ratings.loc[b] = r_b + k_factor * (score_b - exp_b)

    save_ratings(ratings, path)
    return ratings


__all__ = ["update_ratings", "load_ratings", "save_ratings"]
