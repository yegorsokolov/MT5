"""Correlation-weighted cross-asset confirmation scores.

This module computes a simple confirmation metric by combining
correlations and latest returns from related instruments or
indices. The resulting score can be used to gate trade entries and
exits. A positive score indicates that related markets support the
trade direction while a negative score suggests divergence.

The implementation intentionally keeps dependencies light so unit tests
can exercise the behaviour using synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


def _returns(series: pd.Series) -> pd.Series:
    """Return percentage changes of ``series``."""

    return series.pct_change().fillna(0.0)


def compute_score(
    prices: pd.DataFrame,
    target: str,
    related: Iterable[str],
    window: int = 30,
) -> float:
    """Return correlation weighted confirmation score.

    Parameters
    ----------
    prices:
        DataFrame indexed by time with a column per symbol containing mid or
        close prices.
    target:
        Symbol to score.
    related:
        Iterable of related symbols or indices.
    window:
        Rolling window used when computing correlations. Defaults to ``30``.
    """

    if target not in prices.columns:
        raise ValueError(f"target {target} not in prices")
    score = 0.0
    target_ret = _returns(prices[target])
    for sym in related:
        if sym not in prices.columns:
            continue
        rel_ret = _returns(prices[sym])
        corr = target_ret.rolling(window).corr(rel_ret).iloc[-1]
        if np.isnan(corr):
            continue
        score += corr * np.sign(rel_ret.iloc[-1])
    return float(score)


def should_open(score: float) -> bool:
    """Return ``True`` when ``score`` supports opening a trade."""

    return score > 0.0


def should_close(score: float) -> bool:
    """Return ``True`` when ``score`` supports closing a trade."""

    return score < 0.0


@dataclass
class ConfirmationResult:
    """Container holding a score and individual correlations."""

    score: float
    correlations: Dict[str, float]


def detailed_score(
    prices: pd.DataFrame,
    target: str,
    related: Iterable[str],
    window: int = 30,
) -> ConfirmationResult:
    """Return both score and per-symbol correlations."""

    corrs: Dict[str, float] = {}
    target_ret = _returns(prices[target])
    score = 0.0
    for sym in related:
        if sym not in prices.columns:
            continue
        rel_ret = _returns(prices[sym])
        corr = target_ret.rolling(window).corr(rel_ret).iloc[-1]
        if np.isnan(corr):
            continue
        corrs[sym] = float(corr)
        score += corr * np.sign(rel_ret.iloc[-1])
    return ConfirmationResult(float(score), corrs)


__all__ = [
    "compute_score",
    "detailed_score",
    "should_open",
    "should_close",
    "ConfirmationResult",
]
