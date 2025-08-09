"""Utilities for walk-forward analysis.

This module provides helpers to split a time series into rolling
train/validation windows and aggregate simple performance metrics.  It
is intentionally lightweight so it can be used inside CI checks to
validate the stability of a strategy before merging changes.
"""
from __future__ import annotations

from typing import Iterator, List, Tuple, Dict
import pandas as pd


def rolling_windows(
    data: pd.DataFrame,
    train_size: int,
    val_size: int,
    step: int,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Split ``data`` into rolling train/validation windows.

    Parameters
    ----------
    data:
        Time ordered ``DataFrame`` containing at least a ``return`` column.
    train_size:
        Number of rows to use for the training portion of each window.
    val_size:
        Number of rows to use for the validation portion of each window.
    step:
        How many rows to advance the window after each iteration.

    Returns
    -------
    list of tuple
        A list where each item contains the train and validation ``DataFrame``
        for a single window.
    """

    windows: List[Tuple[pd.DataFrame, pd.DataFrame]] = []
    start = 0
    end = train_size + val_size
    while end <= len(data):
        train = data.iloc[start : start + train_size]
        val = data.iloc[start + train_size : end]
        windows.append((train, val))
        start += step
        end = start + train_size + val_size
    return windows


def _sharpe_ratio(returns: pd.Series) -> float:
    mean = returns.mean()
    std = returns.std(ddof=0)
    if std == 0:
        return 0.0
    return float(mean / std)


def _max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative / peak) - 1.0
    return float(drawdown.min())


def aggregate_metrics(
    data: pd.DataFrame,
    train_size: int,
    val_size: int,
    step: int,
    return_col: str = "return",
) -> Dict[str, float]:
    """Run walk-forward evaluation and aggregate metrics.

    The Sharpe ratio is averaged across validation windows while the worst
    (minimum) drawdown is reported.
    """

    windows = rolling_windows(data, train_size, val_size, step)
    if not windows:
        return {"avg_sharpe": 0.0, "worst_drawdown": 0.0}

    sharpes = []
    drawdowns = []
    for _, val in windows:
        returns = val[return_col]
        sharpes.append(_sharpe_ratio(returns))
        drawdowns.append(_max_drawdown(returns))

    avg_sharpe = float(sum(sharpes) / len(sharpes))
    worst_drawdown = float(min(drawdowns))
    return {"avg_sharpe": avg_sharpe, "worst_drawdown": worst_drawdown}
