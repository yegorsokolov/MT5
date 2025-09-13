"""Strategy benchmarking utilities.

This module provides a :func:`run_benchmark` helper that executes a strategy
across multiple datasets and risk profiles, aggregating common performance
metrics like Sharpe ratio, Conditional Value at Risk (CVaR) and portfolio
turnover.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import pandas as pd


@dataclass
class BenchmarkResult:
    """Container for benchmark metrics."""

    dataset: str
    risk_profile: str
    sharpe: float
    cvar: float
    turnover: float


# default annualization factor for daily returns
_ANN_FACTOR = np.sqrt(252)


def _sharpe_ratio(returns: pd.Series) -> float:
    """Compute the annualized Sharpe ratio for a return series."""
    if returns.std(ddof=0) == 0:
        return 0.0
    return float(_ANN_FACTOR * returns.mean() / returns.std(ddof=0))


def _cvar(returns: pd.Series, level: float = 0.95) -> float:
    """Compute Conditional Value at Risk at the given confidence level."""
    if returns.empty:
        return 0.0
    var = returns.quantile(1 - level)
    tail = returns[returns <= var]
    if tail.empty:
        return float(var)
    return float(tail.mean())


def _turnover(positions: pd.Series) -> float:
    """Compute turnover from a position series."""
    if positions.empty:
        return 0.0
    return float(positions.diff().abs().sum())


def run_benchmark(
    strategy: Any,
    datasets: Mapping[str, pd.DataFrame],
    risk_profiles: Iterable[Any],
) -> pd.DataFrame:
    """Run backtests and aggregate metrics for a strategy.

    Parameters
    ----------
    strategy:
        Object exposing a ``backtest`` method returning a dataframe with
        ``returns`` and ``position`` columns.
    datasets:
        Mapping of dataset name to dataframe containing market data.
    risk_profiles:
        Iterable of risk profile configurations passed directly to the
        strategy's ``backtest`` method.

    Returns
    -------
    pd.DataFrame
        Aggregated metrics for each dataset and risk profile combination.
    """
    results: list[BenchmarkResult] = []
    for ds_name, df in datasets.items():
        for rp in risk_profiles:
            bt_df = strategy.backtest(df, rp)
            returns = bt_df["returns"]
            positions = bt_df.get("position", pd.Series(dtype=float))
            result = BenchmarkResult(
                dataset=ds_name,
                risk_profile=getattr(rp, "name", str(rp)),
                sharpe=_sharpe_ratio(returns),
                cvar=_cvar(returns),
                turnover=_turnover(positions),
            )
            results.append(result)
    return pd.DataFrame([r.__dict__ for r in results])
