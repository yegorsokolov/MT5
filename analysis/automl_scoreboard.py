from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Mapping

import numpy as np
import pandas as pd

Strategy = Callable[[np.ndarray], np.ndarray]


def sharpe_ratio(returns: np.ndarray) -> float:
    """Compute the Sharpe ratio of a return series."""
    return float(np.mean(returns) / (np.std(returns) + 1e-9))


def max_drawdown(returns: np.ndarray) -> float:
    cumulative = (1 + returns).cumprod()
    return float((np.maximum.accumulate(cumulative) - cumulative).max())


def cross_validated_metrics(
    strategy: Strategy, data: np.ndarray, n_splits: int = 3
) -> Dict[str, float]:
    """Evaluate ``strategy`` on ``data`` using simple cross-validation."""
    length = len(data)
    fold = max(1, length // n_splits)
    sharpe_scores = []
    drawdowns = []
    for i in range(n_splits):
        start = i * fold
        end = length if i == n_splits - 1 else (i + 1) * fold
        returns = strategy(data[start:end])
        sharpe_scores.append(sharpe_ratio(returns))
        drawdowns.append(max_drawdown(returns))
    return {
        "sharpe": float(np.mean(sharpe_scores)),
        "drawdown": float(np.mean(drawdowns)),
    }


def build_scoreboard(
    strategies: Mapping[str, Strategy],
    regimes: Mapping[int, np.ndarray],
    path: str | Path = "reports/scoreboard.parquet",
    n_splits: int = 3,
) -> pd.DataFrame:
    """Run cross-validated backtests for ``strategies`` over ``regimes``."""
    records = []
    for regime, data in regimes.items():
        for name, strategy in strategies.items():
            metrics = cross_validated_metrics(strategy, np.asarray(data), n_splits)
            records.append({"regime": regime, "algorithm": name, **metrics})
    df = pd.DataFrame(records).set_index(["regime", "algorithm"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def load_scoreboard(path: str | Path = "reports/scoreboard.parquet") -> pd.DataFrame:
    """Load a previously persisted scoreboard."""
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(
            columns=["sharpe", "drawdown"],
            index=pd.MultiIndex.from_tuples([], names=["regime", "algorithm"]),
        )
    return pd.read_parquet(path)


__all__ = ["build_scoreboard", "load_scoreboard"]
