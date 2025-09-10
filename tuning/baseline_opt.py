from __future__ import annotations

"""Optuna search for baseline strategy parameters.

This module tunes moving-average periods, CVD confirmation thresholds and
ATR stop multipliers by backtesting :class:`~strategies.baseline.BaselineStrategy`
on historical data.
"""

from typing import Dict

import optuna
import pandas as pd

from backtesting.walk_forward import aggregate_metrics
from strategies.baseline import BaselineStrategy


def _simulate_returns(data: pd.DataFrame, params: Dict[str, float]) -> pd.Series:
    """Generate trade returns for ``params`` using historical ``data``."""

    strat = BaselineStrategy(
        short_window=int(params["short_window"]),
        long_window=int(params["long_window"]),
        atr_window=int(params["atr_window"]),
        atr_stop_long=float(params["stop_mult"]),
        atr_stop_short=float(params["stop_mult"]),
    )
    returns = []
    prev_price: float | None = None
    position = 0
    for row in data.itertuples():
        cvd = getattr(row, "cvd", None)
        if cvd is not None and abs(cvd) < params["cvd_threshold"]:
            cvd = None
        signal = strat.update(row.Close, high=row.High, low=row.Low, cvd=cvd)
        if prev_price is not None:
            ret = position * (row.Close - prev_price) / prev_price
            returns.append(ret)
        position += signal
        prev_price = row.Close
    return pd.Series(returns)


def backtest(params: Dict[str, float], data: pd.DataFrame) -> float:
    """Return average Sharpe ratio for ``params`` over ``data``."""

    trade_returns = _simulate_returns(data, params)
    df = pd.DataFrame({"return": trade_returns})
    metrics = aggregate_metrics(df, train_size=200, val_size=50, step=50)
    return float(metrics["avg_sharpe"])


def run_search(data: pd.DataFrame, *, n_trials: int = 30) -> Dict[str, float]:
    """Search for optimal baseline parameters on ``data``."""

    def objective(trial: optuna.Trial) -> float:
        params = {
            "short_window": trial.suggest_int("short_window", 2, 20),
            "long_window": trial.suggest_int("long_window", 20, 100),
            "atr_window": trial.suggest_int("atr_window", 5, 50),
            "cvd_threshold": trial.suggest_float("cvd_threshold", 0.0, 100.0),
            "stop_mult": trial.suggest_float("stop_mult", 1.0, 5.0),
        }
        return backtest(params, data)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


__all__ = ["run_search", "backtest"]
