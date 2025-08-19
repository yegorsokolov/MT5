from __future__ import annotations

"""Optimize entry/exit parameters via walk-forward backtests."""

import time
from typing import Dict, Optional

import optuna
import pandas as pd

from analysis import regime_detection
from backtesting.walk_forward import aggregate_metrics
from models import model_store
from models.hot_reload import hot_reload


class EntryExitOptimizer:
    """Optimise stop-loss, take-profit and holding period settings.

    Optimisation runs when the market regime changes or a fixed time interval
    elapses.  The best parameters per regime are persisted to ``model_store``
    and applied via hot-reload without interrupting trading.
    """

    def __init__(self, *, n_trials: int = 20, interval: int = 3600) -> None:
        self.n_trials = n_trials
        self.interval = interval
        self.last_run: float = 0.0
        self.last_regime: Optional[int] = None
        self.best_params: Dict[int, Dict[str, float]] = {}
        self.best_scores: Dict[int, float] = {}

    def _current_regime(self, data: pd.DataFrame) -> int:
        labels = regime_detection.detect_regimes(data.tail(200))
        return int(labels.iloc[-1]) if not labels.empty else 0

    def should_run(self, data: pd.DataFrame) -> tuple[bool, int]:
        regime = self._current_regime(data)
        now = time.time()
        trigger = (
            self.last_regime is None
            or regime != self.last_regime
            or now - self.last_run >= self.interval
        )
        return trigger, regime

    @staticmethod
    def _simulate_returns(returns: pd.Series, stop: float, profit: float, hold: int) -> pd.Series:
        """Generate trade returns under the given parameters."""
        out = []
        n = len(returns)
        for i in range(n):
            cum = 0.0
            for j in range(hold):
                if i + j >= n:
                    break
                cum += float(returns.iloc[i + j])
                if cum <= stop or cum >= profit:
                    break
            out.append(cum)
        return pd.Series(out)

    def _objective(self, params: Dict[str, float], data: pd.DataFrame) -> float:
        trade_returns = self._simulate_returns(
            data["return"], params["stop"], params["profit"], int(params["holding"])
        )
        df = pd.DataFrame({"return": trade_returns})
        metrics = aggregate_metrics(df, train_size=50, val_size=20, step=10)
        return float(metrics["avg_sharpe"])

    def maybe_optimize(self, data: pd.DataFrame) -> Optional[Dict[str, float]]:
        run, regime = self.should_run(data)
        if not run:
            return None

        study = optuna.create_study(direction="maximize")
        prev = self.best_params.get(regime)
        if prev:
            study.enqueue_trial(prev)

        def objective(trial: optuna.Trial) -> float:
            params = {
                "stop": trial.suggest_float("stop", -0.05, -0.001),
                "profit": trial.suggest_float("profit", 0.001, 0.05),
                "holding": trial.suggest_int("holding", 1, 20),
            }
            return self._objective(params, data)

        study.optimize(objective, n_trials=self.n_trials)
        params = study.best_params
        score = float(study.best_value)

        self.last_run = time.time()
        self.last_regime = regime

        if score > self.best_scores.get(regime, float("-inf")):
            self.best_scores[regime] = score
            self.best_params[regime] = params
            model_store.save_model({}, params, {"val_score": score, "regime": regime})
            hot_reload(params)
            return params
        return None


__all__ = ["EntryExitOptimizer"]
