from __future__ import annotations

"""Automatic hyperparameter optimisation triggered by regime shifts or time."""

import time
from typing import Any, Callable, Dict, Optional

import optuna
import pandas as pd

from analysis import regime_detection
from models import model_store
from models.hot_reload import hot_reload


class AutoOptimizer:
    """Periodically optimises model parameters using Optuna.

    Optimisation is triggered when the detected market regime changes or when a
    fixed time interval has elapsed since the last optimisation. The previous
    best parameters are used as warm-start trials.
    """

    def __init__(
        self,
        objective_fn: Callable[[Dict[str, Any], pd.DataFrame], float],
        *,
        n_trials: int = 10,
        interval: int = 3600,
    ) -> None:
        self.objective_fn = objective_fn
        self.n_trials = n_trials
        self.interval = interval
        self.last_run: float = 0.0
        self.last_regime: Optional[int] = None
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: float = float("-inf")

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

    def maybe_optimize(self, data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        run, regime = self.should_run(data)
        if not run:
            return None

        study = optuna.create_study(direction="maximize")
        if self.best_params:
            study.enqueue_trial(self.best_params)

        def objective(trial: optuna.Trial) -> float:
            params = {"lr": trial.suggest_float("lr", 1e-5, 1e-1)}
            return self.objective_fn(params, data)

        study.optimize(objective, n_trials=self.n_trials)
        params = study.best_params
        score = float(study.best_value)

        self.last_run = time.time()
        self.last_regime = regime

        if score > self.best_score:
            self.best_score = score
            self.best_params = params
            model_id = model_store.save_model(
                {}, params, {"val_score": score, "regime": regime}
            )
            hot_reload(params)
            return params
        return None


__all__ = ["AutoOptimizer"]
