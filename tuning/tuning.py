from __future__ import annotations

from typing import Callable, Dict, Any

import optuna
from analytics import mlflow_client as mlflow

TrainFn = Callable[[Dict[str, Any], optuna.trial.Trial], float]


def tune_lightgbm(train_fn: TrainFn, *, n_trials: int = 20) -> Dict[str, Any]:
    """Run an Optuna study optimising common LightGBM parameters.

    Parameters
    ----------
    train_fn:
        Callable receiving ``(params, trial)`` and returning the objective
        value for the given parameters.
    n_trials:
        Number of trials to evaluate.

    Returns
    -------
    Dict[str, Any]
        Best set of parameters discovered.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "learning_rate": trial.suggest_float(
                "learning_rate", 1e-3, 3e-1, log=True
            ),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }
        score = float(train_fn(params, trial))
        # Log parameters and score for this trial
        mlflow.log_params({f"trial_{trial.number}_{k}": v for k, v in params.items()})
        mlflow.log_metric("tuning_score", score, step=trial.number)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
    return study.best_params


__all__ = ["tune_lightgbm"]
