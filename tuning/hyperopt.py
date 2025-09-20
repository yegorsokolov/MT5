"""Hyperparameter search using Optuna."""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path

import optuna
from analytics import mlflow_client as mlflow


def _study(storage: Path, name: str) -> optuna.Study:
    """Create or load an Optuna study backed by SQLite."""
    storage_url = f"sqlite:///{storage}"
    return optuna.create_study(
        study_name=name,
        direction="maximize",
        storage=storage_url,
        load_if_exists=True,
    )


@contextmanager
def _mlflow_run(cfg: dict, run_name: str = "tuning") -> None:
    """Context manager to ensure MLflow runs are always closed."""

    mlflow.start_run(run_name, cfg)
    try:
        yield
    finally:
        mlflow.end_run()


def tune_lgbm(cfg: dict, n_trials: int = 20) -> None:
    """Tune LightGBM parameters used by ``train.py``.

    Parameters
    ----------
    cfg:
        Base configuration.
    n_trials:
        Number of Optuna trials to run.
    """

    from train import main as train_main

    storage = Path(__file__).resolve().with_name("lgbm_tuning.db")
    study = _study(storage, "lgbm")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.loguniform("learning_rate", 1e-4, 2e-1),
            "num_leaves": trial.suggest_int("num_leaves", 16, 255),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
        }
        cfg_trial = deepcopy(cfg)
        cfg_trial.update(params)
        return float(train_main(cfg_trial))

    with _mlflow_run(cfg):
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_f1", study.best_value)


def tune_transformer(cfg: dict, n_trials: int = 20) -> None:
    """Tune Transformer parameters used by ``train_nn.py``."""

    from train_nn import launch as nn_launch

    storage = Path(__file__).resolve().with_name("nn_tuning.db")
    study = _study(storage, "transformer")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "learning_rate": trial.loguniform("learning_rate", 1e-5, 1e-2),
            "num_layers": trial.suggest_int("num_layers", 1, 4),
            "d_model": trial.suggest_int("d_model", 32, 256),
        }
        cfg_trial = deepcopy(cfg)
        cfg_trial.update(params)
        cfg_trial["ddp"] = False
        return float(nn_launch(cfg_trial))

    with _mlflow_run(cfg):
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_accuracy", study.best_value)


def tune_rl(cfg: dict, n_trials: int = 20) -> None:
    """Tune reinforcement learning parameters used by ``train_rl.py``."""

    from train_rl import launch as rl_launch

    storage = Path(__file__).resolve().with_name("rl_tuning.db")
    study = _study(storage, "rl")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "rl_learning_rate": trial.loguniform("rl_learning_rate", 1e-5, 1e-2),
            "rl_gamma": trial.suggest_float("rl_gamma", 0.90, 0.999),
        }
        cfg_trial = deepcopy(cfg)
        cfg_trial.update(params)
        cfg_trial["ddp"] = False
        return float(rl_launch(cfg_trial))

    with _mlflow_run(cfg):
        study.optimize(objective, n_trials=n_trials)
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_return", study.best_value)


__all__ = ["tune_lgbm", "tune_transformer", "tune_rl"]
