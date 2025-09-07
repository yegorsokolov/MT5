"""Bayesian hyper-parameter search utilities using Optuna.

This module provides a thin wrapper around :mod:`optuna` to perform
Bayesian optimisation of common hyper-parameters. The search explores
learning rates, model depth, dropout rate and batch size. The best
configuration for a study is persisted via :func:`model_store.save_tuned_params`.
"""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Dict

import optuna

try:  # pragma: no cover - optional dependency in minimal environments
    from models import model_store
except Exception:  # pragma: no cover
    model_store = None  # type: ignore


TrainFn = Callable[[Dict[str, Any], optuna.trial.Trial], float]


def run_search(
    train_fn: TrainFn,
    base_cfg: Dict[str, Any] | None = None,
    *,
    n_trials: int = 20,
    direction: str = "maximize",
    store_dir: str | Path | None = None,
) -> Dict[str, Any]:
    """Run Bayesian hyper-parameter search.

    Parameters
    ----------
    train_fn:
        Callable receiving ``(cfg, trial)`` and returning the objective value.
        The callable should honour any hyper-parameters injected into ``cfg``
        and may utilise ``trial`` for reporting intermediate metrics in order
        to enable Optuna's pruning based early stopping.
    base_cfg:
        Starting configuration which is copied for every trial.
    n_trials:
        Number of trials to evaluate.
    direction:
        Optimisation direction ("maximize" or "minimize").
    store_dir:
        Optional model store directory used when persisting tuned parameters.
    """

    base_cfg = base_cfg or {}

    def objective(trial: optuna.Trial) -> float:
        cfg = deepcopy(base_cfg)
        # Sample common hyper-parameters
        cfg.update(
            {
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-5, 1e-1, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 2, 8),
                "dropout": trial.suggest_float("dropout", 0.0, 0.5),
                "batch_size": trial.suggest_categorical(
                    "batch_size", [16, 32, 64, 128]
                ),
            }
        )
        # Provide generic early stopping hints
        cfg.setdefault("early_stopping_rounds", 10)
        cfg.setdefault("patience", 3)
        return float(train_fn(cfg, trial))

    study = optuna.create_study(
        direction=direction, pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials)

    if model_store is not None:  # pragma: no cover - model_store optional in tests
        model_store.save_tuned_params(study.best_params, store_dir=store_dir)

    return study.best_params


__all__ = ["run_search"]
