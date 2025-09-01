"""Distributed hyperparameter search utilities."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable, Mapping
import importlib.util

import optuna

# ---------------------------------------------------------------------------
# Lazy loading of ResourceMonitor to avoid heavy package imports during tests
# ---------------------------------------------------------------------------

def _load_monitor():
    """Return the global ``ResourceMonitor`` instance.

    The project defines :mod:`utils.resource_monitor` which normally lives inside
    the ``utils`` package. Importing the package pulls in optional dependencies
    (``yaml``, ``mlflow`` etc.) which may be unavailable in lightweight test
    environments. We attempt the regular import first and fall back to loading
    the module directly from its file path which has no third‑party requirements.
    """

    try:  # Try the standard import path
        from utils.resource_monitor import monitor  # type: ignore
        return monitor
    except Exception:  # pragma: no cover - fallback for minimal environments
        spec = importlib.util.spec_from_file_location(
            "resource_monitor",
            Path(__file__).resolve().parents[1] / "utils" / "resource_monitor.py",
        )
        rm_mod = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(rm_mod)  # type: ignore
        return rm_mod.monitor  # type: ignore


monitor = _load_monitor()


SearchSpace = Mapping[str, Callable[[optuna.Trial], Any]]


def _trial_concurrency(cpus_per_trial: int = 1, gpus_per_trial: int = 0) -> int:
    """Determine how many trials to run concurrently."""

    cap = monitor.capabilities
    usage = monitor.latest_usage.get("cpu", 0.0)
    avail_cpus = max(1, int(cap.cpus * (1 - usage / 100)))
    cpu_slots = max(1, avail_cpus // max(cpus_per_trial, 1))
    if gpus_per_trial > 0 and getattr(cap, "gpu_count", 0) > 0:
        gpu_slots = max(1, cap.gpu_count // gpus_per_trial)
        return max(1, min(cpu_slots, gpu_slots))
    return cpu_slots


def run_search(
    train_fn: Callable[[dict], float],
    base_cfg: dict,
    param_space: SearchSpace,
    *,
    n_trials: int = 20,
    study_name: str = "search",
    sampler: optuna.samplers.BaseSampler | None = None,
    cpus_per_trial: int = 1,
    gpus_per_trial: int = 0,
    storage: Path | None = None,
) -> tuple[dict, int]:
    """Execute distributed hyperparameter search.

    Returns ``(best_params, concurrency)`` where ``concurrency`` indicates how
    many trials were executed in parallel.
    """

    storage_path = storage or Path(__file__).resolve().with_name(f"{study_name}.db")
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        storage=f"sqlite:///{storage_path}",
        load_if_exists=True,
        sampler=sampler,
    )

    def objective(trial: optuna.Trial) -> float:
        params = {name: fn(trial) for name, fn in param_space.items()}
        cfg_trial = deepcopy(base_cfg)
        cfg_trial.update(params)
        return float(train_fn(cfg_trial))

    concurrency = _trial_concurrency(cpus_per_trial, gpus_per_trial)
    study.optimize(objective, n_trials=n_trials, n_jobs=concurrency)
    best_params = study.best_params

    try:  # pragma: no cover - model_store may be absent in tests
        from models import model_store

        model_store.save_tuned_params(best_params)
    except Exception:  # pragma: no cover - optional persistence
        pass

    return best_params, concurrency


def tune_lgbm(cfg: dict, n_trials: int = 20) -> dict:
    """Tune LightGBM parameters used by :mod:`train.py`."""

    from train import main as train_main

    space = {
        "learning_rate": lambda t: t.loguniform("learning_rate", 1e-4, 2e-1),
        "num_leaves": lambda t: t.suggest_int("num_leaves", 16, 255),
        "max_depth": lambda t: t.suggest_int("max_depth", 3, 12),
    }
    best, _ = run_search(
        train_main,
        cfg,
        space,
        n_trials=n_trials,
        study_name="lgbm_distributed",
        cpus_per_trial=1,
    )
    return best


def tune_transformer(cfg: dict, n_trials: int = 20) -> dict:
    """Tune Transformer parameters used by :mod:`train_nn.py`."""

    from train_nn import launch as nn_launch

    space = {
        "learning_rate": lambda t: t.loguniform("learning_rate", 1e-5, 1e-2),
        "num_layers": lambda t: t.suggest_int("num_layers", 1, 4),
        "d_model": lambda t: t.suggest_int("d_model", 32, 256),
    }
    best, _ = run_search(
        nn_launch,
        cfg,
        space,
        n_trials=n_trials,
        study_name="transformer_distributed",
        cpus_per_trial=1,
        gpus_per_trial=1 if monitor.capabilities.has_gpu else 0,
    )
    return best


def tune_rl(cfg: dict, n_trials: int = 20) -> dict:
    """Tune reinforcement learning parameters used by :mod:`train_rl.py`."""

    from train_rl import launch as rl_launch

    space = {
        "rl_learning_rate": lambda t: t.loguniform("rl_learning_rate", 1e-5, 1e-2),
        "rl_gamma": lambda t: t.suggest_float("rl_gamma", 0.90, 0.999),
    }
    best, _ = run_search(
        rl_launch,
        cfg,
        space,
        n_trials=n_trials,
        study_name="rl_distributed",
        cpus_per_trial=1,
        gpus_per_trial=1 if monitor.capabilities.has_gpu else 0,
    )
    return best


__all__ = ["run_search", "tune_lgbm", "tune_transformer", "tune_rl"]
