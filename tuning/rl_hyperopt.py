from __future__ import annotations

"""Hyperparameter optimisation for reinforcement learning policies."""

from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from analysis import regime_detection
from data.history import load_history_config
from data.features import make_features
from models import model_store
from models.hot_reload import hot_reload
from ray_utils import ray, init as ray_init, shutdown as ray_shutdown
from train_rl import launch as rl_launch

# Access Ray Tune via the imported/stubbed Ray module
_tune = ray.tune
OptunaSearch = _tune.search.optuna.OptunaSearch

_BEST_SCORE: float = float("-inf")
_BEST_PARAMS: Dict[str, float] = {}
_REGIME: int = 0


def _current_regime(cfg: Dict[str, Any]) -> int:
    """Detect the latest market regime from historical data."""
    try:
        root = Path(__file__).resolve().parents[1]
        symbols = cfg.get("symbols") or [cfg.get("symbol")]
        dfs = []
        for sym in symbols:
            df_sym = load_history_config(sym, cfg, root, validate=cfg.get("validate", False))
            df_sym["Symbol"] = sym
            dfs.append(df_sym)
        df = pd.concat(dfs, ignore_index=True)
        try:
            df = make_features(df, validate=cfg.get("validate", False))
        except TypeError:  # pragma: no cover - stubbed version
            df = make_features(df)
        labels = regime_detection.detect_regimes(df.tail(200))
        return int(labels.iloc[-1]) if not labels.empty else 0
    except Exception:  # pragma: no cover - if dependencies missing
        return 0


def _trainable(config: Dict[str, float], base_cfg: Dict[str, Any]) -> None:
    """Ray Tune trainable that evaluates a single configuration."""
    global _BEST_SCORE, _BEST_PARAMS
    trial_cfg = deepcopy(base_cfg)
    trial_cfg.update(
        {
            "rl_learning_rate": config["lr"],
            "rl_entropy_coef": config["entropy"],
            "rl_gamma": config["discount"],
        }
    )
    score = float(rl_launch(trial_cfg))
    _tune.report(score=score)
    if score > _BEST_SCORE:
        _BEST_SCORE = score
        _BEST_PARAMS = {
            "rl_learning_rate": config["lr"],
            "rl_entropy_coef": config["entropy"],
            "rl_gamma": config["discount"],
        }
        model_store.save_model({}, _BEST_PARAMS, {"val_score": score, "regime": _REGIME})
        hot_reload(_BEST_PARAMS)


def tune_rl(cfg: Dict[str, Any], n_trials: int = 20) -> Dict[str, float]:
    """Run hyperparameter search for RL settings.

    Parameters
    ----------
    cfg:
        Base training configuration.
    n_trials:
        Number of hyperparameter combinations to evaluate.

    Returns
    -------
    Dict[str, float]
        Best hyperparameter combination discovered.
    """
    global _BEST_SCORE, _BEST_PARAMS, _REGIME
    _BEST_SCORE = float("-inf")
    _BEST_PARAMS = {}
    _REGIME = _current_regime(cfg)
    ray_init()
    search_space = {
        "lr": _tune.loguniform(1e-5, 1e-2),
        "entropy": _tune.uniform(0.0, 0.02),
        "discount": _tune.uniform(0.90, 0.999),
    }
    algo = OptunaSearch(metric="score", mode="max")
    _tune.run(
        _tune.with_parameters(_trainable, base_cfg=cfg),
        config=search_space,
        num_samples=n_trials,
        search_alg=algo,
    )
    ray_shutdown()
    return _BEST_PARAMS


__all__ = ["tune_rl"]
