from __future__ import annotations

"""Automated model and hyperparameter search using Optuna.

This module evaluates multiple model families (LightGBM, MultiHeadTransformer,
CrossAssetTransformer) across a cross-validation split and logs results to
MLflow. It exposes :func:`run_search` which accepts ``X`` and ``y`` arrays and
returns the best model configuration along with a per-fold performance summary.
"""

from typing import Any, Dict, Sequence, Tuple

import numpy as np
import optuna
import pandas as pd

from analytics import mlflow_client as mlflow
from models.multi_head import MultiHeadTransformer
from models.cross_asset_transformer import CrossAssetTransformer

try:  # LightGBM is optional in some environments
    import lightgbm as lgb
except Exception:  # pragma: no cover - optional dependency
    lgb = None  # type: ignore



def _kfold_split(n_samples: int, n_splits: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    rng.shuffle(indices)
    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[: n_samples % n_splits] += 1
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        yield train_idx, val_idx
        current = stop


# ---------------------------------------------------------------------------
# Model-specific training helpers
# ---------------------------------------------------------------------------

def _train_lightgbm(
    params: Dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> float:
    assert lgb is not None, "LightGBM is not installed"
    model = lgb.LGBMRegressor(
        num_leaves=int(params["num_leaves"]),
        learning_rate=float(params["learning_rate"]),
        max_depth=int(params["max_depth"]),
    )
    model.fit(X_tr, y_tr)
    preds = model.predict(X_va)
    return -float(np.mean((y_va - preds) ** 2))


def _train_multihead(
    params: Dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> float:
    import torch
    from torch import nn
    input_dim = X_tr.shape[1]
    model = MultiHeadTransformer(
        input_size=input_dim,
        num_symbols=1,
        d_model=int(params["d_model"]),
        nhead=int(params["nhead"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params.get("dropout", 0.1)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    X_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1)
    y_t = torch.tensor(y_tr, dtype=torch.float32)
    for _ in range(1):  # minimal epochs for speed
        opt.zero_grad()
        out = model(X_t, symbol=0)
        loss = nn.functional.mse_loss(out, y_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = model(torch.tensor(X_va, dtype=torch.float32).unsqueeze(1), symbol=0)
    return -float(np.mean((y_va - preds.numpy()) ** 2))


def _train_crossasset(
    params: Dict[str, Any],
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
) -> float:
    import torch
    from torch import nn
    input_dim = X_tr.shape[1]
    model = CrossAssetTransformer(
        input_dim=input_dim,
        n_symbols=1,
        d_model=int(params["d_model"]),
        nhead=int(params["nhead"]),
        num_layers=int(params["num_layers"]),
        dropout=float(params.get("dropout", 0.1)),
    )
    opt = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
    X_t = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1).unsqueeze(1)
    y_t = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    for _ in range(1):
        opt.zero_grad()
        out = model(X_t).squeeze(1)
        loss = nn.functional.mse_loss(out, y_t)
        loss.backward()
        opt.step()
    with torch.no_grad():
        preds = (
            model(torch.tensor(X_va, dtype=torch.float32).unsqueeze(1).unsqueeze(1))
            .squeeze(1)
            .numpy()
        )
    preds = preds.reshape(-1)
    return -float(np.mean((y_va - preds) ** 2))


# ---------------------------------------------------------------------------
# Search routine
# ---------------------------------------------------------------------------

def run_search(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_trials: int = 10,
    n_splits: int = 3,
    model_types: Sequence[str] | None = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Run hyperparameter search across multiple model families.

    Parameters
    ----------
    X, y:
        Training data.
    n_trials:
        Number of Optuna trials.
    n_splits:
        Number of cross-validation folds.
    model_types:
        Optional subset of model types to explore. Defaults to all available
        models.

    Returns
    -------
    Tuple[Dict[str, Any], DataFrame]
        Best model configuration and a per-fold score summary for that model.
    """

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    model_types = list(model_types or ["lightgbm", "multihead", "crossasset"])
    n_samples = X.shape[0]

    def evaluate(model_type: str, params: Dict[str, Any]) -> float:
        scores = []
        for tr_idx, va_idx in _kfold_split(n_samples, n_splits):
            X_tr, X_va = X[tr_idx], X[va_idx]
            y_tr, y_va = y[tr_idx], y[va_idx]
            if model_type == "lightgbm":
                score = _train_lightgbm(params, X_tr, y_tr, X_va, y_va)
            elif model_type == "multihead":
                score = _train_multihead(params, X_tr, y_tr, X_va, y_va)
            else:
                score = _train_crossasset(params, X_tr, y_tr, X_va, y_va)
            scores.append(float(score))
        return float(np.mean(scores))

    def objective(trial: optuna.Trial) -> float:
        model_type = trial.suggest_categorical("model_type", model_types)
        if model_type == "lightgbm":
            params = {
                "num_leaves": trial.suggest_int("num_leaves", 16, 32),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 1e-2, 3e-1, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 6),
            }
        else:
            params = {
                "d_model": trial.suggest_int("d_model", 16, 64),
                "nhead": trial.suggest_int("nhead", 1, 2),
                "num_layers": trial.suggest_int("num_layers", 1, 2),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
            }
        score = evaluate(model_type, params)
        log_params = {f"trial_{trial.number}_{k}": v for k, v in params.items()}
        log_params[f"trial_{trial.number}_model_type"] = model_type
        mlflow.log_params(log_params)
        mlflow.log_metric("score", score, step=trial.number)
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    best_type = study.best_trial.params.get("model_type", "lightgbm")
    best_params = {k: v for k, v in study.best_trial.params.items() if k != "model_type"}
    mlflow.log_param("best_model_type", best_type)
    mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})

    summary_records = []
    for i, (tr_idx, va_idx) in enumerate(_kfold_split(n_samples, n_splits)):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]
        if best_type == "lightgbm":
            score = _train_lightgbm(best_params, X_tr, y_tr, X_va, y_va)
        elif best_type == "multihead":
            score = _train_multihead(best_params, X_tr, y_tr, X_va, y_va)
        else:
            score = _train_crossasset(best_params, X_tr, y_tr, X_va, y_va)
        mlflow.log_metric(f"best_fold_{i}", score)
        summary_records.append({"fold": i, "score": score})

    summary = pd.DataFrame(summary_records)
    best = {"model_type": best_type, "params": best_params, "score": study.best_value}
    return best, summary


__all__ = ["run_search"]
