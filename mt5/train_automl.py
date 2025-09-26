"""Train a FLAML AutoML ensemble for MT5 signals."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from flaml import AutoML
from joblib import dump
from sklearn import metrics

from utils import load_config
from data.history import save_history_parquet, load_history_config
from data.features import make_features, train_test_split
from mt5.log_utils import setup_logging, log_exceptions
from .train_tabular import _build_feature_list, _prepare_datasets, _json_default

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for AutoML training."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


def _serialise_history(automl: AutoML) -> list[dict[str, Any]]:
    """Return a JSON-serialisable representation of the FLAML model history."""

    history: list[dict[str, Any]] = []
    for key, info in (automl.model_history or {}).items():
        estimator = None
        trial_index = None
        if isinstance(key, tuple) and key:
            estimator = key[0]
            if len(key) > 1:
                trial_index = key[1]
        elif isinstance(key, str):
            estimator = key
        record: dict[str, Any] = {
            "estimator": estimator,
            "trial": trial_index,
        }
        if isinstance(info, dict):
            for name in ("train_time", "pred_time", "val_loss", "config", "metric_for_logging"):
                if name in info:
                    value = info[name]
                    if isinstance(value, (np.floating, np.integer)):
                        value = float(value)
                    record[name] = value
        history.append(record)
    history.sort(key=lambda item: item.get("val_loss", float("inf")))
    return history


def _collect_metrics(
    y_true: pd.Series,
    y_pred: Iterable[Any],
    y_proba: np.ndarray | None,
) -> dict[str, float | None]:
    """Compute evaluation metrics for the AutoML predictor."""

    scores: dict[str, float | None] = {}
    scores["accuracy"] = float(metrics.accuracy_score(y_true, y_pred))
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    scores["precision"] = float(precision)
    scores["recall"] = float(recall)
    scores["f1"] = float(f1)
    try:
        scores["balanced_accuracy"] = float(
            metrics.balanced_accuracy_score(y_true, y_pred)
        )
    except Exception:  # pragma: no cover - defensive guard
        scores["balanced_accuracy"] = None
    if y_proba is not None:
        try:
            scores["roc_auc"] = float(metrics.roc_auc_score(y_true, y_proba[:, 1]))
        except Exception:  # pragma: no cover - probability may be degenerate
            scores["roc_auc"] = None
        try:
            scores["log_loss"] = float(metrics.log_loss(y_true, y_proba))
        except Exception:  # pragma: no cover - guard against invalid values
            scores["log_loss"] = None
    else:
        scores["roc_auc"] = None
        scores["log_loss"] = None
    return scores


logger = logging.getLogger(__name__)


@log_exceptions
def main() -> None:
    init_logging()
    cfg = load_config()
    seed = int(cfg.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)

    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs: list[pd.DataFrame] = []
    for sym in symbols:
        df_sym = load_history_config(sym, cfg, root, validate=cfg.get("validate", False))
        df_sym["Symbol"] = sym
        dfs.append(df_sym)

    df = pd.concat(dfs, ignore_index=True)
    save_history_parquet(df, root / "data" / "history.parquet")

    df = make_features(df, validate=cfg.get("validate", False))
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

    train_df, test_df = train_test_split(df, cfg.get("train_rows", len(df) // 2))

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["target"] = (train_df["return"].shift(-1) > 0).astype(int)
    test_df["target"] = (test_df["return"].shift(-1) > 0).astype(int)
    train_df = train_df.dropna(subset=["target"])
    test_df = test_df.dropna(subset=["target"])

    features = _build_feature_list(df)
    X_train, y_train, X_test, y_test = _prepare_datasets(train_df, test_df, features)

    automl = AutoML()

    out_path = root / "models" / "automl"
    out_path.mkdir(parents=True, exist_ok=True)

    metric = cfg.get("automl_metric") or cfg.get("autogluon_eval_metric") or "accuracy"
    time_budget = (
        cfg.get("automl_time_budget")
        or cfg.get("automl_time_limit")
        or cfg.get("autogluon_time_limit")
    )
    estimators = cfg.get("automl_estimators")
    n_jobs = cfg.get("automl_n_jobs")
    eval_method = cfg.get("automl_eval_method")
    user_settings = cfg.get("automl_settings")

    fit_settings: dict[str, Any] = {
        "task": "classification",
        "metric": metric,
        "time_budget": time_budget,
        "estimator_list": estimators,
        "log_file_name": str(out_path / "flaml.log"),
        "seed": seed,
        "n_jobs": n_jobs,
        "eval_method": eval_method,
    }
    if isinstance(user_settings, dict):
        fit_settings.update({k: v for k, v in user_settings.items() if v is not None})

    fit_settings = {k: v for k, v in fit_settings.items() if v is not None}

    logger.info("Starting FLAML AutoML training with metric=%s", metric)
    automl.fit(X_train=X_train, y_train=y_train, **fit_settings)

    y_pred = automl.predict(X_test)
    y_proba = None
    try:
        y_proba = automl.predict_proba(X_test)
    except Exception:  # pragma: no cover - some estimators lack predict_proba
        logger.debug("AutoML estimator does not expose predict_proba()")

    scores = _collect_metrics(y_test, y_pred, y_proba)
    history = _serialise_history(automl)

    logger.info("AutoML metrics: %s", scores)
    logger.info("Best AutoML estimator: %s", automl.best_estimator)

    dump(automl, out_path / "model.joblib")

    metadata: dict[str, Any] = {
        "features": X_train.columns.tolist(),
        "performance": {
            "metrics": scores,
            "leaderboard": history,
            "best_estimator": getattr(automl, "best_estimator", None),
            "best_config": getattr(automl, "best_config", None),
            "best_loss": getattr(automl, "best_loss", None),
        },
        "training_config": {
            "model_type": "automl",
            "seed": seed,
            "symbols": symbols,
            "train_rows": cfg.get("train_rows"),
            "automl_metric": metric,
            "automl_time_budget": time_budget,
            "automl_estimators": estimators,
            "automl_n_jobs": n_jobs,
            "automl_eval_method": eval_method,
            "automl_settings": user_settings,
        },
    }

    metadata_path = out_path / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=_json_default),
        encoding="utf-8",
    )

    feature_path = out_path / "features.json"
    feature_path.write_text(
        json.dumps(X_train.columns.tolist(), indent=2),
        encoding="utf-8",
    )

    if history:
        leaderboard_path = out_path / "leaderboard.json"
        leaderboard_path.write_text(
            json.dumps(history, indent=2, default=_json_default),
            encoding="utf-8",
        )

    logger.info("AutoML model saved to %s", out_path)


if __name__ == "__main__":
    main()


__all__ = ["main"]
