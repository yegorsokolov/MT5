"""Train a tabular gradient boosting model for MT5 signals."""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline

from utils import load_config
from data.history import save_history_parquet, load_history_config
from data.features import make_features, train_test_split
from mt5.log_utils import setup_logging, log_exceptions

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for tabular model training."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):  # pragma: no cover - defensive
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.ndarray, pd.Series, pd.Index)):
        return value.tolist()
    return str(value)


def _build_feature_list(df: pd.DataFrame) -> list[str]:
    features = [
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "volatility_30",
        "spread",
        "rsi_14",
        "news_sentiment",
        "market_regime",
    ]
    features += [
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ]
    optional_columns = [
        "volume_ratio",
        "volume_imbalance",
        "SymbolCode",
    ]
    features.extend([col for col in optional_columns if col in df.columns])
    return [col for col in features if col in df.columns]


def _prepare_datasets(
    train_df: pd.DataFrame, test_df: pd.DataFrame, features: list[str]
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    for frame in (train_df, test_df):
        frame.replace([np.inf, -np.inf], np.nan, inplace=True)
        if "Symbol" in frame.columns:
            frame["SymbolCode"] = frame["Symbol"].astype("category").cat.codes
    feature_subset = [col for col in features if col in train_df.columns]
    if not feature_subset:
        raise RuntimeError("No usable features found for training.")
    X_train = train_df[feature_subset]
    y_train = train_df["target"].astype(int)
    X_test = test_df[feature_subset]
    y_test = test_df["target"].astype(int)
    return X_train, y_train, X_test, y_test


def _build_model(cfg: dict[str, Any], seed: int) -> Pipeline:
    model_cfg = cfg.get("tabular_model")
    if not isinstance(model_cfg, dict):
        model_cfg = {}

    max_depth = model_cfg.get("max_depth", cfg.get("tabular_max_depth"))
    learning_rate = model_cfg.get("learning_rate", cfg.get("tabular_learning_rate", 0.1))
    max_iter = model_cfg.get("max_iter", cfg.get("tabular_max_iter", 400))
    l2_reg = model_cfg.get("l2_regularization", cfg.get("tabular_l2_regularization", 0.0))
    class_weight = model_cfg.get("class_weight", cfg.get("tabular_class_weight", "balanced"))

    estimator = HistGradientBoostingClassifier(
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_iter=max_iter,
        l2_regularization=l2_reg,
        class_weight=class_weight,
        early_stopping=True,
        random_state=seed,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", estimator),
    ])
    return pipeline


def _log_feature_importance(model: Pipeline, features: list[str]) -> None:
    estimator = model.named_steps.get("model")
    if estimator is None:
        return
    importances = getattr(estimator, "feature_importances_", None)
    if importances is None:
        return
    if len(importances) != len(features):
        return
    pairs = sorted(zip(features, importances), key=lambda item: item[1], reverse=True)
    top = ", ".join(f"{name}: {score:.4f}" for name, score in pairs[:10])
    logger.info("Top feature importances: %s", top)


@log_exceptions
def main() -> None:
    init_logging()
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)

    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
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

    model = _build_model(cfg, seed)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= cfg.get("tabular_decision_threshold", 0.5)).astype(int)

    accuracy = accuracy_score(y_test, preds)
    try:
        roc_auc = roc_auc_score(y_test, proba)
    except ValueError:
        roc_auc = None
    report = classification_report(y_test, preds, digits=4)

    logger.info("Tabular model accuracy: %.4f", accuracy)
    if roc_auc is not None:
        logger.info("Tabular model ROC-AUC: %.4f", roc_auc)
    logger.info("Classification report:\n%s", report)

    _log_feature_importance(model, X_train.columns.tolist())

    out_path = root / "models" / "tabular"
    out_path.mkdir(parents=True, exist_ok=True)
    model_path = out_path / "model.joblib"
    dump(model, model_path)

    metadata = {
        "features": X_train.columns.tolist(),
        "performance": {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "classification_report": report,
        },
        "training_config": {
            "model_type": "tabular",
            "seed": seed,
            "symbols": symbols,
            "train_rows": cfg.get("train_rows"),
            "tabular_model": cfg.get("tabular_model"),
        },
    }
    metadata_path = out_path / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=_json_default), encoding="utf-8")

    feature_path = out_path / "features.json"
    feature_path.write_text(json.dumps(X_train.columns.tolist(), indent=2), encoding="utf-8")

    logger.info("Model saved to %s", model_path)


if __name__ == "__main__":
    main()
