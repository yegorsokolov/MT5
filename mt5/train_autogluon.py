"""Train an AutoGluon tabular ensemble for MT5 signals."""

from __future__ import annotations

import json
import logging
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from autogluon.tabular import TabularPredictor
from joblib import dump

from utils import load_config
from data.history import save_history_parquet, load_history_config
from data.features import make_features, train_test_split
from mt5.log_utils import setup_logging, log_exceptions
from .train_tabular import _build_feature_list, _prepare_datasets, _json_default

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for AutoGluon training."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


def _serialise_leaderboard(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    """Return a JSON serialisable representation of the leaderboard."""

    if df is None or df.empty:
        return []
    cleaned = df.replace({np.nan: None})
    return cleaned.to_dict(orient="records")


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

    train_data = X_train.copy()
    train_data["target"] = y_train
    test_data = X_test.copy()
    test_data["target"] = y_test

    presets = cfg.get("autogluon_presets")
    time_limit = cfg.get("autogluon_time_limit")
    hyperparameters = cfg.get("autogluon_hyperparameters")
    num_stack_levels = cfg.get("autogluon_num_stack_levels")
    eval_metric = cfg.get("autogluon_eval_metric", "accuracy")

    out_path = root / "models" / "autogluon"
    artifacts_path = out_path / "AutogluonModels"
    out_path.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(
        label="target",
        eval_metric=eval_metric,
        problem_type="binary",
        path=str(artifacts_path),
        verbosity=2,
    )

    fit_kwargs: dict[str, Any] = {
        "train_data": train_data,
        "tuning_data": test_data,
        "presets": presets,
        "time_limit": time_limit,
        "hyperparameters": hyperparameters,
        "num_stack_levels": num_stack_levels,
        "ag_args_fit": {"random_seed": seed},
    }
    # Remove ``None`` entries to keep AutoGluon happy
    fit_kwargs = {k: v for k, v in fit_kwargs.items() if v is not None}

    logger.info("Starting AutoGluon training with presets=%s", presets)
    predictor = predictor.fit(**fit_kwargs)

    scores = predictor.evaluate(test_data, auxiliary_metrics=True)
    leaderboard = predictor.leaderboard(test_data, silent=True)

    safe_scores = {
        k: float(v) if isinstance(v, (np.floating, np.integer, float, int)) else v
        for k, v in scores.items()
    }
    logger.info("AutoGluon metrics: %s", safe_scores)
    logger.info("Best AutoGluon model: %s", predictor.get_model_best())

    dump(predictor, out_path / "model.joblib")
    predictor_dir = out_path / "predictor.ag"
    if predictor_dir.exists():
        if predictor_dir.is_dir():
            shutil.rmtree(predictor_dir)
        else:
            predictor_dir.unlink()
    predictor.save(str(predictor_dir))

    metadata: dict[str, Any] = {
        "features": X_train.columns.tolist(),
        "performance": {
            "metrics": scores,
            "leaderboard": _serialise_leaderboard(leaderboard),
            "best_model": predictor.get_model_best(),
        },
        "training_config": {
            "model_type": "autogluon",
            "seed": seed,
            "symbols": symbols,
            "train_rows": cfg.get("train_rows"),
            "autogluon_presets": presets,
            "autogluon_time_limit": time_limit,
            "autogluon_hyperparameters": hyperparameters,
            "autogluon_num_stack_levels": num_stack_levels,
            "autogluon_eval_metric": eval_metric,
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

    logger.info("AutoGluon model saved to %s", out_path)


if __name__ == "__main__":
    main()


__all__ = ["main"]
