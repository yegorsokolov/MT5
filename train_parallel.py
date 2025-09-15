"""Parallel training across symbols using Ray."""

from pathlib import Path
from typing import Any, Callable, Dict, Iterable

import logging
import joblib
import pandas as pd
import numpy as np
import optuna
import random
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split as sklearn_train_test_split
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from ray_utils import ray, init as ray_init, shutdown as ray_shutdown

from log_utils import setup_logging, log_exceptions


setup_logging()
logger = logging.getLogger(__name__)


DEFAULT_LGBM_PARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "objective": "binary",
    "n_jobs": -1,
    "verbose": -1,
}

_FEATURE_SCALER_CLS: type | None = None


def _get_feature_scaler_cls() -> type:
    """Return the feature scaler class, falling back to ``StandardScaler``."""

    global _FEATURE_SCALER_CLS
    if _FEATURE_SCALER_CLS is not None:
        return _FEATURE_SCALER_CLS

    try:
        from data.feature_scaler import FeatureScaler as scaler_cls  # type: ignore
    except Exception:  # pragma: no cover - optional dependency fallback
        from sklearn.preprocessing import StandardScaler

        class _FallbackScaler(StandardScaler):
            def save(self, path: Path) -> None:
                joblib.dump(self, path)

        scaler_cls = _FallbackScaler

    _FEATURE_SCALER_CLS = scaler_cls
    return scaler_cls


def build_lightgbm_pipeline(
    params: Dict[str, Any],
    use_scaler: bool = True,
) -> Pipeline:
    """Create the LightGBM pipeline with optional feature scaling."""

    steps: list[tuple[str, Any]] = []
    if use_scaler:
        scaler_cls = _get_feature_scaler_cls()
        steps.append(("scaler", scaler_cls()))
    clf_params = dict(params)
    steps.append(("clf", LGBMClassifier(**clf_params)))
    return Pipeline(steps)


def best_f1_threshold(
    y_true: Iterable[int],
    probabilities: Iterable[float],
) -> tuple[float, float]:
    """Return the probability threshold that maximises the F1 score."""

    probs = np.asarray(list(probabilities), dtype=float)
    if probs.size == 0:
        return 0.5, 0.0

    y = np.asarray(list(y_true), dtype=int)
    candidate_thresholds = np.unique(
        np.concatenate(
            (
                np.linspace(0.01, 0.99, 99),
                np.clip(probs, 0.0, 1.0),
                np.array([0.5]),
            )
        )
    )
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in candidate_thresholds:
        preds = (probs >= threshold).astype(int)
        score = f1_score(y, preds, zero_division=0)
        if score > best_f1 or (np.isclose(score, best_f1) and threshold < best_threshold):
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold, best_f1


def _compose_params(seed: int, overrides: Dict[str, Any] | None = None) -> Dict[str, Any]:
    params = dict(DEFAULT_LGBM_PARAMS)
    if overrides:
        params.update(overrides)
    params["random_state"] = seed
    return params


def tune_lightgbm_hyperparameters(
    train_df: pd.DataFrame,
    features: list[str],
    seed: int,
    use_scaler: bool,
    n_trials: int = 25,
    base_params: Dict[str, Any] | None = None,
    study_factory: Callable[..., optuna.Study] | None = None,
) -> tuple[Dict[str, Any], float]:
    """Search LightGBM parameters that maximise validation F1."""

    base_overrides = dict(base_params or {})
    params_base = _compose_params(seed, base_overrides)
    if n_trials <= 0 or train_df["tb_label"].nunique() < 2:
        return params_base, 0.0

    val_fraction = min(0.25, max(0.2, 1.0 / max(len(train_df), 1)))
    try:
        X_train, X_val, y_train, y_val = sklearn_train_test_split(
            train_df[features],
            train_df["tb_label"],
            test_size=val_fraction,
            random_state=seed,
            stratify=train_df["tb_label"],
        )
    except ValueError:
        X_train, X_val, y_train, y_val = sklearn_train_test_split(
            train_df[features],
            train_df["tb_label"],
            test_size=val_fraction,
            random_state=seed,
            stratify=None,
        )

    def objective(trial: optuna.Trial) -> float:
        candidate = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        }
        combined = {**base_overrides, **candidate}
        pipeline = build_lightgbm_pipeline(
            _compose_params(seed, combined),
            use_scaler=use_scaler,
        )
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_val)[:, 1]
        _, score = best_f1_threshold(y_val, probs)
        return score

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = (
        study_factory(direction="maximize", sampler=sampler)
        if study_factory
        else optuna.create_study(direction="maximize", sampler=sampler)
    )
    study.optimize(objective, n_trials=n_trials)
    best_params = _compose_params(seed, {**base_overrides, **study.best_params})
    return best_params, float(study.best_value)


@ray.remote
@log_exceptions
def train_symbol(sym: str, cfg: Dict, root: Path) -> str:
    """Load data for one symbol, train model and save it."""
    from data.features import make_features, train_test_split
    from data.history import load_history_config
    from data.labels import triple_barrier

    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    df = load_history_config(sym, cfg, root, validate=cfg.get("validate", False))
    df["Symbol"] = sym
    df = make_features(df, validate=cfg.get("validate", False))
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes
    df["tb_label"] = triple_barrier(
        df["mid"],
        cfg.get("pt_mult", 0.01),
        cfg.get("sl_mult", 0.01),
        cfg.get("max_horizon", 10),
    )
    train_df, test_df = train_test_split(df, cfg.get("train_rows", len(df) // 2))
    features = [
        c
        for c in [
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
        if c in df.columns
    ]
    features += [
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    if "SymbolCode" in df.columns:
        features.append("SymbolCode")

    use_scaler = cfg.get("use_scaler", True)
    n_trials = cfg.get("lgbm_optuna_trials", 25)
    base_params = dict(cfg.get("lightgbm_params") or {})

    try:
        tuned_params, tuned_score = tune_lightgbm_hyperparameters(
            train_df,
            features,
            seed,
            use_scaler,
            n_trials=n_trials,
            base_params=base_params,
        )
        logger.info(
            "Optuna tuning for %s achieved inner F1=%.4f", sym, tuned_score
        )
    except Exception as exc:  # pragma: no cover - optuna failures
        logger.exception("Parameter search failed for %s: %s", sym, exc)
        tuned_params = _compose_params(seed, base_params)

    pipe = build_lightgbm_pipeline(tuned_params, use_scaler=use_scaler)
    pipe.tuned_params_ = tuned_params
    pipe.fit(train_df[features], train_df["tb_label"])
    val_probs = pipe.predict_proba(test_df[features])[:, 1]
    best_threshold, val_f1 = best_f1_threshold(test_df["tb_label"], val_probs)
    preds = (val_probs >= best_threshold).astype(int)
    pipe.best_threshold_ = best_threshold
    pipe.validation_f1_ = val_f1
    pipe.training_features_ = features
    report = classification_report(
        test_df["tb_label"], preds, zero_division=0
    )
    logger.info(
        "Best threshold for %s is %.4f yielding F1=%.4f",
        sym,
        best_threshold,
        val_f1,
    )
    out_path = root / "models" / f"{sym}_model.joblib"
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, out_path)
    if "scaler" in pipe.named_steps:
        scaler = pipe.named_steps["scaler"]
        scaler_path = root / "models" / f"{sym}_scaler.pkl"
        save_method = getattr(scaler, "save", None)
        if callable(save_method):
            save_method(scaler_path)
        else:  # pragma: no cover - fallback for sklearn scalers
            joblib.dump(scaler, scaler_path)
    return (
        f"{sym} saved to {out_path} (F1={val_f1:.4f}, threshold={best_threshold:.4f})\n"
        f"{report}"
    )


@log_exceptions
def main() -> None:
    from utils import load_config

    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    root = Path(__file__).resolve().parent
    ray_init(num_cpus=cfg.get("ray_num_cpus"))
    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    futures = [train_symbol.remote(sym, cfg, root) for sym in symbols]
    for res in ray.get(futures):
        logger.info(res)
    ray_shutdown()


if __name__ == "__main__":
    main()
