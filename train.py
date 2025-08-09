"""Training routine for the Adaptive MT5 bot."""

from pathlib import Path
import random
import json
import logging
import argparse
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMClassifier
import mlflow

from log_utils import setup_logging, log_exceptions, LOG_DIR
import numpy as np

try:
    import shap
except Exception:  # noqa: E722
    shap = None

from utils import load_config, mlflow_run
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
    load_history_iter,
)
from data.features import make_features
from state_manager import save_checkpoint, load_latest_checkpoint
from analysis.regime_detection import periodic_reclassification

setup_logging()
logger = logging.getLogger(__name__)


def log_shap_importance(
    pipe: Pipeline,
    X_train: pd.DataFrame,
    features: list[str],
    report_dir: Path | None = None,
) -> None:
    """Compute SHAP values, saving ranked features and optional plot."""
    if shap is None:
        logger.info("shap not installed, skipping feature importance")
        return
    try:
        X_used = X_train
        if "scaler" in pipe.named_steps:
            X_used = pipe.named_steps["scaler"].transform(X_used)
        explainer = shap.TreeExplainer(pipe.named_steps["clf"])
        shap_values = explainer.shap_values(X_used)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        fi = pd.DataFrame(
            {
                "feature": features,
                "importance": np.abs(shap_values).mean(axis=0),
            }
        )
        out = LOG_DIR / "feature_importance.csv"
        fi.sort_values("importance", ascending=False).to_csv(out, index=False)
        logger.info("Logged feature importance to %s", out)
        if report_dir is not None:
            import matplotlib.pyplot as plt

            report_dir.mkdir(exist_ok=True)
            plt.figure()
            shap.summary_plot(shap_values, X_used, show=False, plot_type="bar")
            plt.tight_layout()
            plt.savefig(report_dir / "feature_importance.png")
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to compute SHAP values: %s", e)


@log_exceptions
def main(cfg: dict | None = None, export: bool = False) -> float:
    """Train LightGBM model and return weighted F1 score."""
    if cfg is None:
        cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    with mlflow_run("training", cfg):
        symbols = cfg.get("symbols") or [cfg.get("symbol")]
        all_dfs = []
        chunk_size = cfg.get("stream_chunk_size", 100_000)
        stream = cfg.get("stream_history", False)
        for sym in symbols:
            if stream:
                pq_path = root / "data" / f"{sym}_history.parquet"
                if pq_path.exists():
                    for chunk in load_history_iter(pq_path, chunk_size):
                        chunk["Symbol"] = sym
                        all_dfs.append(chunk)
                else:
                    df_sym = load_history_config(
                        sym, cfg, root, validate=cfg.get("validate", False)
                    )
                    df_sym["Symbol"] = sym
                    all_dfs.append(df_sym)
            else:
                df_sym = load_history_config(
                    sym, cfg, root, validate=cfg.get("validate", False)
                )
                df_sym["Symbol"] = sym
                all_dfs.append(df_sym)

        df = pd.concat(all_dfs, ignore_index=True)
        save_history_parquet(df, root / "data" / "history.parquet")

        df = make_features(df, validate=cfg.get("validate", False))
        df = periodic_reclassification(df, step=cfg.get("regime_reclass_period", 500))
        if "Symbol" in df.columns:
            df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

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
    features.extend(
        [
            c
            for c in df.columns
            if c.startswith("cross_corr_")
            or c.startswith("factor_")
            or c.startswith("cross_mom_")
        ]
    )
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    if "SymbolCode" in df.columns:
        features.append("SymbolCode")

    X = df[features]
    y = (df["return"].shift(-1) > 0).astype(int)

    tscv = TimeSeriesSplit(n_splits=cfg.get("n_splits", 5))
    all_preds: list[int] = []
    all_true: list[int] = []
    final_pipe: Pipeline | None = None
    X_train_final: pd.DataFrame | None = None
    start_fold = 0
    ckpt = load_latest_checkpoint(cfg.get("checkpoint_dir"))
    if ckpt:
        last_fold, state = ckpt
        start_fold = last_fold + 1
        all_preds = state.get("all_preds", [])
        all_true = state.get("all_true", [])
        final_pipe = state.get("model")
        logger.info("Resuming from checkpoint at fold %s", last_fold)

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        if fold < start_fold:
            continue
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if cfg.get("use_data_augmentation", False) or cfg.get(
            "use_diffusion_aug", False
        ):
            fname = (
                "synthetic_sequences_diffusion.npz"
                if cfg.get("use_diffusion_aug", False)
                else "synthetic_sequences.npz"
            )
            aug_path = root / "data" / "augmented" / fname
            if aug_path.exists():
                data = np.load(aug_path)
                X_aug = data["X"][:, -1, :]
                y_aug = data["y"]
                df_aug = pd.DataFrame(X_aug, columns=features)
                X_train = pd.concat([X_train, df_aug], ignore_index=True)
                y_train = pd.concat(
                    [
                        y_train,
                        pd.Series(
                            y_aug, index=range(len(y_train), len(y_train) + len(y_aug))
                        ),
                    ],
                    ignore_index=True,
                )

        steps = []
        if cfg.get("use_scaler", True):
            steps.append(("scaler", StandardScaler()))
        steps.append(
            (
                "clf",
                LGBMClassifier(
                    n_estimators=200,
                    n_jobs=cfg.get("n_jobs", 1),
                    random_state=seed,
                ),
            )
        )
        pipe = Pipeline(steps)

        pipe.fit(
            X_train,
            y_train,
            clf__eval_set=[(X_val, y_val)],
            clf__early_stopping_rounds=cfg.get("early_stopping_rounds", 50),
            clf__verbose=False,
        )

        preds = pipe.predict(X_val)
        report = classification_report(y_val, preds, output_dict=True)
        logger.info("Fold %d\n%s", fold, classification_report(y_val, preds))
        mlflow.log_metric(
            f"fold_{fold}_f1_weighted", report["weighted avg"]["f1-score"]
        )

        all_preds.extend(preds)
        all_true.extend(y_val)
        save_checkpoint(
            {
                "model": pipe,
                "all_preds": all_preds,
                "all_true": all_true,
                "metrics": report,
            },
            fold,
            cfg.get("checkpoint_dir"),
        )

        if fold == tscv.n_splits - 1:
            final_pipe = pipe
            X_train_final = X_train

    aggregate_report = classification_report(all_true, all_preds, output_dict=True)
    logger.info("\n%s", classification_report(all_true, all_preds))

    joblib.dump(final_pipe, root / "model.joblib")
    logger.info("Model saved to %s", root / "model.joblib")
    mlflow.log_param("use_scaler", cfg.get("use_scaler", True))
    mlflow.log_metric("f1_weighted", aggregate_report["weighted avg"]["f1-score"])
    mlflow.log_artifact(str(root / "model.joblib"))

    # Train dedicated models for each market regime
    base_features = [f for f in features if f != "market_regime"]
    regime_models: dict[int, Pipeline] = {}
    for regime in sorted(df["market_regime"].unique()):
        mask = df["market_regime"] == regime
        X_reg = df.loc[mask, base_features]
        y_reg = (df.loc[mask, "return"].shift(-1) > 0).astype(int)
        steps_reg: list[tuple[str, object]] = []
        if cfg.get("use_scaler", True):
            steps_reg.append(("scaler", StandardScaler()))
        steps_reg.append(
            (
                "clf",
                LGBMClassifier(
                    n_estimators=200,
                    n_jobs=cfg.get("n_jobs", 1),
                    random_state=seed,
                ),
            )
        )
        pipe_reg = Pipeline(steps_reg)
        pipe_reg.fit(X_reg, y_reg)
        regime_models[int(regime)] = pipe_reg
        logger.info("Trained regime-specific model for regime %s", regime)

    regimes_path = root / "regime_models.joblib"
    joblib.dump(regime_models, regimes_path)
    mlflow.log_artifact(str(regimes_path))
    logger.info("Regime-specific models saved to %s", regimes_path)

    out = root / "classification_report.json"
    with out.open("w") as f:
        json.dump(aggregate_report, f, indent=2)
    mlflow.log_artifact(str(out))

    if (
        final_pipe is not None
        and X_train_final is not None
        and cfg.get("feature_importance", False)
    ):
        report_dir = Path(__file__).resolve().parent / "reports"
        log_shap_importance(final_pipe, X_train_final, features, report_dir)

    if export and final_pipe is not None:
        from models.export import export_lightgbm

        sample = X.iloc[: min(len(X), 10)]
        if "scaler" in final_pipe.named_steps:
            sample = final_pipe.named_steps["scaler"].transform(sample)
        clf = final_pipe.named_steps.get("clf", final_pipe)
        export_lightgbm(clf, sample)

    return float(aggregate_report.get("weighted avg", {}).get("f1-score", 0.0))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter search")
    parser.add_argument("--export", action="store_true", help="Export model to ONNX")
    args = parser.parse_args()
    if args.tune:
        from tuning.hyperopt import tune_lgbm

        cfg = load_config()
        tune_lgbm(cfg)
    else:
        main(export=args.export)
