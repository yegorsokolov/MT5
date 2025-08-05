"""Training routine for the Adaptive MT5 bot."""

from pathlib import Path
import random
import json
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

logger = setup_logging()


def log_shap_importance(pipe: Pipeline, X_train: pd.DataFrame, features: list[str]) -> None:
    """Compute SHAP values and save ranked features."""
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
        fi = pd.DataFrame({
            "feature": features,
            "importance": np.abs(shap_values).mean(axis=0),
        })
        out = LOG_DIR / "feature_importance.csv"
        fi.sort_values("importance", ascending=False).to_csv(out, index=False)
        logger.info("Logged feature importance to %s", out)
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to compute SHAP values: %s", e)


@log_exceptions
def main():
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
                    df_sym = load_history_config(sym, cfg, root)
                    df_sym["Symbol"] = sym
                    all_dfs.append(df_sym)
            else:
                df_sym = load_history_config(sym, cfg, root)
                df_sym["Symbol"] = sym
                all_dfs.append(df_sym)

        df = pd.concat(all_dfs, ignore_index=True)
        save_history_parquet(df, root / "data" / "history.parquet")

        df = make_features(df)
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
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        if cfg.get("use_data_augmentation", False) or cfg.get("use_diffusion_aug", False):
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
                    [y_train, pd.Series(y_aug, index=range(len(y_train), len(y_train) + len(y_aug)))],
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
        mlflow.log_metric(f"fold_{fold}_f1_weighted", report["weighted avg"]["f1-score"])

        all_preds.extend(preds)
        all_true.extend(y_val)

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

    out = root / "classification_report.json"
    with out.open("w") as f:
        json.dump(aggregate_report, f, indent=2)
    mlflow.log_artifact(str(out))

    if final_pipe is not None and X_train_final is not None:
        log_shap_importance(final_pipe, X_train_final, features)


if __name__ == "__main__":
    main()
