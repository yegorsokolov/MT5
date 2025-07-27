"""Training routine for the Adaptive MT5 bot."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
import mlflow

from log_utils import setup_logging, log_exceptions, LOG_DIR
import numpy as np
try:
    import shap
except Exception:  # noqa: E722
    shap = None

from utils import load_config
from dataset import (
    load_history_parquet,
    save_history_parquet,
    make_features,
    train_test_split,
    load_history_config,
)

logger = setup_logging()


@log_exceptions
def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    mlflow.set_experiment("training")
    with mlflow.start_run():
        symbols = cfg.get("symbols") or [cfg.get("symbol")]
        all_dfs = []
        for sym in symbols:
            df_sym = load_history_config(sym, cfg, root)
            df_sym["Symbol"] = sym
            all_dfs.append(df_sym)

        df = pd.concat(all_dfs, ignore_index=True)
        # also store combined history
        save_history_parquet(df, root / "data" / "history.parquet")

        df = make_features(df)
        if "Symbol" in df.columns:
            df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

        train_df, test_df = train_test_split(df, cfg.get("train_rows", len(df) // 2))

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
        X_train = train_df[features]
        y_train = (train_df["return"].shift(-1) > 0).astype(int)

        X_test = test_df[features]
        y_test = (test_df["return"].shift(-1) > 0).astype(int)

        steps = []
        if cfg.get("use_scaler", True):
            steps.append(("scaler", StandardScaler()))
        steps.append(("clf", LGBMClassifier(n_estimators=200, random_state=42)))
        pipe = Pipeline(steps)

        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)
        report = classification_report(y_test, preds, output_dict=True)
        print(classification_report(y_test, preds))

        joblib.dump(pipe, root / "model.joblib")
        print("Model saved to", root / "model.joblib")
        mlflow.log_param("use_scaler", cfg.get("use_scaler", True))
        mlflow.log_metric("f1_weighted", report["weighted avg"]["f1-score"])
        mlflow.log_artifact(str(root / "model.joblib"))

        if shap is not None:
            try:
                X_used = X_train
                if "scaler" in pipe.named_steps:
                    X_used = pipe.named_steps["scaler"].transform(X_used)
                explainer = shap.TreeExplainer(pipe.named_steps["clf"])
                shap_values = explainer.shap_values(X_used)
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]
                imp = np.abs(shap_values).mean(axis=0)
                fi = pd.DataFrame({"feature": features, "importance": imp})
                fi.sort_values("importance", ascending=False).to_csv(
                    LOG_DIR / "feature_importance.csv", index=False
                )
                logger.info(
                    "Logged feature importance to %s",
                    LOG_DIR / "feature_importance.csv",
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to compute SHAP values: %s", e)
        else:
            logger.info("shap not installed, skipping feature importance")


if __name__ == "__main__":
    main()
