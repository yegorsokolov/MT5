"""Training routine for the Adaptive MT5 bot."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier

from log_utils import setup_logging, log_exceptions

from utils import load_config
from dataset import load_history, load_history_from_urls, make_features, train_test_split

logger = setup_logging()


@log_exceptions
def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    all_dfs = []
    for sym in symbols:
        sym_path = root / "data" / f"{sym}_history.csv"
        if sym_path.exists():
            df_sym = load_history(sym_path)
        else:
            urls = cfg.get("data_urls", {}).get(sym)
            if not urls:
                raise FileNotFoundError(f"No history found for {sym} and no URL configured")
            df_sym = load_history_from_urls(urls)
            df_sym.to_csv(sym_path, index=False)
        df_sym["Symbol"] = sym
        all_dfs.append(df_sym)

    df = pd.concat(all_dfs, ignore_index=True)
    # also store combined history
    df.to_csv(root / "data" / "history.csv", index=False)

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
    ]
    features.extend([
        c
        for c in df.columns
        if c.startswith("cross_corr_")
        or c.startswith("factor_")
        or c.startswith("cross_mom_")
    ])
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
    print(classification_report(y_test, preds))

    joblib.dump(pipe, Path(__file__).resolve().parent / "model.joblib")
    print("Model saved to", Path(__file__).resolve().parent / "model.joblib")


if __name__ == "__main__":
    main()
