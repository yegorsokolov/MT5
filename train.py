"""Training routine for the Adaptive MT5 bot."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils import load_config
from dataset import load_history, load_history_from_urls, make_features, train_test_split


def main():
    cfg = load_config()
    data_path = Path(__file__).resolve().parent / "data" / "history.csv"
    data_path.parent.mkdir(exist_ok=True)
    if data_path.exists():
        df = load_history(data_path)
    elif cfg.get("data_urls"):
        urls_cfg = cfg["data_urls"]
        if isinstance(urls_cfg, dict):
            symbol = cfg.get("symbol")
            urls = urls_cfg.get(symbol)
            if not urls:
                raise ValueError(f"No data URLs configured for symbol {symbol}")
        else:
            urls = urls_cfg
        df = load_history_from_urls(urls)
        df.to_csv(data_path, index=False)
    else:
        raise FileNotFoundError(
            "Historical CSV not found and no data_urls provided in config"
        )

    df = make_features(df)

    train_df, test_df = train_test_split(df, cfg.get("train_rows", len(df) // 2))

    features = ["return", "ma_10", "ma_30", "rsi_14"]
    X_train = train_df[features]
    y_train = (train_df["return"].shift(-1) > 0).astype(int)

    X_test = test_df[features]
    y_test = (test_df["return"].shift(-1) > 0).astype(int)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(n_estimators=100, random_state=42)),
    ])

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    print(classification_report(y_test, preds))

    joblib.dump(pipe, Path(__file__).resolve().parent / "model.joblib")
    print("Model saved to", Path(__file__).resolve().parent / "model.joblib")


if __name__ == "__main__":
    main()
