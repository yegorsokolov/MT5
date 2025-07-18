"""Training routine for the Adaptive MT5 bot."""

from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils import load_config
from dataset import load_history, make_features, train_test_split


def main():
    cfg = load_config()
    data_path = Path(__file__).resolve().parent / "data" / "history.csv"
    if not data_path.exists():
        raise FileNotFoundError("Historical CSV not found under data/history.csv")

    df = load_history(data_path)
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
