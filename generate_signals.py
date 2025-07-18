"""Generate per-tick probability signals for the EA."""

from pathlib import Path
import joblib
import pandas as pd

from dataset import load_history, make_features


def main():
    model = joblib.load(Path(__file__).resolve().parent / "model.joblib")
    df = make_features(load_history(Path(__file__).resolve().parent / "data" / "history.csv"))
    features = ["return", "ma_10", "ma_30", "rsi_14"]
    probs = model.predict_proba(df[features])[:, 1]

    out = pd.DataFrame({"Timestamp": df["Timestamp"], "prob": probs})
    out.to_csv(Path(__file__).resolve().parent / "signals.csv", index=False)
    print("Signals written to", Path(__file__).resolve().parent / "signals.csv")


if __name__ == "__main__":
    main()
