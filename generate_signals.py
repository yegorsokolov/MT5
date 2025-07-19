"""Generate per-tick probability signals for the EA."""

from pathlib import Path
import joblib
import pandas as pd

import numpy as np

from utils import load_config
from dataset import load_history, make_features


def main():
    cfg = load_config()

    model = joblib.load(Path(__file__).resolve().parent / "model.joblib")
    df = load_history(Path(__file__).resolve().parent / "data" / "history.csv")
    df = df[df.get("Symbol").isin([cfg.get("symbol")])]
    df = make_features(df)

    # optional macro indicators merged on timestamp
    macro_path = Path(__file__).resolve().parent / "data" / "macro.csv"
    if macro_path.exists():
        macro = pd.read_csv(macro_path)
        macro["Timestamp"] = pd.to_datetime(macro["Timestamp"])
        df = df.merge(macro, on="Timestamp", how="left")
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
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    if "SymbolCode" in df.columns:
        features.append("SymbolCode")
    probs = model.predict_proba(df[features])[:, 1]

    ma_ok = df["ma_cross"] == 1
    rsi_ok = df["rsi_14"] > cfg.get("rsi_buy", 55)

    boll_ok = True
    if "boll_break" in df.columns:
        boll_ok = df["boll_break"] == 1

    vol_ok = True
    if "volume_spike" in df.columns:
        vol_ok = df["volume_spike"] == 1

    macro_ok = True
    if "macro_indicator" in df.columns:
        macro_ok = df["macro_indicator"] > cfg.get("macro_threshold", 0.0)

    combined = np.where(ma_ok & rsi_ok & boll_ok & vol_ok & macro_ok, probs, 0.0)

    out = pd.DataFrame({"Timestamp": df["Timestamp"], "prob": combined})
    out.to_csv(Path(__file__).resolve().parent / "signals.csv", index=False)
    print("Signals written to", Path(__file__).resolve().parent / "signals.csv")


if __name__ == "__main__":
    main()
