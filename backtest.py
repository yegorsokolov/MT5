"""Simple backtesting for the Adaptive MT5 bot."""

from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from utils import load_config
from dataset import load_history, make_features


def trailing_stop(entry_price: float, current_price: float, stop: float, distance: float) -> float:
    """Update trailing stop based on price movement."""
    if current_price - distance > stop:
        return current_price - distance
    return stop


def compute_metrics(returns: pd.Series) -> dict:
    cumulative = (1 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
    return {
        "sharpe": sharpe,
        "max_drawdown": drawdown.min() * 100,
        "total_return": cumulative.iloc[-1] - 1,
        "win_rate": (returns > 0).mean() * 100,
    }


def main():
    cfg = load_config()
    data_path = Path(__file__).resolve().parent / "data" / "history.csv"
    if not data_path.exists():
        raise FileNotFoundError("Historical CSV not found under data/history.csv")

    model = joblib.load(Path(__file__).resolve().parent / "model.joblib")

    df = load_history(data_path)
    df = df[df.get("Symbol").isin([cfg.get("symbol")])]
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
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    if "SymbolCode" in df.columns:
        features.append("SymbolCode")
    X = df[features]
    probs = model.predict_proba(X)[:, 1]

    threshold = cfg.get("threshold", 0.55)
    distance = cfg.get("trailing_stop_pips", 20) * 1e-4

    in_position = False
    entry = 0.0
    stop = 0.0
    returns = []

    for price, prob in zip(df["mid"], probs):
        if not in_position and prob > threshold:
            in_position = True
            entry = price
            stop = price - distance
            continue
        if in_position:
            stop = trailing_stop(entry, price, stop, distance)
            if price <= stop:
                returns.append((price - entry) / entry)
                in_position = False
        
    metrics = compute_metrics(pd.Series(returns))
    for k, v in metrics.items():
        if k == "max_drawdown" or k == "win_rate":
            print(f"{k}: {v:.2f}%")
        else:
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    main()
