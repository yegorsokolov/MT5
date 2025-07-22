"""Generate per-tick probability signals for the EA."""

from pathlib import Path
import joblib
import pandas as pd

import numpy as np

from utils import load_config
from dataset import load_history, load_history_parquet, make_features

logger = setup_logging()
from log_utils import setup_logging, log_exceptions


def load_models(paths):
    """Load multiple joblib models from relative paths."""
    models = []
    for p in paths:
        mp = Path(__file__).resolve().parent / p
        if mp.exists():
            models.append(joblib.load(mp))
    return models


def bayesian_average(prob_arrays):
    """Combine probabilities using a simple Bayesian model averaging."""
    logits = [np.log(p / (1 - p + 1e-12)) for p in prob_arrays]
    avg_logit = np.mean(logits, axis=0)
    return 1 / (1 + np.exp(-avg_logit))


@log_exceptions
def main():
    cfg = load_config()

    model_paths = cfg.get("ensemble_models", ["model.joblib"])
    models = load_models(model_paths)
    if not models:
        models = [joblib.load(Path(__file__).resolve().parent / "model.joblib")]
    hist_path_csv = Path(__file__).resolve().parent / "data" / "history.csv"
    hist_path_pq = Path(__file__).resolve().parent / "data" / "history.parquet"
    if hist_path_pq.exists():
        df = load_history_parquet(hist_path_pq)
    else:
        df = load_history(hist_path_csv)
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
        "ma_h4",
        "volatility_30",
        "spread",
        "rsi_14",
        "hour_sin",
        "hour_cos",
        "news_sentiment",
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

    prob_list = [m.predict_proba(df[features])[:, 1] for m in models]
    if len(prob_list) == 1:
        probs = prob_list[0]
    else:
        method = cfg.get("ensemble_method", "average")
        if method == "bayesian":
            probs = bayesian_average(prob_list)
        else:
            probs = np.mean(prob_list, axis=0)

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

    news_ok = True
    if not cfg.get("enable_news_trading", True):
        window = cfg.get("avoid_news_minutes", 5)
        if "nearest_news_minutes" in df.columns:
            news_ok = df["nearest_news_minutes"] > window

    sent_ok = True
    if "news_sentiment" in df.columns:
        sent_ok = df["news_sentiment"] > 0

    mom_ok = True
    factor_cols = [c for c in df.columns if c.startswith("factor_")]
    if factor_cols:
        mom_ok = df[factor_cols[0]] > 0

    combined = np.where(
        ma_ok & rsi_ok & boll_ok & vol_ok & macro_ok & news_ok & sent_ok & mom_ok,
        probs,
        0.0,
    )

    out = pd.DataFrame({"Timestamp": df["Timestamp"], "prob": combined})
    out.to_csv(Path(__file__).resolve().parent / "signals.csv", index=False)
    print("Signals written to", Path(__file__).resolve().parent / "signals.csv")


if __name__ == "__main__":
    main()
