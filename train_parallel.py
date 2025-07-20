"""Parallel training across symbols using Ray."""

from pathlib import Path
from typing import Dict

import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
import ray

from utils import load_config
from dataset import load_history, load_history_from_urls, make_features, train_test_split


@ray.remote
def train_symbol(sym: str, cfg: Dict, root: Path) -> str:
    """Load data for one symbol, train model and save it."""
    path = root / "data" / f"{sym}_history.csv"
    if path.exists():
        df = load_history(path)
    else:
        urls = cfg.get("data_urls", {}).get(sym)
        if not urls:
            raise FileNotFoundError(f"No history for {sym} and no URL configured")
        df = load_history_from_urls(urls)
        path.parent.mkdir(exist_ok=True)
        df.to_csv(path, index=False)
    df["Symbol"] = sym
    df = make_features(df)
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes
    train_df, test_df = train_test_split(df, cfg.get("train_rows", len(df)//2))
    features = [c for c in [
        "return","ma_5","ma_10","ma_30","ma_60","volatility_30",
        "spread","rsi_14","cross_corr","cross_momentum","news_sentiment",
    ] if c in df.columns]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio","volume_imbalance"])
    if "SymbolCode" in df.columns:
        features.append("SymbolCode")

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(n_estimators=200, random_state=42)),
    ])
    pipe.fit(train_df[features], (train_df["return"].shift(-1) > 0).astype(int))
    preds = pipe.predict(test_df[features])
    report = classification_report(
        (test_df["return"].shift(-1) > 0).astype(int), preds
    )
    out_path = root / "models" / f"{sym}_model.joblib"
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, out_path)
    return f"{sym} saved to {out_path}\n{report}"


def main() -> None:
    cfg = load_config()
    root = Path(__file__).resolve().parent
    ray.init(num_cpus=cfg.get("ray_num_cpus"))
    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    futures = [train_symbol.remote(sym, cfg, root) for sym in symbols]
    for res in ray.get(futures):
        print(res)
    ray.shutdown()


if __name__ == "__main__":
    main()

