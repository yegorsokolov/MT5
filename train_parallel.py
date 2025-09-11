"""Parallel training across symbols using Ray."""

from pathlib import Path
from typing import Dict

import logging
import joblib
import pandas as pd
import numpy as np
import random
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier
from ray_utils import ray, init as ray_init, shutdown as ray_shutdown
from data.labels import triple_barrier

from log_utils import setup_logging, log_exceptions

from utils import load_config
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
)
from data.features import (
    make_features,
    train_test_split,
)

setup_logging()
logger = logging.getLogger(__name__)


@ray.remote
@log_exceptions
def train_symbol(sym: str, cfg: Dict, root: Path) -> str:
    """Load data for one symbol, train model and save it."""
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    df = load_history_config(sym, cfg, root, validate=cfg.get("validate", False))
    df["Symbol"] = sym
    df = make_features(df, validate=cfg.get("validate", False))
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes
    df["tb_label"] = triple_barrier(
        df["mid"],
        cfg.get("pt_mult", 0.01),
        cfg.get("sl_mult", 0.01),
        cfg.get("max_horizon", 10),
    )
    train_df, test_df = train_test_split(df, cfg.get("train_rows", len(df) // 2))
    features = [
        c
        for c in [
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
        if c in df.columns
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

    steps = []
    if cfg.get("use_scaler", True):
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", LGBMClassifier(n_estimators=200, random_state=seed)))
    pipe = Pipeline(steps)
    pipe.fit(train_df[features], train_df["tb_label"])
    preds = pipe.predict(test_df[features])
    report = classification_report(test_df["tb_label"], preds)
    out_path = root / "models" / f"{sym}_model.joblib"
    out_path.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, out_path)
    return f"{sym} saved to {out_path}\n{report}"


@log_exceptions
def main() -> None:
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    root = Path(__file__).resolve().parent
    ray_init(num_cpus=cfg.get("ray_num_cpus"))
    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    futures = [train_symbol.remote(sym, cfg, root) for sym in symbols]
    for res in ray.get(futures):
        logger.info(res)
    ray_shutdown()


if __name__ == "__main__":
    main()
