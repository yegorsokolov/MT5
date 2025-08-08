from pathlib import Path
import random
import logging
import numpy as np
import pandas as pd

from utils import load_config
from data.history import save_history_parquet, load_history_config
from data.features import make_features, train_test_split
from log_utils import setup_logging, log_exceptions

setup_logging()
logger = logging.getLogger(__name__)


@log_exceptions
def main():
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    root = Path(__file__).resolve().parent
    root.joinpath("data").mkdir(exist_ok=True)

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    for sym in symbols:
        df_sym = load_history_config(sym, cfg, root)
        df_sym["Symbol"] = sym
        dfs.append(df_sym)

    df = pd.concat(dfs, ignore_index=True)
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

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["target"] = (train_df["return"].shift(-1) > 0).astype(int)
    test_df["target"] = (test_df["return"].shift(-1) > 0).astype(int)
    train_df = train_df.dropna(subset=["target"])
    test_df = test_df.dropna(subset=["target"])

    from autogluon.tabular import TabularPredictor

    out_path = root / "models" / "autogluon"
    predictor = TabularPredictor(label="target", path=str(out_path))
    time_limit = cfg.get("ag_time_limit")
    presets = cfg.get("ag_presets", "best_quality")
    fit_args = {}
    if time_limit is not None:
        fit_args["time_limit"] = time_limit
    predictor.fit(train_df[features + ["target"]], presets=presets, **fit_args)

    perf = predictor.evaluate(test_df[features + ["target"]])
    logger.info("%s", perf)
    logger.info("Model saved to %s", out_path)


if __name__ == "__main__":
    main()
