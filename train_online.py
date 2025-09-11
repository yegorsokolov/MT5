import time
from pathlib import Path
import random
import logging
import numpy as np
import pandas as pd
import joblib
from river import compose, preprocessing, linear_model

from utils import load_config
from log_utils import setup_logging, log_exceptions
from feature_store import load_feature, latest_version

setup_logging()
logger = logging.getLogger(__name__)


def fetch_features(version: str | None = None) -> pd.DataFrame:
    """Retrieve features for ``version`` from the feature store."""
    ver = version or latest_version()
    if ver is None:
        return pd.DataFrame()
    return load_feature(ver)


@log_exceptions
def train_online() -> None:
    """Incrementally update a river model with the latest realtime features."""
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    root = Path(__file__).resolve().parent
    model_dir = root / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "online.joblib"

    version = latest_version()
    if version is None:
        logger.warning("No features available in store")
        return

    last_ts = None
    if model_path.exists():
        try:
            model, last_ts = joblib.load(model_path)
            logger.info("Loaded existing online model from %s", model_path)
        except Exception as exc:  # pragma: no cover - just warn
            logger.warning("Failed to load online model: %s", exc)
            model = compose.Pipeline(
                preprocessing.StandardScaler(), linear_model.LogisticRegression()
            )
            last_ts = None
    else:
        model = compose.Pipeline(
            preprocessing.StandardScaler(), linear_model.LogisticRegression()
        )

    while True:
        df = fetch_features(version)
        if last_ts is not None:
            df = df[df["Timestamp"] > last_ts]

        if df.empty:
            time.sleep(60)
            continue

        if "Symbol" in df.columns:
            df = df.sort_values(["Symbol", "Timestamp"])
            df["next_ret"] = df.groupby("Symbol")["return"].shift(-1)
        else:
            df = df.sort_values("Timestamp")
            df["next_ret"] = df["return"].shift(-1)

        df.dropna(subset=["next_ret"], inplace=True)

        if "Symbol" in df.columns and "SymbolCode" not in df.columns:
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
            "market_regime",
        ]
        if "news_sentiment" in df.columns:
            features.append("news_sentiment")
        features.extend(
            [
                c
                for c in df.columns
                if c.startswith("cross_corr_")
                or c.startswith("factor_")
                or c.startswith("cross_mom_")
            ]
        )
        if "volume_ratio" in df.columns:
            features.extend(["volume_ratio", "volume_imbalance"])
        if "SymbolCode" in df.columns:
            features.append("SymbolCode")

        for row in df.itertuples(index=False):
            x = {f: getattr(row, f) for f in features}
            y = int(getattr(row, "next_ret") > 0)
            model.learn_one(x, y)
            last_ts = getattr(row, "Timestamp")

        joblib.dump((model, last_ts), model_path)
        logger.info("Updated online model with %d rows", len(df))
        time.sleep(300)


if __name__ == "__main__":
    train_online()
