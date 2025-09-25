"""Train TimeGAN on historical features and save synthetic sequences."""
from pathlib import Path
import random
import logging
import numpy as np
import pandas as pd
try:
    import tensorflow as tf
except Exception:  # pragma: no cover - optional dependency
    tf = None

try:
    from ydata_synthetic.synthesizers.timeseries import TimeGAN
    from ydata_synthetic.preprocessing.timeseries import TimeSeriesScalerMinMax
except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency
    raise ModuleNotFoundError(
        "ydata-synthetic is required for TimeGAN training."
        " Install it with `pip install ydata-synthetic` or run the"
        " setup script using Python < 3.13."
    ) from exc

from utils import load_config
from data.history import load_history_config
from data.features import (
    make_features,
    make_sequence_arrays,
)
from mt5.log_utils import setup_logging, log_exceptions

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for TimeGAN training."""

    global _LOGGING_INITIALIZED
    if not _LOGGING_INITIALIZED:
        setup_logging()
        _LOGGING_INITIALIZED = True
    return logging.getLogger(__name__)


logger = logging.getLogger(__name__)


@log_exceptions
def main() -> None:
    init_logging()
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    if tf is not None:
        tf.random.set_seed(seed)
    root = Path(__file__).resolve().parents[1]
    aug_dir = root / "data" / "augmented"
    aug_dir.mkdir(parents=True, exist_ok=True)

    symbols = cfg.get("symbols") or [cfg.get("symbol")]
    dfs = []
    for sym in symbols:
        df_sym = load_history_config(sym, cfg, root)
        df_sym["Symbol"] = sym
        dfs.append(df_sym)

    df = make_features(pd.concat(dfs, ignore_index=True))
    if "Symbol" in df.columns:
        df["SymbolCode"] = df["Symbol"].astype("category").cat.codes

    seq_len = cfg.get("sequence_length", 50)
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

    X, _ = make_sequence_arrays(df, features, seq_len)

    scaler = TimeSeriesScalerMinMax()
    X_scaled = scaler.fit_transform(X)

    model_params = {
        "batch_size": cfg.get("gan_batch_size", 128),
        "rnn_hidden_dim": 24,
        "latent_dim": 8,
        "learning_rate": 5e-4,
    }
    gan = TimeGAN(model_params, seq_len=seq_len, n_seq=len(features))
    gan.train(X_scaled, cfg.get("gan_epochs", 5))

    n_samples = cfg.get("gan_num_samples", len(X))
    synthetic = gan.sample(n_samples)
    synthetic = scaler.inverse_transform(synthetic)

    return_idx = features.index("return")
    y_syn = (synthetic[:, -1, return_idx] > 0).astype(int)

    out_path = aug_dir / "synthetic_sequences.npz"
    np.savez(out_path, X=synthetic, y=y_syn)
    logger.info("Saved synthetic data to %s", out_path)


if __name__ == "__main__":
    main()
