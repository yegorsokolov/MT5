"""Train a diffusion model on historical features and save synthetic sequences."""
from pathlib import Path
import random
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from diffusers import UNet1DModel, DDPMScheduler
from ydata_synthetic.preprocessing.timeseries import TimeSeriesScalerMinMax

from utils import load_config
from data.history import load_history_config
from data.features import (
    make_features,
    make_sequence_arrays,
)
from log_utils import setup_logging, log_exceptions

_LOGGING_INITIALIZED = False


def init_logging() -> logging.Logger:
    """Initialise structured logging for diffusion training."""

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
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_tensor = torch.tensor(X_scaled, dtype=torch.float32).permute(0, 2, 1)
    loader = DataLoader(train_tensor, batch_size=cfg.get("gan_batch_size", 128), shuffle=True)

    model = UNet1DModel(
        sample_size=seq_len,
        in_channels=len(features),
        out_channels=len(features),
        layers_per_block=2,
        block_out_channels=(32, 64),
    ).to(device)
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    for epoch in range(cfg.get("diffusion_epochs", 5)):
        model.train()
        for batch in loader:
            batch = batch.to(device)
            noise = torch.randn_like(batch)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch.size(0),), device=device).long()
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)
            noise_pred = model(noisy, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)
            optim.zero_grad()
            loss.backward()
            optim.step()

    n_samples = cfg.get("gan_num_samples", len(X))
    model.eval()
    noise_scheduler.set_timesteps(50)
    sample = torch.randn(n_samples, len(features), seq_len, device=device)
    for t in noise_scheduler.timesteps:
        noise_pred = model(sample, t).sample
        sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
    synthetic = sample.permute(0, 2, 1).cpu().numpy()
    synthetic = scaler.inverse_transform(synthetic)

    return_idx = features.index("return")
    y_syn = (synthetic[:, -1, return_idx] > 0).astype(int)

    out_path = aug_dir / "synthetic_sequences_diffusion.npz"
    np.savez(out_path, X=synthetic, y=y_syn)
    logger.info("Saved synthetic data to %s", out_path)


if __name__ == "__main__":
    main()
