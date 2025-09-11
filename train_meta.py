"""Meta-learning training with neural adapters."""

from pathlib import Path
from typing import List

import logging
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from data.feature_scaler import FeatureScaler
import random

from utils import load_config
from data.history import (
    load_history_parquet,
    save_history_parquet,
    load_history_config,
)
from data.features import make_features
from log_utils import setup_logging, log_exceptions

setup_logging()
logger = logging.getLogger(__name__)


class MetaAdapterNet(nn.Module):
    """Small neural network that adapts base model outputs per symbol."""

    def __init__(self, num_features: int, num_symbols: int, emb_dim: int = 8):
        super().__init__()
        self.embed = nn.Embedding(num_symbols, emb_dim)
        self.fc1 = nn.Linear(num_features + emb_dim, 32)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor, sym_idx: torch.Tensor) -> torch.Tensor:
        emb = self.embed(sym_idx)
        x = torch.cat([x, emb], dim=1)
        x = self.act(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x).squeeze(1)


def load_symbol_data(sym: str, cfg: dict, root: Path) -> pd.DataFrame:
    """Load history for a single symbol, downloading if needed."""
    df = load_history_config(sym, cfg, root, validate=cfg.get("validate", False))
    df["Symbol"] = sym
    return df


def train_base_model(df: pd.DataFrame, features: List[str], cfg: dict):
    """Train a global base model on all symbols."""
    X = df[features]
    y = (df["return"].shift(-1) > 0).astype(int)
    steps = []
    if cfg.get("use_scaler", True):
        steps.append(("scaler", FeatureScaler()))
    seed = cfg.get("seed", 42)
    steps.append(("clf", LGBMClassifier(n_estimators=200, random_state=seed)))
    pipe = Pipeline(steps)
    pipe.fit(X, y)
    return pipe


def train_meta_network(
    df: pd.DataFrame, features: List[str], base_model, out_path: Path
):
    """Train a neural adapter model jointly across symbols."""
    out_path.parent.mkdir(exist_ok=True)

    # target and base probabilities
    y = (df["return"].shift(-1) > 0).astype(int).iloc[:-1].values
    base_probs = base_model.predict_proba(df[features])[:, 1]
    X = df.iloc[:-1][[f for f in features if f != "SymbolCode"]].values
    sym_idx = df.iloc[:-1]["SymbolCode"].values.astype(int)
    base_probs = base_probs[:-1]

    X_tensor = torch.tensor(
        np.hstack([X, base_probs.reshape(-1, 1)]), dtype=torch.float32
    )
    sym_tensor = torch.tensor(sym_idx, dtype=torch.long)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    dataset = TensorDataset(X_tensor, sym_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = MetaAdapterNet(X_tensor.shape[1], df["Symbol"].nunique())
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.BCELoss()

    for _ in range(5):
        for xb, sb, yb in loader:
            optim.zero_grad()
            preds = model(xb, sb)
            loss = loss_fn(preds, yb)
            loss.backward()
            optim.step()

    torch.save(model.state_dict(), out_path)


@log_exceptions
def main():
    cfg = load_config()
    seed = cfg.get("seed", 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    root = Path(__file__).resolve().parent
    symbols = cfg.get("symbols") or [cfg.get("symbol")]

    # load and combine histories
    dfs = [load_symbol_data(s, cfg, root) for s in symbols]
    df = make_features(
        pd.concat(dfs, ignore_index=True), validate=cfg.get("validate", False)
    )
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
    features.append("SymbolCode")

    base_model = train_base_model(df, features, cfg)
    models_dir = root / "models"
    models_dir.mkdir(exist_ok=True)
    joblib.dump(base_model, models_dir / "base_model.joblib")
    if "scaler" in base_model.named_steps:
        base_model.named_steps["scaler"].save(models_dir / "base_scaler.pkl")

    train_meta_network(df, features, base_model, root / "models" / "meta_adapter.pt")
    logger.info("Meta-learning model saved to %s", root / "models" / "meta_adapter.pt")


if __name__ == "__main__":
    main()
