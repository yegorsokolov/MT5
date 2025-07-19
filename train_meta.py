"""Meta-learning training with neural adapters."""

from pathlib import Path
from typing import List

import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier

from utils import load_config
from dataset import (
    load_history,
    load_history_from_urls,
    make_features,
)


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
    path = root / "data" / f"{sym}_history.csv"
    if path.exists():
        df = load_history(path)
    else:
        urls = cfg.get("data_urls", {}).get(sym)
        if not urls:
            raise FileNotFoundError(f"No history found for {sym} and no URL configured")
        df = load_history_from_urls(urls)
        df.to_csv(path, index=False)
    df["Symbol"] = sym
    return df


def train_base_model(df: pd.DataFrame, features: List[str]):
    """Train a global base model on all symbols."""
    X = df[features]
    y = (df["return"].shift(-1) > 0).astype(int)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LGBMClassifier(n_estimators=200, random_state=42)),
    ])
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


def main():
    cfg = load_config()
    root = Path(__file__).resolve().parent
    symbols = cfg.get("symbols") or [cfg.get("symbol")]

    # load and combine histories
    dfs = [load_symbol_data(s, cfg, root) for s in symbols]
    df = make_features(pd.concat(dfs, ignore_index=True))
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
        "cross_corr",
        "cross_momentum",
        "news_sentiment",
    ]
    if "volume_ratio" in df.columns:
        features.extend(["volume_ratio", "volume_imbalance"])
    features.append("SymbolCode")

    base_model = train_base_model(df, features)
    joblib.dump(base_model, root / "models" / "base_model.joblib")

    train_meta_network(df, features, base_model, root / "models" / "meta_adapter.pt")
    print("Meta-learning model saved to", root / "models" / "meta_adapter.pt")


if __name__ == "__main__":
    main()
