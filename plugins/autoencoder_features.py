"""Autoencoder embedding feature plugin.

min_cpus: 2
min_mem_gb: 2
requires_gpu: true
"""

from __future__ import annotations

MIN_CPUS = 2
MIN_MEM_GB = 2.0
REQUIRES_GPU = True

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from . import register_feature
from utils import load_config

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "autoencoders"
EPOCHS = 5


class Autoencoder(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 16),
            nn.ReLU(),
            nn.Linear(16, seq_len),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


def _preprocess(windows: np.ndarray) -> np.ndarray:
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-6
    return (windows - mean) / std


def train_autoencoder(data: np.ndarray, seq_len: int, embed_dim: int, path: Path) -> Autoencoder:
    model = Autoencoder(seq_len, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    tensor = torch.tensor(data, dtype=torch.float32)
    ds = torch.utils.data.TensorDataset(tensor, tensor)
    loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for batch, target in loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = loss_fn(out, target)
            loss.backward()
            optimizer.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    model.eval()
    return model


@register_feature
def add_autoencoder_embeddings(df: pd.DataFrame, **_) -> pd.DataFrame:
    """Add autoencoder embeddings computed from rolling mid price windows."""
    cfg = load_config()
    if not cfg.get("use_autoencoder_features", False):
        return df

    if "mid" not in df.columns and {"Bid", "Ask"}.issubset(df.columns):
        df = df.assign(mid=(df["Bid"] + df["Ask"]) / 2)
    if "mid" not in df.columns:
        return df

    seq_len = cfg.get("autoencoder_window", 30)
    embed_dim = cfg.get("autoencoder_dim", 8)
    model_path = MODELS_DIR / f"ae_{seq_len}_{embed_dim}.pt"

    if model_path.exists():
        model = Autoencoder(seq_len, embed_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
    else:
        sequences = []
        if "Symbol" in df.columns:
            for _, g in df.groupby("Symbol"):
                arr = g["mid"].values
                if len(arr) >= seq_len:
                    win = np.lib.stride_tricks.sliding_window_view(arr, seq_len)
                    sequences.append(_preprocess(win))
        else:
            arr = df["mid"].values
            if len(arr) >= seq_len:
                win = np.lib.stride_tricks.sliding_window_view(arr, seq_len)
                sequences.append(_preprocess(win))
        if not sequences:
            for i in range(embed_dim):
                df[f"ae_{i}"] = np.nan
            return df
        data = np.concatenate(sequences, axis=0)
        model = train_autoencoder(data, seq_len, embed_dim, model_path)

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        arr = group["mid"].values
        feats = np.full((len(group), embed_dim), np.nan)
        if len(arr) >= seq_len:
            windows = np.lib.stride_tricks.sliding_window_view(arr, seq_len)
            windows = _preprocess(windows)
            tensor = torch.tensor(windows, dtype=torch.float32)
            with torch.no_grad():
                emb = model.encode(tensor).numpy()
            feats[seq_len - 1 :] = emb
        out = group.copy()
        for i in range(embed_dim):
            out[f"ae_{i}"] = feats[:, i]
        return out

    if "Symbol" in df.columns:
        df = df.groupby("Symbol", group_keys=False).apply(_apply)
    else:
        df = _apply(df)

    return df
