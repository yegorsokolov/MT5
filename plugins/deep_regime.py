"""Deep learning regime feature plugin.

min_cpus: 2
min_mem_gb: 4
requires_gpu: true
"""

from __future__ import annotations

MIN_CPUS = 2
MIN_MEM_GB = 4.0
REQUIRES_GPU = True

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
import joblib
from pathlib import Path

from . import register_feature
from utils import load_config

MODELS_DIR = Path(__file__).resolve().parents[1] / "models" / "deep_regime"
EPOCHS = 5


class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len: int, embed_dim: int):
        super().__init__()
        self.seq_len = seq_len
        self.encoder = nn.LSTM(input_size=1, hidden_size=embed_dim, batch_first=True)
        self.decoder = nn.LSTM(input_size=embed_dim, hidden_size=1, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        repeated = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        decoded, _ = self.decoder(repeated)
        return decoded

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.encoder(x)
        return h[-1]


def _preprocess(windows: np.ndarray) -> np.ndarray:
    mean = windows.mean(axis=1, keepdims=True)
    std = windows.std(axis=1, keepdims=True) + 1e-6
    return (windows - mean) / std


def train_model(
    data: np.ndarray,
    seq_len: int,
    embed_dim: int,
    n_states: int,
    model_path: Path,
    cluster_path: Path,
) -> tuple[LSTMAutoencoder, KMeans]:
    model = LSTMAutoencoder(seq_len, embed_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.MSELoss()
    tensor = torch.tensor(data[:, :, None], dtype=torch.float32)
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
    with torch.no_grad():
        embeddings = model.encode(tensor).numpy()
    kmeans = KMeans(n_clusters=n_states, n_init=10)
    kmeans.fit(embeddings)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    joblib.dump(kmeans, cluster_path)
    model.eval()
    return model, kmeans


@register_feature
def add_deep_regime(df: pd.DataFrame) -> pd.DataFrame:
    """Add deep learning based regime labels when enabled."""
    cfg = load_config()
    if not cfg.get("use_deep_regime", False):
        return df

    if "mid" not in df.columns and {"Bid", "Ask"}.issubset(df.columns):
        df = df.assign(mid=(df["Bid"] + df["Ask"]) / 2)
    if "mid" not in df.columns:
        return df

    seq_len = cfg.get("deep_regime_window", 30)
    embed_dim = cfg.get("deep_regime_dim", 8)
    n_states = cfg.get("deep_regime_states", 3)

    model_path = MODELS_DIR / f"lstm_{seq_len}_{embed_dim}.pt"
    cluster_path = MODELS_DIR / f"kmeans_{seq_len}_{embed_dim}_{n_states}.joblib"

    if model_path.exists() and cluster_path.exists():
        model = LSTMAutoencoder(seq_len, embed_dim)
        model.load_state_dict(torch.load(model_path))
        model.eval()
        kmeans: KMeans = joblib.load(cluster_path)
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
            df["regime_dl"] = np.nan
            return df
        data = np.concatenate(sequences, axis=0)
        model, kmeans = train_model(data, seq_len, embed_dim, n_states, model_path, cluster_path)

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        arr = group["mid"].values
        labels = np.full(len(group), np.nan)
        if len(arr) >= seq_len:
            windows = np.lib.stride_tricks.sliding_window_view(arr, seq_len)
            windows = _preprocess(windows)
            tensor = torch.tensor(windows[:, :, None], dtype=torch.float32)
            with torch.no_grad():
                emb = model.encode(tensor).numpy()
            preds = kmeans.predict(emb)
            labels[seq_len - 1 :] = preds
        out = group.copy()
        out["regime_dl"] = labels
        return out

    if "Symbol" in df.columns:
        df = df.groupby("Symbol", group_keys=False).apply(_apply)
    else:
        df = _apply(df)

    return df
