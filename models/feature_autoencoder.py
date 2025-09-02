from __future__ import annotations

"""Simple feature autoencoder for unsupervised dimensionality reduction.

The module provides a tiny autoencoder that can be pretrained on pooled raw
feature vectors.  The encoder weights can be exported to :mod:`model_store` and
later reused to initialise downstream forecasting or reinforcement-learning
models.  The design mirrors the lightweight :mod:`models.ts2vec` utilities used
throughout the repository so tests can run quickly on CPU-only environments.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import model_store


class FeatureAutoencoder(nn.Module):
    """Two-layer feed-forward autoencoder."""

    def __init__(self, input_dim: int, embed_dim: int = 8) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, embed_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


@dataclass
class TrainResult:
    model: FeatureAutoencoder
    version_id: Optional[str]


def train_feature_autoencoder(
    data: torch.Tensor,
    embed_dim: int = 8,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    store_dir: Optional[Path] = None,
) -> TrainResult:
    """Train the autoencoder and persist encoder weights via :mod:`model_store`.

    Parameters
    ----------
    data:
        Tensor of shape ``(N, D)`` containing unlabeled feature vectors.
    embed_dim:
        Size of the latent embedding.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Optimiser learning rate.
    store_dir:
        Optional model store directory.  Defaults to the global store.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = FeatureAutoencoder(data.shape[1], embed_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    last_loss = torch.tensor(0.0)
    for _ in range(max(1, epochs)):
        for (x,) in loader:
            x = x.to(device)
            opt.zero_grad()
            recon = model(x)
            loss = loss_fn(recon, x)
            loss.backward()
            opt.step()
            last_loss = loss.detach().cpu()
    model.eval()

    version_id: Optional[str] = None
    try:  # pragma: no cover - saving is best effort
        version_id = model_store.save_model(
            model.encoder.state_dict(),
            training_config={
                "feature_autoencoder": True,
                "input_dim": data.shape[1],
                "embed_dim": embed_dim,
            },
            performance={"loss": float(last_loss)},
            store_dir=store_dir,
        )
    except Exception:
        pass
    return TrainResult(model, version_id)


def load_pretrained_feature_autoencoder(
    store_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Load the latest saved feature autoencoder encoder state_dict."""

    try:
        versions = model_store.list_versions(store_dir)
    except Exception:  # pragma: no cover - store may not exist
        return None
    for meta in reversed(versions):
        cfg = meta.get("training_config", {})
        if cfg.get("feature_autoencoder"):
            state, _ = model_store.load_model(meta["version_id"], store_dir)
            if isinstance(state, dict):
                return state
    return None


def initialize_model_with_feature_ae(
    model: nn.Module, store_dir: Optional[Path] = None
) -> nn.Module:
    """Initialise ``model`` with weights from a pretrained feature autoencoder."""

    state = load_pretrained_feature_autoencoder(store_dir)
    if not state:
        return model
    try:
        model.load_state_dict(state, strict=False)
    except Exception:
        for name, param in model.state_dict().items():
            if name in state and state[name].shape == param.shape:
                param.copy_(state[name])
    return model


__all__ = [
    "FeatureAutoencoder",
    "train_feature_autoencoder",
    "load_pretrained_feature_autoencoder",
    "initialize_model_with_feature_ae",
    "TrainResult",
]
