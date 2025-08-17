from __future__ import annotations

"""Simple contrastive encoder using a SimCLR-style objective.

The module provides utilities to pretrain an encoder on unlabeled windows of
multiple symbols and to bootstrap downstream models with the learnt
representation.  The implementation is intentionally lightweight so tests run
quickly and without heavy dependencies.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import math
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import model_store


def _augment(x: torch.Tensor) -> torch.Tensor:
    """Return a slightly noised copy of ``x`` used as data augmentation."""
    noise = torch.randn_like(x) * 0.01
    return x + noise


def _info_nce(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Compute a simple NT-Xent loss."""
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(len(z1), device=z1.device)
    return nn.functional.cross_entropy(logits, labels)


class ContrastiveEncoder(nn.Module):
    """Tiny MLP based encoder used for representation learning."""

    def __init__(self, input_dim: int, hidden_dim: int = 32, projection_dim: int = 16) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )
        self.projector = nn.Sequential(nn.ReLU(), nn.Linear(projection_dim, projection_dim))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x)
        z = self.projector(h)
        return h, z


@dataclass
class TrainResult:
    model: ContrastiveEncoder
    version_id: Optional[str]


def train_contrastive_encoder(
    data: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    store_dir: Optional[Path] = None,
) -> TrainResult:
    """Train the contrastive encoder and persist it via :mod:`model_store`.

    Parameters
    ----------
    data:
        Tensor of shape ``(N, D)`` containing unlabeled windows.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Learning rate.
    store_dir:
        Optional model store directory.  Defaults to the global store.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    encoder = ContrastiveEncoder(data.shape[1]).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    encoder.train()
    for _ in range(max(1, epochs)):
        for (x,) in loader:
            x = x.to(device)
            x1 = _augment(x)
            x2 = _augment(x)
            _, z1 = encoder(x1)
            _, z2 = encoder(x2)
            loss = _info_nce(z1, z2)
            opt.zero_grad()
            loss.backward()
            opt.step()
    encoder.eval()

    version_id: Optional[str] = None
    try:  # pragma: no cover - saving is best effort
        version_id = model_store.save_model(
            encoder.state_dict(),
            training_config={"contrastive_encoder": True, "input_dim": data.shape[1]},
            performance={"loss": float(loss.detach().cpu())},
            store_dir=store_dir,
        )
    except Exception:
        pass
    return TrainResult(encoder, version_id)


def load_pretrained_contrastive_encoder(store_dir: Optional[Path] = None) -> Optional[dict]:
    """Load the latest saved contrastive encoder state_dict."""
    try:
        versions = model_store.list_versions(store_dir)
    except Exception:  # pragma: no cover - store may not exist
        return None
    for meta in reversed(versions):
        cfg = meta.get("training_config", {})
        if cfg.get("contrastive_encoder"):
            state, _ = model_store.load_model(meta["version_id"], store_dir)
            if isinstance(state, dict):
                return state
    return None


def initialize_model_with_contrastive(model: nn.Module, store_dir: Optional[Path] = None) -> nn.Module:
    """Initialise ``model`` with weights from a pretrained contrastive encoder."""
    state = load_pretrained_contrastive_encoder(store_dir)
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
    "ContrastiveEncoder",
    "train_contrastive_encoder",
    "load_pretrained_contrastive_encoder",
    "initialize_model_with_contrastive",
    "TrainResult",
]
