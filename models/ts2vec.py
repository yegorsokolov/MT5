from __future__ import annotations

"""Simplified TS2Vec-style encoder with masked-token pretraining.

The implementation focuses on light-weight components so unit tests can run
quickly on CPU only environments.  The encoder operates on windows containing
multiple symbols (features) and learns to reconstruct randomly masked time
steps.  Trained weights can be exported to :mod:`model_store` and later used to
bootstrap forecasting or reinforcement-learning models.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import model_store


def _mask_windows(x: torch.Tensor, mask_ratio: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask a percentage of time steps in ``x``.

    Parameters
    ----------
    x:
        Tensor of shape ``(batch, seq_len, features)``.
    mask_ratio:
        Fraction of time steps to mask.  Each sample has the same mask across
        all feature dimensions to simulate missing multi-symbol observations.
    Returns
    -------
    tuple of masked tensor and boolean mask of shape ``(batch, seq_len, 1)``.
    """
    batch, seq_len, _ = x.shape
    mask = torch.rand(batch, seq_len, device=x.device) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask.unsqueeze(-1)


class TS2VecEncoder(nn.Module):
    """Tiny GRU based encoder/decoder used for masked-token training."""

    def __init__(self, input_dim: int, hidden_dim: int = 32) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.rnn(x)
        recon = self.decoder(h)
        return h, recon


@dataclass
class TrainResult:
    model: TS2VecEncoder
    version_id: Optional[str]


def train_ts2vec_encoder(
    data: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    mask_ratio: float = 0.15,
    store_dir: Optional[Path] = None,
) -> TrainResult:
    """Train the TS2Vec encoder and persist it via :mod:`model_store`.

    Parameters
    ----------
    data:
        Tensor of shape ``(N, L, D)`` containing unlabeled windows.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    lr:
        Learning rate.
    mask_ratio:
        Probability of masking a given time step.
    store_dir:
        Optional model store directory.  Defaults to the global store.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    encoder = TS2VecEncoder(data.shape[2]).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    encoder.train()
    last_loss = torch.tensor(0.0)
    for _ in range(max(1, epochs)):
        for (x,) in loader:
            x = x.to(device)
            x_masked, mask = _mask_windows(x, mask_ratio)
            _, recon = encoder(x_masked)
            loss = ((recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.detach().cpu()
    encoder.eval()

    version_id: Optional[str] = None
    try:  # pragma: no cover - saving is best effort
        version_id = model_store.save_model(
            encoder.state_dict(),
            training_config={
                "ts2vec": True,
                "input_dim": data.shape[2],
                "seq_len": data.shape[1],
            },
            performance={"loss": float(last_loss)},
            store_dir=store_dir,
        )
    except Exception:
        pass
    return TrainResult(encoder, version_id)


def load_pretrained_ts2vec_encoder(store_dir: Optional[Path] = None) -> Optional[dict]:
    """Load the latest saved TS2Vec encoder state_dict."""
    try:
        versions = model_store.list_versions(store_dir)
    except Exception:  # pragma: no cover - store may not exist
        return None
    for meta in reversed(versions):
        cfg = meta.get("training_config", {})
        if cfg.get("ts2vec"):
            state, _ = model_store.load_model(meta["version_id"], store_dir)
            if isinstance(state, dict):
                return state
    return None


def initialize_model_with_ts2vec(model: nn.Module, store_dir: Optional[Path] = None) -> nn.Module:
    """Initialise ``model`` with weights from a pretrained TS2Vec encoder."""
    state = load_pretrained_ts2vec_encoder(store_dir)
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
    "TS2VecEncoder",
    "train_ts2vec_encoder",
    "load_pretrained_ts2vec_encoder",
    "initialize_model_with_ts2vec",
    "TrainResult",
]
