from __future__ import annotations

"""Masked-segment encoder for multi-instrument time series.

This module implements a compact GRU based encoder/decoder trained by masking
contiguous time segments across all instruments.  The model is intended for
unsupervised representation learning on historical data.  Trained weights are
stored via :mod:`model_store` so downstream forecasting and reinforcement-
learning models can be initialised from the learned representations.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import model_store


def _mask_segments(
    x: torch.Tensor,
    mask_ratio: float,
    segment_len: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask contiguous time segments in ``x``.

    Parameters
    ----------
    x:
        Tensor of shape ``(batch, seq_len, features)``.
    mask_ratio:
        Approximate fraction of the sequence to mask.
    segment_len:
        Length of each masked segment.  The same mask is applied across all
        feature dimensions to simulate simultaneous data outages for multiple
        instruments.
    Returns
    -------
    tuple of masked tensor and boolean mask ``(batch, seq_len, 1)``.
    """
    batch, seq_len, feat = x.shape
    mask = torch.zeros(batch, seq_len, 1, dtype=torch.bool, device=x.device)
    num_segments = max(1, int(seq_len * mask_ratio / max(1, segment_len)))
    for b in range(batch):
        for _ in range(num_segments):
            start = int(torch.randint(0, max(1, seq_len - segment_len + 1), (1,), device=x.device))
            mask[b, start : start + segment_len, 0] = True
    x_masked = x.clone()
    x_masked[mask.expand(-1, -1, feat)] = 0.0
    return x_masked, mask


class TSMaskedEncoder(nn.Module):
    """Tiny GRU based encoder/decoder for masked-segment reconstruction."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.decoder = nn.Linear(hidden_dim, input_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h, _ = self.rnn(x)
        recon = self.decoder(h)
        return h, recon


@dataclass
class TrainResult:
    model: TSMaskedEncoder
    version_id: Optional[str]


def train_ts_masked_encoder(
    data: torch.Tensor,
    epochs: int = 20,
    batch_size: int = 32,
    lr: float = 1e-3,
    mask_ratio: float = 0.2,
    segment_len: int = 4,
    store_dir: Optional[Path] = None,
) -> TrainResult:
    """Pre-train the encoder on unlabeled windows and persist to the store."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = TensorDataset(data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    encoder = TSMaskedEncoder(data.shape[2]).to(device)
    opt = torch.optim.Adam(encoder.parameters(), lr=lr)

    encoder.train()
    last_loss = torch.tensor(0.0)
    for _ in range(max(1, epochs)):
        for (x,) in loader:
            x = x.to(device)
            x_masked, mask = _mask_segments(x, mask_ratio, segment_len)
            _, recon = encoder(x_masked)
            loss = ((recon - x) ** 2 * mask).sum() / mask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.detach().cpu()
    encoder.eval()

    version_id: Optional[str] = None
    try:  # pragma: no cover - best effort persistence
        version_id = model_store.save_model(
            encoder.state_dict(),
            training_config={
                "ts_masked": True,
                "input_dim": data.shape[2],
                "seq_len": data.shape[1],
                "segment_len": segment_len,
            },
            performance={"loss": float(last_loss)},
            store_dir=store_dir,
        )
    except Exception:
        pass
    return TrainResult(encoder, version_id)


def load_pretrained_ts_masked_encoder(
    store_dir: Optional[Path] = None,
) -> Optional[dict]:
    """Return the latest saved encoder state_dict if available."""
    try:
        versions = model_store.list_versions(store_dir)
    except Exception:  # pragma: no cover
        return None
    for meta in reversed(versions):
        cfg = meta.get("training_config", {})
        if cfg.get("ts_masked"):
            state, _ = model_store.load_model(meta["version_id"], store_dir)
            if isinstance(state, dict):
                return state
    return None


def initialize_model_with_ts_masked_encoder(
    model: nn.Module,
    store_dir: Optional[Path] = None,
) -> nn.Module:
    """Initialise ``model`` with weights from the masked-segment encoder."""
    state = load_pretrained_ts_masked_encoder(store_dir)
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
    "TSMaskedEncoder",
    "train_ts_masked_encoder",
    "load_pretrained_ts_masked_encoder",
    "initialize_model_with_ts_masked_encoder",
    "TrainResult",
]
