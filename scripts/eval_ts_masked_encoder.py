"""Evaluate the benefits of masked-segment pretraining.

The script pre-trains :class:`TSMaskedEncoder` on synthetic unlabeled data and
compares the convergence of a simple forecasting head initialised with the
pretrained weights against a randomly initialised baseline.
"""

from __future__ import annotations
import sys
from pathlib import Path

import torch
from torch import nn


sys.path.append(str(Path(__file__).resolve().parents[1]))
from models.ts_masked_encoder import (
    TSMaskedEncoder,
    initialize_model_with_ts_masked_encoder,
    train_ts_masked_encoder,
)


def _make_dataset(n_pre: int = 200, n_shift: int = 40, seq_len: int = 8):
    t = torch.linspace(0, 20, steps=n_pre + n_shift + seq_len + 1)
    f1 = torch.sin(t)
    f2 = torch.cos(t)
    f1[n_pre:] *= 2  # regime change
    features = torch.stack([f1, f2], dim=1)
    windows = torch.stack([features[i : i + seq_len] for i in range(n_pre + n_shift)])
    targets = f1[seq_len : n_pre + n_shift + seq_len].unsqueeze(-1)
    return windows[:n_pre], windows[n_pre:], targets[n_pre:]


class Forecaster(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = TSMaskedEncoder(2)
        self.head = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, _ = self.encoder(x)
        return self.head(h[:, -1])


def _train_one(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.MSELoss()
    opt.zero_grad()
    pred = model(x)
    loss = loss_fn(pred, y)
    loss.backward()
    opt.step()
    return float(loss.detach())


def main() -> None:
    torch.manual_seed(0)
    pre_windows, shift_windows, shift_targets = _make_dataset()
    train_ts_masked_encoder(pre_windows, epochs=5, batch_size=16)

    model_pre = Forecaster()
    initialize_model_with_ts_masked_encoder(model_pre.encoder)
    model_rand = Forecaster()

    loss_pre = _train_one(model_pre, shift_windows[:10], shift_targets[:10])
    loss_rand = _train_one(model_rand, shift_windows[:10], shift_targets[:10])

    print(f"val_loss_pretrained={loss_pre:.4f}")
    print(f"val_loss_random={loss_rand:.4f}")
    if loss_pre < loss_rand:
        print("Pretraining improves convergence.")
    else:
        print("No improvement observed.")


if __name__ == "__main__":
    main()
