"""Synthetic time-series generation using a simple GAN.

This module implements a lightweight generative adversarial network (GAN)
for learning joint price dynamics.  It is intentionally compact so it can be
trained within unit tests.  The model operates on sliding windows of a price
series and learns to generate sequences that preserve basic statistics such as
volatility and auto-correlation.

The primary entry point is :class:`TimeSeriesGAN` with two main methods:

``fit(series)``
    Train the GAN on an array-like object of shape ``(T, n_features)``.

``generate(length)``
    Produce a synthetic series of ``length`` observations following the learned
    dynamics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

# The model depends on PyTorch.  The repository already uses PyTorch for
# reinforcement learning components so it is expected to be available.
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader


class _WindowDataset(Dataset):
    """Simple sliding window dataset used for training."""

    def __init__(self, array: np.ndarray, seq_len: int) -> None:
        self.array = array.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:  # pragma: no cover - trivial
        return max(len(self.array) - self.seq_len, 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.array[idx : idx + self.seq_len]
        return torch.from_numpy(window)


class _Generator(nn.Module):
    def __init__(self, latent_dim: int, seq_len: int, n_features: int, hidden: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.n_features = n_features
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, seq_len * n_features),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        x = self.net(z)
        return x.view(-1, self.seq_len, self.n_features)


class _Discriminator(nn.Module):
    def __init__(self, seq_len: int, n_features: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(seq_len * n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        flat = x.view(x.size(0), -1)
        return self.net(flat)


@dataclass
class TimeSeriesGAN:
    """Lightweight GAN for synthetic time-series generation."""

    seq_len: int = 20
    latent_dim: int = 16
    hidden_dim: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.gen = _Generator(self.latent_dim, self.seq_len, 1, self.hidden_dim).to(self.device)
        self.disc = _Discriminator(self.seq_len, 1, self.hidden_dim).to(self.device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, series: Iterable[float], epochs: int = 100, batch_size: int = 32) -> None:
        """Train the GAN on a univariate price series."""
        arr = np.asarray(series, dtype=np.float32).reshape(-1, 1)
        dataset = _WindowDataset(arr, self.seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        opt_g = optim.Adam(self.gen.parameters(), lr=1e-3)
        opt_d = optim.Adam(self.disc.parameters(), lr=1e-3)
        loss_fn = nn.BCELoss()

        for _ in range(epochs):
            for real in loader:
                real = real.to(self.device)
                bsz = real.size(0)

                # Train discriminator
                z = torch.randn(bsz, self.latent_dim, device=self.device)
                fake = self.gen(z).detach()
                opt_d.zero_grad()
                loss_d = loss_fn(self.disc(real), torch.ones(bsz, 1, device=self.device))
                loss_d += loss_fn(self.disc(fake), torch.zeros(bsz, 1, device=self.device))
                loss_d.backward()
                opt_d.step()

                # Train generator
                z = torch.randn(bsz, self.latent_dim, device=self.device)
                fake = self.gen(z)
                opt_g.zero_grad()
                loss_g = loss_fn(self.disc(fake), torch.ones(bsz, 1, device=self.device))
                loss_g.backward()
                opt_g.step()

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------
    def generate(self, length: int) -> np.ndarray:
        """Generate a synthetic series of ``length`` observations."""
        segments = []
        while sum(len(s) for s in segments) < length:
            z = torch.randn(1, self.latent_dim, device=self.device)
            with torch.no_grad():
                seg = self.gen(z).cpu().numpy()[0]
            segments.append(seg)
        out = np.concatenate(segments, axis=0)[:length]
        return out.squeeze()


__all__ = ["TimeSeriesGAN"]
