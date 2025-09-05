"""Diffusion based long-horizon scenario generation.

This module implements a very small denoising diffusion probabilistic model
(DDPM) for one dimensional time-series.  The goal is not state of the art
performance but to provide a deterministic and easily testable component for
unit tests.  The model learns the distribution of sliding windows of a PnL
series and can later sample synthetic paths including crash/recovery patterns.

Example
-------
>>> model = ScenarioDiffusion(seq_len=100)
>>> model.fit(series)        # ``series`` is an iterable of floats
>>> path = model.sample_crash_recovery(100)

The returned ``path`` can be fed into :mod:`stress_tests.scenario_runner` using
its ``synthetic_generator`` argument.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


class _WindowDataset(Dataset):
    """Simple sliding window dataset for long horizon series."""

    def __init__(self, array: np.ndarray, seq_len: int) -> None:
        self.array = array.astype(np.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:  # pragma: no cover - straightforward
        return max(len(self.array) - self.seq_len, 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        window = self.array[idx : idx + self.seq_len]
        return torch.from_numpy(window)


class _NoisePredictor(nn.Module):
    """Small MLP used to predict noise at a given diffusion step."""

    def __init__(self, seq_len: int, hidden: int) -> None:
        super().__init__()
        self.seq_len = seq_len
        self.net = nn.Sequential(
            nn.Linear(seq_len + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, seq_len),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:  # pragma: no cover - simple
        t = t.float().unsqueeze(1) / 1000.0
        inp = torch.cat([x, t], dim=1)
        return self.net(inp)


@dataclass
class ScenarioDiffusion:
    """Minimal diffusion model for time-series scenario generation."""

    seq_len: int
    n_steps: int = 100
    hidden_dim: int = 64
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model = _NoisePredictor(self.seq_len, self.hidden_dim).to(self.device)
        betas = torch.linspace(1e-4, 0.02, self.n_steps, device=self.device)
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, series: Iterable[float], epochs: int = 50, batch_size: int = 32) -> None:
        """Train the diffusion model on a univariate series."""
        arr = np.asarray(list(series), dtype=np.float32)
        dataset = _WindowDataset(arr, self.seq_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = optim.Adam(self.model.parameters(), lr=1e-3)

        for _ in range(epochs):
            for x0 in loader:
                x0 = x0.to(self.device)
                bsz = x0.size(0)
                t = torch.randint(0, self.n_steps, (bsz,), device=self.device)
                noise = torch.randn_like(x0)
                a = self.sqrt_alphas_cumprod[t].unsqueeze(1)
                b = self.sqrt_one_minus_alphas_cumprod[t].unsqueeze(1)
                xt = a * x0 + b * noise
                pred = self.model(xt, t)
                loss = (noise - pred).pow(2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def generate(self, length: int) -> np.ndarray:
        """Generate an unconditional synthetic path of ``length`` observations."""
        x = torch.randn(1, self.seq_len, device=self.device)
        for i in reversed(range(self.n_steps)):
            t = torch.full((1,), i, device=self.device)
            pred = self.model(x, t)
            beta = self.betas[i]
            alpha = self.alphas[i]
            alpha_bar = self.alphas_cumprod[i]
            coeff1 = 1 / torch.sqrt(alpha)
            coeff2 = beta / torch.sqrt(1 - alpha_bar)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = coeff1 * (x - coeff2 * pred) + torch.sqrt(beta) * noise
        path = x[0].detach().cpu().numpy()
        if length != self.seq_len:
            path = np.interp(
                np.linspace(0, self.seq_len - 1, length),
                np.arange(self.seq_len),
                path,
            )
        return path.astype(float)

    def sample_crash_recovery(
        self, length: int, crash: float = -0.3, recovery: float = 0.05
    ) -> np.ndarray:
        """Generate a path with an initial crash followed by recovery."""
        path = self.generate(length)
        if length > 0:
            path[0] += crash
            for i in range(1, length):
                path[i] += (0 - path[i - 1]) * recovery
        return path


__all__ = ["ScenarioDiffusion"]
