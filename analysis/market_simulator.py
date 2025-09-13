"""Adversarial market simulator.

This module implements a simple generative adversarial model that perturbs
historical price sequences in order to maximise the loss of a trading agent.
It is intentionally lightweight so it can be used within unit tests.  The
simulator learns a small neural network that proposes perturbations of a price
series.  The network is updated via gradient ascent on the agent's loss,
analogous to the generator in a GAN.

The primary entry point is :class:`AdversarialMarketSimulator` with a single
method :meth:`perturb` which adjusts a given price series.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch
from torch import nn, optim


@dataclass
class AdversarialMarketSimulator:
    """Generate adversarial price paths to stress test agents.

    Parameters
    ----------
    seq_len:
        Length of the price sequence the simulator operates on.  The same
        length is expected for all calls to :meth:`perturb`.
    hidden:
        Width of the hidden layer of the perturbation network.
    device:
        Torch device to use.  Defaults to CPU so tests remain lightweight.
    lr:
        Learning rate for the perturbation network's optimiser.
    eps:
        Maximum magnitude of the perturbation applied to each price element.
    """

    seq_len: int = 32
    hidden: int = 32
    device: str = "cpu"
    lr: float = 1e-2
    eps: float = 0.05

    def __post_init__(self) -> None:
        self.net = nn.Sequential(
            nn.Linear(self.seq_len, self.hidden),
            nn.Tanh(),
            nn.Linear(self.hidden, self.seq_len),
        ).to(self.device)
        self.opt = optim.Adam(self.net.parameters(), lr=self.lr)

    # ------------------------------------------------------------------
    def perturb(self, prices: Iterable[float], policy: nn.Module, steps: int = 10) -> np.ndarray:
        """Return a price series adversarial to ``policy``.

        The method performs ``steps`` rounds of gradient ascent on a neural
        network that outputs perturbations.  The resulting perturbed sequence
        is returned as a NumPy array.
        """

        prices_t = torch.tensor(list(prices), dtype=torch.float32, device=self.device)
        if prices_t.numel() != self.seq_len:
            raise ValueError(f"Expected price series of length {self.seq_len}")

        for _ in range(steps):
            self.opt.zero_grad()
            noise = torch.tanh(self.net(prices_t)) * self.eps
            adv = prices_t + noise
            loss = -policy.loss(adv)
            loss.backward()
            self.opt.step()

        with torch.no_grad():
            noise = torch.tanh(self.net(prices_t)) * self.eps
            adv = prices_t + noise
        return adv.cpu().numpy()


__all__ = ["AdversarialMarketSimulator"]
