"""Unsupervised VAE based market regime embedding.

This module provides a lightweight variational autoencoder (VAE) that learns a
latent representation of sliding windows of market features.  After training,
clusters in the latent space correspond to different market regimes.  The
resulting regime labels can be fed into the strategy router to adapt algorithm
weights to the prevailing market state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.cluster import KMeans


def window_features(arr: np.ndarray, window: int) -> np.ndarray:
    """Return a 2D array of sliding windows flattened into feature vectors.

    Parameters
    ----------
    arr: np.ndarray
        Array of shape ``(n_samples, n_features)``.
    window: int
        Length of each sliding window.

    Returns
    -------
    np.ndarray
        Array of shape ``(n_samples - window + 1, window * n_features)``.
    """

    if window <= 0 or window > len(arr):
        raise ValueError("window must be within (0, len(arr)]")
    windows = [arr[i - window : i].ravel() for i in range(window, len(arr) + 1)]
    return np.vstack(windows)


class _VAE(nn.Module):
    """Simple feed-forward variational autoencoder."""

    def __init__(self, input_dim: int, latent_dim: int = 2, hidden_dim: int = 32) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc2 = nn.Linear(latent_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.fc2(z))
        return self.fc3(h)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar


@dataclass
class VAERegime:
    """Wrapper around :class:`_VAE` for regime detection."""

    input_dim: int
    latent_dim: int = 2
    hidden_dim: int = 32
    device: str = "cpu"

    def __post_init__(self) -> None:
        self.model = _VAE(self.input_dim, self.latent_dim, self.hidden_dim).to(self.device)

    # Training -----------------------------------------------------------------
    def fit(
        self,
        data: np.ndarray,
        epochs: int = 50,
        batch_size: int = 32,
        lr: float = 1e-3,
    ) -> None:
        """Train the VAE on ``data``."""

        dataset = TensorDataset(torch.from_numpy(data.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        opt = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.model.train()
        for _ in range(epochs):
            for (batch,) in loader:
                batch = batch.to(self.device)
                opt.zero_grad()
                recon, mu, logvar = self.model(batch)
                recon_loss = nn.functional.mse_loss(recon, batch, reduction="sum") / len(batch)
                kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / len(batch)
                loss = recon_loss + kld
                loss.backward()
                opt.step()

    # Inference ----------------------------------------------------------------
    def transform(self, data: np.ndarray, batch_size: int = 256) -> np.ndarray:
        """Return latent embeddings for ``data``."""

        self.model.eval()
        embeddings: list[np.ndarray] = []
        dataset = TensorDataset(torch.from_numpy(data.astype(np.float32)))
        loader = DataLoader(dataset, batch_size=batch_size)
        with torch.no_grad():
            for (batch,) in loader:
                batch = batch.to(self.device)
                mu, _ = self.model.encode(batch)
                embeddings.append(mu.cpu().numpy())
        return np.vstack(embeddings)

    def assign_regimes(self, embeddings: np.ndarray, n_clusters: int = 3) -> np.ndarray:
        """Cluster ``embeddings`` to obtain discrete regime labels."""

        model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
        return model.fit_predict(embeddings)

    def fit_predict(self, data: np.ndarray, n_clusters: int = 3, **kwargs) -> np.ndarray:
        """Train the model then return regime labels for ``data``."""

        self.fit(data, **kwargs)
        embeddings = self.transform(data)
        return self.assign_regimes(embeddings, n_clusters=n_clusters)


__all__ = ["VAERegime", "window_features"]
