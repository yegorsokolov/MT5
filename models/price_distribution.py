"""Return distribution modeling using a mixture density network."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

try:  # optional dependency
    import torch
    from torch import nn
except Exception as e:  # pragma: no cover - torch might be missing
    torch = None  # type: ignore
    nn = object  # type: ignore


@dataclass
class PriceDistributionModel:
    """Mixture density network producing full return distributions.

    Parameters
    ----------
    input_dim:
        Number of input features.
    hidden_dim:
        Hidden layer size for the neural network.
    n_components:
        Number of Gaussian mixture components.
    """

    input_dim: int
    hidden_dim: int = 32
    n_components: int = 3

    def __post_init__(self) -> None:
        if torch is None:  # pragma: no cover - handled gracefully
            raise ImportError("PyTorch is required for PriceDistributionModel")
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_components * 3),
        )

    def _forward(self, X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        out = self.net(X)
        logits = out[:, : self.n_components]
        means = out[:, self.n_components : 2 * self.n_components]
        log_std = out[:, 2 * self.n_components :]
        weights = torch.softmax(logits, dim=-1)
        std = torch.exp(log_std)
        return weights, means, std

    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, lr: float = 1e-3) -> None:
        """Fit the mixture density network using maximum likelihood."""
        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for _ in range(epochs):
            opt.zero_grad()
            w, m, s = self._forward(X_t)
            dist = torch.distributions.Normal(m, s)
            log_prob = dist.log_prob(y_t)
            loss = -(torch.log(w + 1e-12) + log_prob).logsumexp(dim=1).mean()
            loss.backward()
            opt.step()
        self.net.eval()

    def predict_params(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return mixture weights, means and std deviations for ``X``."""
        with torch.no_grad():
            X_t = torch.tensor(X, dtype=torch.float32)
            w, m, s = self._forward(X_t)
        return w.numpy(), m.numpy(), s.numpy()

    def sample(self, X: np.ndarray, n_samples: int = 1000) -> np.ndarray:
        """Draw samples from the predicted return distribution for ``X``."""
        weights, means, stds = self.predict_params(X)
        samples = []
        rng = np.random.default_rng()
        for w, m, s in zip(weights, means, stds):
            comps = rng.choice(self.n_components, size=n_samples, p=w)
            draws = rng.normal(m[comps], s[comps])
            samples.append(draws)
        return np.asarray(samples)

    def percentile(self, X: np.ndarray, q: float, n_samples: int = 1000) -> np.ndarray:
        """Return the ``q`` percentile of the predicted distribution for ``X``."""
        samples = self.sample(X, n_samples=n_samples)
        return np.percentile(samples, q * 100, axis=1)

    def expected_shortfall(self, X: np.ndarray, alpha: float, n_samples: int = 1000) -> np.ndarray:
        """Return the expected shortfall at ``alpha`` level for ``X``."""
        samples = self.sample(X, n_samples=n_samples)
        var = np.percentile(samples, alpha * 100, axis=1)
        es = []
        for s, v in zip(samples, var):
            below = s[s <= v]
            es.append(below.mean() if len(below) > 0 else v)
        return np.asarray(es)
