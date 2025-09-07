from __future__ import annotations

"""Lightweight quantile regression models.

This module provides simple wrappers for training quantile regression models
using either gradient boosting or a small neural network.  Multiple ``alpha``
levels can be trained at once and predictions for all quantiles are returned in
a dictionary keyed by the respective ``alpha``.

The implementation purposely avoids heavy dependencies and falls back to basic
scikit‑learn estimators when optional packages such as LightGBM or PyTorch are
missing.  The API is intentionally minimal and designed for unit testing and
small offline experiments rather than production scale training.
"""

from dataclasses import dataclass, field
from typing import Dict, Iterable, Mapping, Sequence, Tuple, Any

import numpy as np

try:  # optional dependency
    import lightgbm as lgb  # type: ignore
except Exception:  # pragma: no cover - lightgbm is optional
    lgb = None  # type: ignore

try:  # optional dependency
    from sklearn.ensemble import GradientBoostingRegressor  # type: ignore
except Exception:  # pragma: no cover - sklearn may be absent
    GradientBoostingRegressor = None  # type: ignore

try:  # optional dependency
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch may be absent
    torch = None  # type: ignore
    nn = object  # type: ignore


# ---------------------------------------------------------------------------
# Gradient boosted quantile model
# ---------------------------------------------------------------------------

@dataclass
class GradientBoostedQuantile:
    """Train separate gradient boosted trees for multiple quantiles."""

    alphas: Sequence[float]
    params: Mapping[str, Any] | None = None
    models: Dict[float, Any] = field(default_factory=dict)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "GradientBoostedQuantile":
        X = np.asarray(X)
        y = np.asarray(y)
        for a in self.alphas:
            if lgb is not None:  # pragma: no branch - simple selection
                model = lgb.LGBMRegressor(objective="quantile", alpha=a, **(self.params or {}))
            else:
                if GradientBoostingRegressor is None:
                    raise ImportError("scikit-learn is required for GradientBoostedQuantile")
                model = GradientBoostingRegressor(loss="quantile", alpha=a, **(self.params or {}))
            model.fit(X, y)
            self.models[float(a)] = model
        return self

    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        preds: Dict[float, np.ndarray] = {}
        for a, m in self.models.items():
            preds[a] = np.asarray(m.predict(X))
        return preds

    def var_es(self, X: np.ndarray, alpha: float) -> Tuple[np.ndarray | None, np.ndarray | None]:
        """Return Value-at-Risk and Expected Shortfall for ``X`` at ``alpha``."""
        preds = self.predict(X)
        var = preds.get(alpha)
        es = None
        tail = [preds[a] for a in self.alphas if a <= alpha and a in preds]
        if tail:
            es = np.mean(tail, axis=0)
        return var, es


# ---------------------------------------------------------------------------
# Neural network quantile model
# ---------------------------------------------------------------------------

@dataclass
class NeuralQuantile:
    """Simple feed‑forward neural network trained with quantile loss."""

    input_dim: int
    alphas: Sequence[float]
    hidden_dim: int = 32
    epochs: int = 100
    lr: float = 1e-3
    dropout: float = 0.0
    patience: int = 10
    models: Dict[float, nn.Module] = field(default_factory=dict)
    epochs_trained: Dict[float, int] = field(default_factory=dict)

    def _build_net(self) -> nn.Module:  # pragma: no cover - small helper
        layers = [
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
        ]
        if self.dropout > 0:
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(self.hidden_dim, 1))
        return nn.Sequential(*layers)

    @staticmethod
    def _quantile_loss(pred: torch.Tensor, target: torch.Tensor, alpha: float) -> torch.Tensor:
        diff = target - pred
        return torch.max(alpha * diff, (alpha - 1) * diff).mean()

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        val: Tuple[np.ndarray, np.ndarray] | None = None,
    ) -> "NeuralQuantile":
        if torch is None:
            raise ImportError("PyTorch is required for NeuralQuantile")
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        y_t = torch.tensor(np.asarray(y), dtype=torch.float32).view(-1, 1)
        if val is not None:
            X_val_t = torch.tensor(np.asarray(val[0]), dtype=torch.float32)
            y_val_t = torch.tensor(np.asarray(val[1]), dtype=torch.float32).view(-1, 1)
        for a in self.alphas:
            net = self._build_net()
            opt = torch.optim.Adam(net.parameters(), lr=self.lr)
            best_val = float("inf")
            bad_epochs = 0
            for epoch in range(self.epochs):
                opt.zero_grad()
                pred = net(X_t)
                loss = self._quantile_loss(pred, y_t, float(a))
                loss.backward()
                opt.step()
                if val is not None:
                    with torch.no_grad():
                        pred_v = net(X_val_t)
                        val_loss = self._quantile_loss(pred_v, y_val_t, float(a)).item()
                    if val_loss < best_val - 1e-6:
                        best_val = val_loss
                        bad_epochs = 0
                    else:
                        bad_epochs += 1
                        if bad_epochs >= self.patience:
                            break
            self.epochs_trained[float(a)] = epoch + 1
            net.eval()
            self.models[float(a)] = net
        return self

    def predict(self, X: np.ndarray) -> Dict[float, np.ndarray]:
        if torch is None:
            raise ImportError("PyTorch is required for NeuralQuantile")
        X_t = torch.tensor(np.asarray(X), dtype=torch.float32)
        preds: Dict[float, np.ndarray] = {}
        for a, net in self.models.items():
            with torch.no_grad():
                preds[a] = net(X_t).squeeze(-1).numpy()
        return preds

    def var_es(self, X: np.ndarray, alpha: float) -> Tuple[np.ndarray | None, np.ndarray | None]:
        preds = self.predict(X)
        var = preds.get(alpha)
        es = None
        tail = [preds[a] for a in self.alphas if a <= alpha and a in preds]
        if tail:
            es = np.mean(tail, axis=0)
        return var, es


__all__ = ["GradientBoostedQuantile", "NeuralQuantile"]
