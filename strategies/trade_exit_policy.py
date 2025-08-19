"""Model-based trade exit policy.

This module trains a model on historical trades to estimate whether the
expected value of continuing to hold a position is positive.  At runtime the
policy can keep track of open trades and recommend closing them when the
expected value turns negative.

Two model variants are provided:

``GBMExitPolicy``
    Lightweight gradient-boosting classifier.
``RLExitPolicy``
    Heavier neural network that can be extended to a transformer or
    reinforcement-learning agent.  It lazily imports PyTorch so tests remain
    light-weight.

The training routine expects a dataframe that includes arbitrary feature
columns along with a ``future_return`` column describing the profit or loss that
would have been realised by holding the trade for a fixed horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier


@dataclass
class TradeExitPolicy:
    """Base policy that scores open trades for exit decisions."""

    model: Any | None = None
    open_trades: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def train(self, trades: pd.DataFrame) -> Any:
        """Train the model on historical trade data.

        Parameters
        ----------
        trades:
            DataFrame containing model features and a ``future_return`` column.
        """

        if "future_return" not in trades.columns:
            raise ValueError("trades dataframe must contain 'future_return'")
        X = trades.drop(columns=["future_return"])
        y = (trades["future_return"] > 0).astype(int)
        self.model = GradientBoostingClassifier(max_depth=3, n_estimators=50)
        self.model.fit(X, y)
        return self.model

    # ------------------------------------------------------------------
    # Live operation helpers
    # ------------------------------------------------------------------
    def register_trade(self, symbol: str, features: Dict[str, Any]) -> None:
        """Record a newly opened trade."""

        self.open_trades[symbol] = dict(features)

    def should_exit(self, symbol: str, features: Dict[str, Any]) -> bool:
        """Return ``True`` when the expected value of continuing the trade is
        negative according to the trained model."""

        if not self.model or symbol not in self.open_trades:
            return False
        X = pd.DataFrame([features])
        try:
            proba = float(self.model.predict_proba(X)[0, 1])
        except Exception:
            proba = float(self.model.predict(X)[0])
        expected_value = 2 * proba - 1
        if expected_value < 0:
            self.open_trades.pop(symbol, None)
            return True
        self.open_trades[symbol] = dict(features)
        return False


class GBMExitPolicy(TradeExitPolicy):
    """Lightweight GBM based policy."""

    pass


class RLExitPolicy(TradeExitPolicy):
    """Heavier policy using a small neural network.

    The implementation serves as a placeholder for a transformer or
    reinforcement-learning based approach.  It uses PyTorch if available.
    """

    def train(self, trades: pd.DataFrame) -> Any:  # pragma: no cover - optional heavy path
        import torch
        import torch.nn as nn
        import torch.optim as optim

        if "future_return" not in trades.columns:
            raise ValueError("trades dataframe must contain 'future_return'")
        X = torch.tensor(trades.drop(columns=["future_return"]).values, dtype=torch.float32)
        y = torch.tensor((trades["future_return"] > 0).astype(int).values, dtype=torch.long)

        model = nn.Sequential(nn.Linear(X.shape[1], 16), nn.ReLU(), nn.Linear(16, 2))
        opt = optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()
        model.train()
        for _ in range(100):
            opt.zero_grad()
            logits = model(X)
            loss = loss_fn(logits, y)
            loss.backward()
            opt.step()
        self.model = model
        return model

    def _predict_proba(self, X: pd.DataFrame) -> float:
        import torch

        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(X.values, dtype=torch.float32)
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
        return float(probs[0, 1])

    def should_exit(self, symbol: str, features: Dict[str, Any]) -> bool:
        if not self.model or symbol not in self.open_trades:
            return False
        X = pd.DataFrame([features])
        proba = self._predict_proba(X)
        expected_value = 2 * proba - 1
        if expected_value < 0:
            self.open_trades.pop(symbol, None)
            return True
        self.open_trades[symbol] = dict(features)
        return False


__all__ = ["TradeExitPolicy", "GBMExitPolicy", "RLExitPolicy"]
