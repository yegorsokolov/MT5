"""Utilities for sizing trading positions based on risk targets."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from metrics import (
    TARGET_RISK,
    REALIZED_RISK,
    ADJ_TARGET_RISK,
    ADJ_REALIZED_RISK,
)


logger = logging.getLogger(__name__)


@dataclass
class PositionSizer:
    """Compute trade sizes using Kelly or volatility targeting."""

    capital: float
    method: str = "kelly"
    target_vol: float = 0.01
    odds: float = 1.0
    weights: dict[str, float] | None = field(default=None, init=False)

    def kelly_fraction(self, prob: float) -> float:
        """Return Kelly fraction for win probability ``prob`` and payoff ``odds``."""
        return max(0.0, min((self.odds * prob - (1 - prob)) / self.odds, 1.0))

    def update_weights(self, weights: dict[str, float]) -> None:
        """Set optimizer-provided asset ``weights``."""
        self.weights = weights

    def volatility_target(self, volatility: float, capital: float) -> float:
        """Return position size to hit ``target_vol`` given current ``volatility``."""
        if volatility <= 0:
            return 0.0
        return capital * (self.target_vol / volatility)

    def size(
        self,
        prob: float,
        symbol: str | None = None,
        volatility: float | None = None,
        confidence: float = 1.0,
    ) -> float:
        """Return position size based on configured sizing method."""
        weight = 1.0
        if self.weights and symbol is not None:
            weight = self.weights.get(symbol, 0.0)
        capital = self.capital * weight
        confidence = max(0.0, min(confidence, 1.0))
        if self.method == "kelly":
            frac = self.kelly_fraction(prob)
            base_size = capital * frac
            target = base_size
            realized = base_size
        else:
            if volatility is None:
                return 0.0
            base_size = self.volatility_target(volatility, capital)
            target = capital * self.target_vol
            realized = base_size * volatility
        size = base_size * confidence
        TARGET_RISK.set(target)
        REALIZED_RISK.set(realized)
        ADJ_TARGET_RISK.set(target * confidence)
        ADJ_REALIZED_RISK.set(realized * confidence)
        logger.info(
            "Position size computed: base=%.4f adjusted=%.4f target=%.4f adj_target=%.4f conf=%.2f",
            base_size,
            size,
            target,
            target * confidence,
            confidence,
        )
        return size

