"""Utilities for sizing trading positions based on risk targets."""

from __future__ import annotations

from dataclasses import dataclass
import logging

from metrics import TARGET_RISK, REALIZED_RISK


logger = logging.getLogger(__name__)


@dataclass
class PositionSizer:
    """Compute trade sizes using Kelly or volatility targeting."""

    capital: float
    method: str = "kelly"
    target_vol: float = 0.01
    odds: float = 1.0

    def kelly_fraction(self, prob: float) -> float:
        """Return Kelly fraction for win probability ``prob`` and payoff ``odds``."""
        return max(0.0, min((self.odds * prob - (1 - prob)) / self.odds, 1.0))

    def volatility_target(self, volatility: float) -> float:
        """Return position size to hit ``target_vol`` given current ``volatility``."""
        if volatility <= 0:
            return 0.0
        return self.capital * (self.target_vol / volatility)

    def size(self, prob: float, volatility: float | None = None) -> float:
        """Return position size based on configured sizing method."""
        if self.method == "kelly":
            frac = self.kelly_fraction(prob)
            size = self.capital * frac
            target = size
            realized = size
        else:
            if volatility is None:
                return 0.0
            size = self.volatility_target(volatility)
            target = self.capital * self.target_vol
            realized = size * volatility
        TARGET_RISK.set(target)
        REALIZED_RISK.set(realized)
        logger.info("Position size computed: size=%.4f target=%.4f", realized, target)
        return size
