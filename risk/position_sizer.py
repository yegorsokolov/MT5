"""Utilities for sizing trading positions based on risk targets."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

from analytics.metrics_store import record_metric


logger = logging.getLogger(__name__)


@dataclass
class PositionSizer:
    """Compute trade sizes using Kelly, VaR/ES limits or volatility targeting."""

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
        var: float | None = None,
        es: float | None = None,
        confidence: float = 1.0,
        slippage: float | None = None,
        liquidity: float | None = None,
    ) -> float:
        """Return position size based on configured sizing method.

        Parameters
        ----------
        prob : float
            Expected win probability of the trade.
        symbol : str, optional
            Asset symbol used when ``weights`` were supplied.
        volatility, var, es : float, optional
            Risk inputs used depending on the configured ``method``.
        confidence : float, default 1.0
            Multiplier representing model confidence in the signal.
        slippage : float, optional
            Estimated fractional execution cost.  Higher slippage reduces the
            returned size.
        liquidity : float, optional
            Available volume at the top of book.  The final size will not
            exceed this value.
        """
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
        elif self.method == "var" and var is not None:
            base_size = capital * (self.target_vol / max(var, 1e-12))
            target = capital * self.target_vol
            realized = base_size * var
        elif self.method == "es" and es is not None:
            base_size = capital * (self.target_vol / max(es, 1e-12))
            target = capital * self.target_vol
            realized = base_size * es
        else:
            if volatility is None:
                return 0.0
            base_size = self.volatility_target(volatility, capital)
            target = capital * self.target_vol
            realized = base_size * volatility
        size = base_size * confidence
        slip_factor = 1.0
        if slippage is not None and slippage > 0:
            slip_factor = 1.0 / (1.0 + slippage)
            size *= slip_factor
        if liquidity is not None:
            size = min(size, liquidity)
        try:
            record_metric("target_risk", target)
            record_metric("realized_risk", realized)
            record_metric("adj_target_risk", target * confidence)
            record_metric("adj_realized_risk", realized * confidence)
            record_metric("slip_adj_target_risk", target * confidence * slip_factor)
            record_metric("slip_adj_realized_risk", realized * confidence * slip_factor)
        except Exception:
            pass
        logger.info(
            "Position size computed: base=%.4f adjusted=%.4f target=%.4f adj_target=%.4f slip=%.4f liq=%.4f conf=%.2f",
            base_size,
            size,
            target,
            target * confidence * slip_factor,
            slippage or 0.0,
            float(liquidity) if liquidity is not None else float("nan"),
            confidence,
        )
        return size

