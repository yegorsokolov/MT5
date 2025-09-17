"""Utilities for sizing trading positions based on risk targets."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math

from analytics.metrics_store import record_metric
from risk.funding_costs import fetch_funding_info


logger = logging.getLogger(__name__)


@dataclass
class PositionSizer:
    """Compute trade sizes using Kelly, VaR/ES limits or volatility targeting."""

    capital: float
    method: str = "kelly"
    target_vol: float = 0.01
    odds: float = 1.0
    max_martingale_multiplier: float = 1.5
    weights: dict[str, float] | None = field(default=None, init=False)
    _last_size: dict[str, float] = field(default_factory=dict, init=False)
    _oversized: dict[str, bool] = field(default_factory=dict, init=False)

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
        expected_return: float | None = None,
        predicted_volatility: float | None = None,
    ) -> float:
        """Return position size based on configured sizing method.

        Parameters
        ----------
        prob : float
            Expected win probability of the trade.
        symbol : str, optional
            Asset symbol used when ``weights`` were supplied and to fetch funding info.
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
        expected_return : float, optional
            Forecasted return magnitude used for logging/diagnostics.
        predicted_volatility : float, optional
            Forecasted volatility used to scale the Kelly allocation.
        """
        weight = 1.0
        if self.weights and symbol is not None:
            weight = self.weights.get(symbol, 0.0)
        capital = self.capital * weight
        confidence = max(0.0, min(confidence, 1.0))
        raw_vol_input = volatility
        if raw_vol_input is None and predicted_volatility is not None:
            raw_vol_input = predicted_volatility
        if self.method == "kelly":
            frac = self.kelly_fraction(prob)
            base_size = capital * frac
            target = base_size
            realized = base_size
            if predicted_volatility is not None and predicted_volatility > 0:
                scale = self.target_vol / max(predicted_volatility, 1e-12)
                base_size *= scale
                target *= scale
                realized = base_size * predicted_volatility
            elif volatility is not None:
                realized = base_size * volatility
        elif self.method == "var" and var is not None:
            base_size = capital * (self.target_vol / max(var, 1e-12))
            target = capital * self.target_vol
            realized = base_size * var
        elif self.method == "es" and es is not None:
            base_size = capital * (self.target_vol / max(es, 1e-12))
            target = capital * self.target_vol
            realized = base_size * es
        else:
            if raw_vol_input is None:
                return 0.0
            base_size = self.volatility_target(raw_vol_input, capital)
            target = capital * self.target_vol
            realized = base_size * raw_vol_input
        size = base_size * confidence
        slip_factor = 1.0
        if slippage is not None and slippage > 0:
            slip_factor = 1.0 / (1.0 + slippage)
            size *= slip_factor
        if liquidity is not None:
            size = min(size, liquidity)
        if symbol is not None:
            last = self._last_size.get(symbol, 0.0)
            oversized = self._oversized.get(symbol, False)
            if last > 0 and abs(size) > last * self.max_martingale_multiplier:
                if not oversized:
                    self._oversized[symbol] = True
                else:
                    size = math.copysign(last * self.max_martingale_multiplier, size)
            else:
                self._oversized[symbol] = False
            self._last_size[symbol] = abs(size)
        fund_cost = 0.0
        margin_required = 0.0
        margin_avail = self.capital
        if symbol is not None:
            try:
                info = fetch_funding_info(symbol)
                fund_cost = abs(size) * info.swap_rate
                size = max(0.0, size - fund_cost)
                margin_required = abs(size) * info.margin_requirement
                margin_avail = info.available_margin
                if info.margin_requirement > 0 and margin_required > margin_avail:
                    size = margin_avail / info.margin_requirement
                    margin_required = margin_avail
            except Exception:
                pass
        try:
            record_metric("target_risk", target)
            record_metric("realized_risk", realized)
            record_metric("adj_target_risk", target * confidence)
            record_metric("adj_realized_risk", realized * confidence)
            record_metric("slip_adj_target_risk", target * confidence * slip_factor)
            record_metric("slip_adj_realized_risk", realized * confidence * slip_factor)
            record_metric("expected_funding_cost", fund_cost)
            record_metric("margin_required", margin_required)
            record_metric("margin_available", margin_avail)
            if expected_return is not None:
                record_metric("expected_return", expected_return)
            if predicted_volatility is not None:
                record_metric("predicted_volatility", predicted_volatility)
        except Exception:
            pass
        logger.info(
            "Position size computed: base=%.4f adjusted=%.4f target=%.4f adj_target=%.4f slip=%.4f liq=%.4f conf=%.2f exp=%.4f pred_vol=%.4f",
            base_size,
            size,
            target,
            target * confidence * slip_factor,
            slippage or 0.0,
            float(liquidity) if liquidity is not None else float("nan"),
            confidence,
            expected_return or 0.0,
            predicted_volatility or (volatility or 0.0),
        )
        return size

    # ------------------------------------------------------------------
    # Position splitting
    # ------------------------------------------------------------------
    def split_size(self, symbol: str | None, size: float) -> list[float]:
        """Split ``size`` into chunks obeying the martingale multiplier cap.

        Parameters
        ----------
        symbol:
            Asset symbol used to track the last trade size.  If ``None`` the
            ``size`` is returned unchanged in a single chunk.
        size:
            Desired position size.

        Returns
        -------
        list[float]
            Sequence of chunk sizes whose sum equals ``size``.  Each chunk is
            limited so that it does not exceed ``max_martingale_multiplier``
            times the previous chunk.
        """

        if symbol is None:
            return [size]

        chunks: list[float] = []
        remaining = abs(size)
        direction = math.copysign(1.0, size)
        last = self._last_size.get(symbol, 0.0)

        cap = last * self.max_martingale_multiplier
        if last <= 0 or remaining <= cap:
            self._last_size[symbol] = remaining
            self._oversized[symbol] = False
            return [size]

        while remaining > cap:
            chunks.append(direction * cap)
            remaining -= cap
            last = cap
            cap = last * self.max_martingale_multiplier

        chunks.append(direction * remaining)
        self._last_size[symbol] = remaining
        self._oversized[symbol] = False
        return chunks

