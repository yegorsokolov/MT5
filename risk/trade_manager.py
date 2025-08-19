"""Adaptive trailing stop management.

This module provides a :class:`TradeManager` capable of deriving trailing
profit and stop loss levels from price action.  The default behaviour uses an
Average True Range (ATR) based mechanism but may also incorporate externally
supplied "regime" features to widen or tighten exits when market conditions
change.  Computed thresholds are pushed to an execution layer and persisted so
that later reprocessing can compare the baseline exits with the adaptive
values used during live trading.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pandas as pd


@dataclass
class TradeManager:
    """Compute and maintain trailing stops for open trades.

    Parameters
    ----------
    execution: Any
        Object exposing ``update_order(order_id, stop_loss, take_profit)`` used
        to push updated thresholds to the trading venue.
    trade_log: Any
        Persistence layer providing ``record_thresholds`` for recording the
        default and adaptive thresholds chosen for each trade.
    atr_period: int, default ``14``
        Window used for ATR calculation.
    atr_mult: float, default ``3.0``
        Multiplier applied to the ATR to derive baseline profit/stop levels.
    vol_threshold: float, default ``1.5``
        Threshold for the ``volatility`` feature beyond which levels are widened.
    """

    execution: Any
    trade_log: Any
    atr_period: int = 14
    atr_mult: float = 3.0
    vol_threshold: float = 1.5

    def __post_init__(self) -> None:  # pragma: no cover - simple assignment
        self._last_regime: Any = None

    # ------------------------------------------------------------------
    # Level computation
    # ------------------------------------------------------------------
    def _atr(self, prices: pd.Series) -> float:
        """Return the ATR of ``prices``."""

        tr = prices.diff().abs()
        return float(tr.rolling(self.atr_period).mean().iloc[-1])

    def compute_levels(self, prices: pd.Series, features: Dict[str, Any]) -> Dict[str, float]:
        """Compute base and adaptive exit levels.

        Parameters
        ----------
        prices:
            Series of mid prices with the most recent price last.
        features:
            Mapping of feature name to value.  ``volatility`` and ``regime`` are
            recognised.
        """

        price = float(prices.iloc[-1])
        atr = self._atr(prices)
        base_sl = price - atr * self.atr_mult
        base_tp = price + atr * self.atr_mult

        factor = 1.0
        vol = features.get("volatility")
        if vol is not None:
            if vol > self.vol_threshold:
                factor *= 1.5
            elif vol < self.vol_threshold / 2:
                factor *= 0.8

        regime = features.get("regime")
        if regime is not None and self._last_regime is not None and regime != self._last_regime:
            # regime change -> widen stops slightly
            factor *= 1.2
        self._last_regime = regime if regime is not None else self._last_regime

        adaptive_sl = price - atr * self.atr_mult * factor
        adaptive_tp = price + atr * self.atr_mult * factor
        return {
            "base_tp": base_tp,
            "base_sl": base_sl,
            "adaptive_tp": adaptive_tp,
            "adaptive_sl": adaptive_sl,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def update_trade(self, order_id: int, prices: pd.Series, features: Dict[str, Any]) -> Dict[str, float]:
        """Recompute thresholds and persist them for ``order_id``.

        The adaptive levels are pushed to the ``execution`` client via its
        ``update_order`` method.  Both baseline and adaptive levels are persisted
        through ``trade_log`` so that later analysis can compare the strategy's
        default exits with those actually used.
        """

        levels = self.compute_levels(prices, features)
        if hasattr(self.execution, "update_order"):
            self.execution.update_order(
                order_id,
                stop_loss=levels["adaptive_sl"],
                take_profit=levels["adaptive_tp"],
            )
        if hasattr(self.trade_log, "record_thresholds"):
            self.trade_log.record_thresholds(
                order_id,
                levels["base_tp"],
                levels["base_sl"],
                levels["adaptive_tp"],
                levels["adaptive_sl"],
            )
        return levels

