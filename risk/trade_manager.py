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
    survival_model: Any, optional
        Model providing ``predict_survival`` returning probability a trade remains profitable.
    survival_threshold: float, default ``0.2``
        Close trade when survival probability falls below this value.
    """

    execution: Any
    trade_log: Any
    atr_period: int = 14
    atr_mult: float = 3.0
    vol_threshold: float = 1.5
    survival_model: Any | None = None
    survival_threshold: float = 0.2

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
    def open_trade(self, order: Dict[str, Any], confirm_score: float) -> int | None:
        """Open a trade if ``confirm_score`` is positive.

        The order is forwarded to the ``execution`` client and persisted via
        ``trade_log`` when ``confirm_score`` is greater than zero.  The
        confirmation score is stored alongside the order for later analysis.
        """

        if confirm_score <= 0:
            return None
        order_id = None
        if hasattr(self.trade_log, "record_order"):
            order_id = int(self.trade_log.record_order(order))
            if hasattr(self.trade_log, "record_confirmation"):
                self.trade_log.record_confirmation(order_id, confirm_score)
        if hasattr(self.execution, "open_order"):
            self.execution.open_order(order)
        elif hasattr(self.execution, "open_trade"):
            self.execution.open_trade(order)
        return order_id

    def update_trade(
        self,
        order_id: int,
        prices: pd.Series,
        features: Dict[str, Any],
        confirm_score: float | None = None,
    ) -> Dict[str, float]:
        """Recompute thresholds and persist them for ``order_id``.

        The adaptive levels are pushed to the ``execution`` client via its
        ``update_order`` method.  Both baseline and adaptive levels are persisted
        through ``trade_log`` so that later analysis can compare the strategy's
        default exits with those actually used.
        """

        levels = self.compute_levels(prices, features)
        if confirm_score is not None and hasattr(self.trade_log, "record_confirmation"):
            self.trade_log.record_confirmation(order_id, confirm_score)
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
        survival_prob = None
        if self.survival_model is not None:
            feat = dict(features)
            feat["age"] = len(prices)
            if hasattr(self.survival_model, 'predict_survival'):
                survival_prob = float(self.survival_model.predict_survival(feat))
            else:
                survival_prob = float(self.survival_model.predict(feat))
            if hasattr(self.trade_log, 'record_survival'):
                self.trade_log.record_survival(order_id, survival_prob)
        force_exit = False
        if survival_prob is not None and survival_prob < self.survival_threshold:
            force_exit = True
        if confirm_score is not None and confirm_score < 0 and not force_exit:
            force_exit = True
        if force_exit:
            if hasattr(self.execution, 'close_order'):
                self.execution.close_order(order_id)
            elif hasattr(self.execution, 'close_trade'):
                self.execution.close_trade(order_id)
            elif hasattr(self.execution, 'close_position'):
                self.execution.close_position(order_id)
        result = dict(levels)
        if survival_prob is not None:
            result['survival_prob'] = survival_prob
        return result

