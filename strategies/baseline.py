"""Enhanced moving-average based trading strategy.

The original baseline strategy implemented a minimal moving-average
crossover algorithm intended for extremely resource constrained
environments.  This module expands upon that baseline by incorporating
additional technical indicators and risk management features commonly
found in production trading systems:

* Relative Strength Index (RSI) to filter overbought/oversold regimes.
* Bollinger Bands to avoid entries in stretched markets.
* Average True Range (ATR) driven risk management including optional
  position sizing adjustments in volatile environments.
* Trailing take-profit and stop-loss logic for open positions.
* Optional liquidity exhaustion filter derived from order book depth.

Despite the extra functionality the implementation remains
dependency-free and operates purely on streaming price data.

Example
-------
>>> BaselineStrategy(
...     short_window=5,
...    long_window=20,
...    atr_stop_long=3,
...    atr_stop_short=3,
...    scale_pos_by_atr=True,
... )
"""

from dataclasses import dataclass
from collections import deque
from math import isnan
from typing import Deque, Dict, Optional, Set

from indicators import atr as calc_atr, bollinger, rsi as calc_rsi, sma
from utils import load_config
from config_models import ConfigError
from analysis.kalman_filter import KalmanState, smooth_price


@dataclass
class IndicatorBundle:
    high: Optional[float] = None
    low: Optional[float] = None
    short_ma: Optional[float] = None
    long_ma: Optional[float] = None
    rsi: Optional[float] = None
    atr: Optional[float] = None
    boll_upper: Optional[float] = None
    boll_lower: Optional[float] = None
    obv: Optional[float] = None
    mfi: Optional[float] = None
    cvd: Optional[float] = None
    ram: Optional[float] = None
    hurst: Optional[float] = None
    htf_ma: Optional[float] = None
    htf_rsi: Optional[float] = None
    supertrend_break: Optional[int] = None
    kama_cross: Optional[int] = None
    kma_cross: Optional[int] = None
    vwap_cross: Optional[int] = None
    macd_cross: Optional[int] = None
    squeeze_break: Optional[int] = None
    div_rsi: Optional[int] = None
    div_macd: Optional[int] = None
    regime: Optional[int] = None
    vae_regime: Optional[int] = None
    microprice_delta: Optional[float] = None
    liq_exhaustion: Optional[int] = None


class BaselineStrategy:
    """Moving-average crossover strategy with basic risk management.

    Parameters
    ----------
    short_window:
        Number of recent prices for the fast moving average.
    long_window:
        Number of recent prices for the slow moving average.
    rsi_window:
        Number of price changes for RSI calculation.
    atr_window:
        Number of periods for ATR calculation.
    atr_stop_long:
        ATR multiples used for initial long stop and profit targets.
    atr_stop_short:
        ATR multiples used for initial short stop and profit targets.
    trailing_stop_pct:
        Percentage distance for trailing stop once the trade moves in
        favour.
    trailing_take_profit_pct:
        Percentage distance from the peak at which profits are locked
        in.  Activated only after the take-profit threshold is reached.
    session_position_limits:
        Optional mapping of session name to max absolute position size.
    default_position_limit:
        Default position limit when a session is not specified.
    scale_pos_by_atr:
        When ``True`` position sizes are scaled inversely with the
        latest ATR value.
    long_regimes:
        Optional set of regime ids permitting long positions. ``None``
        allows longs in any regime.
    short_regimes:
        Optional set of regime ids permitting short positions. ``None``
        allows shorts in any regime.
    long_vae_regimes:
        Optional set of VAE regime ids permitting long positions.
    short_vae_regimes:
        Optional set of VAE regime ids permitting short positions.
    ram_long_threshold:
        Minimum risk-adjusted momentum required to allow a long entry.
    ram_short_threshold:
        Maximum risk-adjusted momentum permitted for a short entry.
    hurst_trend_min:
        Minimum Hurst exponent required to allow a long entry.
    hurst_mean_reversion_max:
        Maximum Hurst exponent permitting a short entry.
    """

    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        rsi_window: int = 14,
        atr_window: int = 14,
        atr_stop_long: float = 3.0,
        atr_stop_short: float = 3.0,
        trailing_stop_pct: float = 0.01,
        trailing_take_profit_pct: float = 0.02,
        session_position_limits: Optional[Dict[str, int]] = None,
        default_position_limit: Optional[int] = None,
        scale_pos_by_atr: bool = False,
        long_regimes: Optional[Set[int]] = None,
        short_regimes: Optional[Set[int]] = None,
        long_vae_regimes: Optional[Set[int]] = None,
        short_vae_regimes: Optional[Set[int]] = None,
        ram_long_threshold: float = 0.0,
        ram_short_threshold: float = 0.0,
        hurst_trend_min: float = 0.5,
        hurst_mean_reversion_max: float = 0.5,
        use_kalman_smoothing: bool | None = None,
    ) -> None:
        if short_window >= long_window:
            raise ValueError("short_window must be < long_window")
        if not 0 < rsi_window:
            raise ValueError("rsi_window must be positive")
        self.short_window = short_window
        self.long_window = long_window
        self.rsi_window = rsi_window
        self.atr_window = atr_window
        self.atr_stop_long = float(atr_stop_long)
        self.atr_stop_short = float(atr_stop_short)
        self.trailing_stop_pct = float(trailing_stop_pct)
        self.trailing_take_profit_pct = float(trailing_take_profit_pct)
        self.scale_pos_by_atr = scale_pos_by_atr
        self.long_regimes = set(long_regimes) if long_regimes is not None else None
        self.short_regimes = set(short_regimes) if short_regimes is not None else None
        self.long_vae_regimes = (
            set(long_vae_regimes) if long_vae_regimes is not None else None
        )
        self.short_vae_regimes = (
            set(short_vae_regimes) if short_vae_regimes is not None else None
        )
        self.ram_long_threshold = float(ram_long_threshold)
        self.ram_short_threshold = float(ram_short_threshold)
        self.hurst_trend_min = float(hurst_trend_min)
        self.hurst_mean_reversion_max = float(hurst_mean_reversion_max)

        self._short: Deque[float] = deque(maxlen=short_window)
        self._long: Deque[float] = deque(maxlen=max(long_window, rsi_window + 1))
        self._prev_short = 0.0
        self._prev_long = 0.0
        self._prev_obv: Optional[float] = None
        self._prev_cvd: Optional[float] = None

        # ATR state
        self._highs: Deque[float] = deque(maxlen=atr_window + 1)
        self._lows: Deque[float] = deque(maxlen=atr_window + 1)
        self._closes: Deque[float] = deque(maxlen=atr_window + 1)
        self.latest_atr: Optional[float] = None
        self.entry_atr: Optional[float] = None

        # Position management
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price: Optional[float] = None
        self.peak_price: Optional[float] = None
        self.trough_price: Optional[float] = None
        self.take_profit_armed = False

        if session_position_limits is None or default_position_limit is None:
            try:
                cfg = load_config().strategy
                if session_position_limits is None:
                    session_position_limits = cfg.session_position_limits
                if default_position_limit is None:
                    default_position_limit = cfg.default_position_limit
            except Exception:
                if session_position_limits is None:
                    session_position_limits = {}
                if default_position_limit is None:
                    default_position_limit = 1

        self.session_position_limits = session_position_limits or {}
        self.default_position_limit = default_position_limit
        self.current_position_limit = self.default_position_limit

        if use_kalman_smoothing is None:
            try:
                use_kalman_smoothing = load_config().get("use_kalman_smoothing", False)
            except Exception:
                use_kalman_smoothing = False
        self.use_kalman_smoothing = bool(use_kalman_smoothing)
        self._kf_state: Optional[KalmanState] = None

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Session helpers
    # ------------------------------------------------------------------
    def set_session(self, session: Optional[str]) -> None:
        """Update position limits based on the active session."""

        if session is None:
            self.current_position_limit = self.default_position_limit
        else:
            self.current_position_limit = self.session_position_limits.get(
                session, self.default_position_limit
            )

    # ------------------------------------------------------------------
    # Main update routine
    # ------------------------------------------------------------------
    def update(
        self,
        price: float,
        indicators: Optional[IndicatorBundle] = None,
        session: Optional[str] = None,
    ) -> int:
        """Process a new bar and return a trading signal."""

        if indicators is None:
            indicators = IndicatorBundle()
        if session is not None:
            self.set_session(session)

        raw_price = price
        if self.use_kalman_smoothing:
            price, self._kf_state = smooth_price(price, self._kf_state)

        signal = self._compute_signal(price, indicators)
        signal = self._apply_filters(signal, indicators, price)
        return self._manage_position(
            raw_price, signal, indicators.regime, indicators.vae_regime
        )

    # ------------------------------------------------------------------
    # Signal computation helpers
    # ------------------------------------------------------------------
    def _compute_signal(self, price: float, ind: IndicatorBundle) -> int:
        high = ind.high if ind.high is not None else price
        low = ind.low if ind.low is not None else price

        self._short.append(price)
        self._long.append(price)
        self._highs.append(high)
        self._lows.append(low)
        self._closes.append(price)

        if ind.atr is not None:
            self.latest_atr = ind.atr
        elif len(self._closes) >= self.atr_window + 1:
            self.latest_atr = calc_atr(
                self._highs, self._lows, self._closes, self.atr_window
            )

        if len(self._long) < self.long_window or self.latest_atr is None:
            return 0

        short_ma_val = (
            ind.short_ma
            if ind.short_ma is not None
            else sma(self._short, self.short_window)
        )
        if (
            ind.long_ma is not None
            and ind.boll_upper is not None
            and ind.boll_lower is not None
        ):
            long_ma_val = ind.long_ma
            upper_band = ind.boll_upper
            lower_band = ind.boll_lower
        else:
            lm, ub, lb = bollinger(self._long, self.long_window)
            long_ma_val = ind.long_ma if ind.long_ma is not None else lm
            upper_band = ind.boll_upper if ind.boll_upper is not None else ub
            lower_band = ind.boll_lower if ind.boll_lower is not None else lb

        rsi_val = (
            ind.rsi if ind.rsi is not None else calc_rsi(self._long, self.rsi_window)
        )
        if isinstance(rsi_val, float) and isnan(rsi_val):
            rsi_val = 50.0

        raw_signal = 0
        if (
            short_ma_val > long_ma_val
            and self._prev_short <= self._prev_long
            and rsi_val < 70
            and price < upper_band
        ):
            raw_signal = 1
        elif (
            short_ma_val < long_ma_val
            and self._prev_short >= self._prev_long
            and rsi_val > 30
            and price > lower_band
        ):
            raw_signal = -1

        self._prev_short, self._prev_long = short_ma_val, long_ma_val
        return raw_signal

    def _apply_filters(
        self, raw_signal: int, ind: IndicatorBundle, price: float
    ) -> int:
        signal = raw_signal

        if signal == 1:
            if (ind.htf_ma is not None and price <= ind.htf_ma) or (
                ind.htf_rsi is not None and ind.htf_rsi <= 50
            ):
                signal = 0
        elif signal == -1:
            if (ind.htf_ma is not None and price >= ind.htf_ma) or (
                ind.htf_rsi is not None and ind.htf_rsi >= 50
            ):
                signal = 0

        if signal != 0 and ind.obv is not None and ind.mfi is not None:
            if signal == 1 and not (
                ind.mfi > 50 and (self._prev_obv is None or ind.obv > self._prev_obv)
            ):
                signal = 0
            elif signal == -1 and not (
                ind.mfi < 50 and (self._prev_obv is None or ind.obv < self._prev_obv)
            ):
                signal = 0
        self._prev_obv = ind.obv if ind.obv is not None else self._prev_obv

        if signal != 0 and ind.cvd is not None:
            if signal == 1 and not (
                ind.cvd > 0 and (self._prev_cvd is None or ind.cvd > self._prev_cvd)
            ):
                signal = 0
            elif signal == -1 and not (
                ind.cvd < 0 and (self._prev_cvd is None or ind.cvd < self._prev_cvd)
            ):
                signal = 0
        self._prev_cvd = ind.cvd if ind.cvd is not None else self._prev_cvd

        if signal != 0 and ind.microprice_delta is not None:
            if signal == 1 and ind.microprice_delta <= 0:
                signal = 0
            elif signal == -1 and ind.microprice_delta >= 0:
                signal = 0

        if signal != 0 and ind.liq_exhaustion is not None and ind.liq_exhaustion != signal:
            signal = 0

        if signal != 0 and ind.ram is not None:
            if signal == 1 and ind.ram < self.ram_long_threshold:
                signal = 0
            elif signal == -1 and ind.ram > self.ram_short_threshold:
                signal = 0

        if signal != 0 and ind.hurst is not None:
            if signal == 1 and ind.hurst < self.hurst_trend_min:
                signal = 0
            elif signal == -1 and ind.hurst > self.hurst_mean_reversion_max:
                signal = 0

        if signal != 0 and ind.vwap_cross is not None and ind.vwap_cross != signal:
            signal = 0
        if (
            signal != 0
            and ind.supertrend_break is not None
            and ind.supertrend_break != signal
        ):
            signal = 0
        if signal != 0 and ind.kama_cross is not None and ind.kama_cross != signal:
            signal = 0
        if signal != 0 and ind.kma_cross is not None and ind.kma_cross != signal:
            signal = 0
        if signal != 0 and ind.macd_cross is not None and ind.macd_cross != signal:
            signal = 0
        if (
            signal != 0
            and ind.squeeze_break is not None
            and ind.squeeze_break != signal
        ):
            signal = 0
        if signal != 0 and ind.div_rsi is not None and ind.div_rsi != signal:
            signal = 0
        if signal != 0 and ind.div_macd is not None and ind.div_macd != signal:
            signal = 0

        return signal

    # ------------------------------------------------------------------
    # Position management helpers
    # ------------------------------------------------------------------
    def _open_long(self, price: float) -> None:
        self.position = 1
        self.entry_price = price
        self.entry_atr = self.latest_atr
        self.peak_price = price
        self.trough_price = None
        self.take_profit_armed = False

    def _open_short(self, price: float) -> None:
        self.position = -1
        self.entry_price = price
        self.entry_atr = self.latest_atr
        self.trough_price = price
        self.peak_price = None
        self.take_profit_armed = False

    def _handle_open_position(self, price: float) -> int:
        if self.position == 1:
            return self._manage_long(price)
        if self.position == -1:
            return self._manage_short(price)
        return 0

    def _manage_position(
        self,
        price: float,
        signal: int,
        regime: Optional[int],
        vae_regime: Optional[int],
    ) -> int:
        # Exit immediately if current regime disallows the held position
        if regime is not None:
            if (
                self.position == 1
                and self.long_regimes is not None
                and regime not in self.long_regimes
            ):
                self.position = 0
                return -1
            if (
                self.position == -1
                and self.short_regimes is not None
                and regime not in self.short_regimes
            ):
                self.position = 0
                return 1
        if vae_regime is not None:
            if (
                self.position == 1
                and self.long_vae_regimes is not None
                and vae_regime not in self.long_vae_regimes
            ):
                self.position = 0
                return -1
            if (
                self.position == -1
                and self.short_vae_regimes is not None
                and vae_regime not in self.short_vae_regimes
            ):
                self.position = 0
                return 1

        # Manage existing position before considering new entries
        exit_signal = self._handle_open_position(price)
        if exit_signal:
            return exit_signal

        # No open position - consider new entries
        if self.position == 0 and signal != 0:
            if regime is not None:
                if (
                    signal == 1
                    and self.long_regimes is not None
                    and regime not in self.long_regimes
                ):
                    return 0
                if (
                    signal == -1
                    and self.short_regimes is not None
                    and regime not in self.short_regimes
                ):
                    return 0
            if vae_regime is not None:
                if (
                    signal == 1
                    and self.long_vae_regimes is not None
                    and vae_regime not in self.long_vae_regimes
                ):
                    return 0
                if (
                    signal == -1
                    and self.short_vae_regimes is not None
                    and vae_regime not in self.short_vae_regimes
                ):
                    return 0
            limit = self.current_position_limit
            if self.scale_pos_by_atr and self.latest_atr:
                limit = max(1, int(limit / self.latest_atr))
            signal = max(min(signal, limit), -limit)
            if signal == 1:
                self._open_long(price)
            else:
                self._open_short(price)
            return signal

        return 0

    def _manage_long(self, price: float) -> int:
        assert self.entry_price is not None and self.entry_atr is not None
        self.peak_price = max(self.peak_price or price, price)
        stop_loss = self.entry_price - self.entry_atr * self.atr_stop_long
        stop_loss = max(stop_loss, self.peak_price * (1 - self.trailing_stop_pct))

        if price <= stop_loss:
            self.position = 0
            return -1

        take_profit_level = self.entry_price + self.entry_atr * self.atr_stop_long
        if not self.take_profit_armed and price >= take_profit_level:
            self.take_profit_armed = True
            self.peak_price = price

        if self.take_profit_armed:
            self.peak_price = max(self.peak_price, price)
            tp_trigger = self.peak_price * (1 - self.trailing_take_profit_pct)
            if price <= tp_trigger:
                self.position = 0
                return -1
        return 0

    def _manage_short(self, price: float) -> int:
        assert self.entry_price is not None and self.entry_atr is not None
        self.trough_price = min(self.trough_price or price, price)
        stop_loss = self.entry_price + self.entry_atr * self.atr_stop_short
        stop_loss = min(stop_loss, self.trough_price * (1 + self.trailing_stop_pct))

        if price >= stop_loss:
            self.position = 0
            return 1

        take_profit_level = self.entry_price - self.entry_atr * self.atr_stop_short
        if not self.take_profit_armed and price <= take_profit_level:
            self.take_profit_armed = True
            self.trough_price = price

        if self.take_profit_armed:
            self.trough_price = min(self.trough_price, price)
            tp_trigger = self.trough_price * (1 + self.trailing_take_profit_pct)
            if price >= tp_trigger:
                self.position = 0
                return 1
        return 0


def run_backtest(
    cfg: dict,
    *,
    latency_ms: int = 0,
    slippage_model=None,
):
    """Run a backtest of the baseline strategy with execution settings."""

    from backtest import run_backtest as _run_backtest

    return _run_backtest(
        cfg,
        latency_ms=latency_ms,
        slippage_model=slippage_model,
    )


__all__ = ["BaselineStrategy", "IndicatorBundle", "run_backtest"]
