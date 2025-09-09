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

from collections import deque
from math import sqrt
from typing import Deque, Dict, Optional


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
        default_position_limit: int = 1,
        scale_pos_by_atr: bool = False,
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

        self._short: Deque[float] = deque(maxlen=short_window)
        self._long: Deque[float] = deque(maxlen=long_window)
        self._price_changes: Deque[float] = deque(maxlen=rsi_window)
        self._prev_price: Optional[float] = None
        self._prev_short = 0.0
        self._prev_long = 0.0
        self._prev_obv: Optional[float] = None

        # ATR state
        self._highs: Deque[float] = deque(maxlen=atr_window)
        self._lows: Deque[float] = deque(maxlen=atr_window)
        self._closes: Deque[float] = deque(maxlen=atr_window)
        self._trs: Deque[float] = deque(maxlen=atr_window)
        self.latest_atr: Optional[float] = None
        self.entry_atr: Optional[float] = None

        # Position management
        self.position = 0  # -1 short, 0 flat, 1 long
        self.entry_price: Optional[float] = None
        self.peak_price: Optional[float] = None
        self.trough_price: Optional[float] = None
        self.take_profit_armed = False

        self.session_position_limits = session_position_limits or {}
        self.default_position_limit = default_position_limit
        self.current_position_limit = default_position_limit

    # ------------------------------------------------------------------
    # Indicator helpers
    # ------------------------------------------------------------------
    def _compute_rsi(self) -> float:
        if len(self._price_changes) < self.rsi_window:
            return 50.0  # Neutral when insufficient data
        gains = [c for c in self._price_changes if c > 0]
        losses = [-c for c in self._price_changes if c < 0]
        avg_gain = sum(gains) / self.rsi_window
        avg_loss = sum(losses) / self.rsi_window
        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100 - 100 / (1 + rs)

    def _compute_bollinger(self, mean: float) -> tuple[float, float]:
        var = sum((p - mean) ** 2 for p in self._long) / self.long_window
        std = sqrt(var)
        upper = mean + 2 * std
        lower = mean - 2 * std
        return upper, lower

    def _true_range(self, high: float, low: float, prev_close: Optional[float]) -> float:
        if prev_close is None:
            return high - low
        return max(high - low, abs(high - prev_close), abs(low - prev_close))

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
        high: Optional[float] = None,
        low: Optional[float] = None,
        atr: Optional[float] = None,
        session: Optional[str] = None,
        obv: Optional[float] = None,
        mfi: Optional[float] = None,
    ) -> int:
        """Process a new bar and return a trading signal.

        Parameters
        ----------
        price:
            Closing price of the bar.
        high, low:
            Optional high and low for ATR calculation.  When omitted they
            default to ``price``.
        atr:
            Optional externally supplied ATR value.
        session:
            Optional session name used for position limits.
        obv, mfi:
            Optional volume indicators used for trend confirmation. When
            provided the strategy only enters long when both indicators
            show bullish pressure (``obv`` rising and ``mfi`` > 50) and
            short when they indicate bearish pressure.

        Returns
        -------
        int
            1 for buy, -1 for sell, 0 for hold.
        """

        if session is not None:
            self.set_session(session)

        if high is None:
            high = price
        if low is None:
            low = price

        if self._prev_price is not None:
            self._price_changes.append(price - self._prev_price)
        self._prev_price = price

        self._short.append(price)
        self._long.append(price)
        self._highs.append(high)
        self._lows.append(low)

        prev_close = self._closes[-1] if self._closes else None
        if atr is not None:
            self.latest_atr = atr
        else:
            tr = self._true_range(high, low, prev_close)
            self._trs.append(tr)
            if len(self._trs) == self.atr_window:
                self.latest_atr = sum(self._trs) / self.atr_window
        self._closes.append(price)

        if len(self._long) < self.long_window or self.latest_atr is None:
            # Not enough data yet
            return 0

        short_ma = sum(self._short) / self.short_window
        long_ma = sum(self._long) / self.long_window
        rsi = self._compute_rsi()
        upper_band, lower_band = self._compute_bollinger(long_ma)

        raw_signal = 0
        if (
            short_ma > long_ma
            and self._prev_short <= self._prev_long
            and rsi < 70
            and price < upper_band
        ):
            raw_signal = 1
        elif (
            short_ma < long_ma
            and self._prev_short >= self._prev_long
            and rsi > 30
            and price > lower_band
        ):
            raw_signal = -1

        # Volume confirmation using OBV and MFI
        if raw_signal != 0 and obv is not None and mfi is not None:
            if raw_signal == 1 and not (mfi > 50 and (self._prev_obv is None or obv > self._prev_obv)):
                raw_signal = 0
            elif raw_signal == -1 and not (mfi < 50 and (self._prev_obv is None or obv < self._prev_obv)):
                raw_signal = 0
        self._prev_obv = obv if obv is not None else self._prev_obv

        self._prev_short, self._prev_long = short_ma, long_ma

        # Manage existing position before considering new entries
        exit_signal = self._manage_position(price)
        if exit_signal:
            return exit_signal

        # No open position - consider new entries
        if self.position == 0 and raw_signal != 0:
            limit = self.current_position_limit
            if self.scale_pos_by_atr and self.latest_atr:
                limit = max(1, int(limit / self.latest_atr))
            raw_signal = max(min(raw_signal, limit), -limit)
            if raw_signal == 1:
                self._open_long(price)
            else:
                self._open_short(price)
            return raw_signal

        return 0

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

    def _manage_position(self, price: float) -> int:
        if self.position == 1:
            return self._manage_long(price)
        if self.position == -1:
            return self._manage_short(price)
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
