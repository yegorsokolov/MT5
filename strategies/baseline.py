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

from collections import deque
from dataclasses import dataclass, fields
from math import isnan
from typing import Deque, Dict, Optional, Sequence, Set, Union

import numpy as np
import pandas as pd

from indicators.common import atr as calc_atr, bollinger, rsi as calc_rsi, sma
from utils import load_config
from config_models import ConfigError
from analysis.kalman_filter import KalmanState, smooth_price


ScalarLike = Union[int, float, np.integer, np.floating]
VectorLike = Union[Sequence[float], np.ndarray, pd.Series]
IndicatorValue = Union[ScalarLike, VectorLike]


@dataclass
class IndicatorBundle:
    high: Optional[IndicatorValue] = None
    low: Optional[IndicatorValue] = None
    short_ma: Optional[IndicatorValue] = None
    long_ma: Optional[IndicatorValue] = None
    rsi: Optional[IndicatorValue] = None
    atr_val: Optional[IndicatorValue] = None
    boll_upper: Optional[IndicatorValue] = None
    boll_lower: Optional[IndicatorValue] = None
    obv: Optional[IndicatorValue] = None
    mfi: Optional[IndicatorValue] = None
    cvd: Optional[IndicatorValue] = None
    ram: Optional[IndicatorValue] = None
    hurst: Optional[IndicatorValue] = None
    htf_ma: Optional[IndicatorValue] = None
    htf_rsi: Optional[IndicatorValue] = None
    supertrend_break: Optional[IndicatorValue] = None
    kama_cross: Optional[IndicatorValue] = None
    kma_cross: Optional[IndicatorValue] = None
    vwap_cross: Optional[IndicatorValue] = None
    macd_cross: Optional[IndicatorValue] = None
    squeeze_break: Optional[IndicatorValue] = None
    div_rsi: Optional[IndicatorValue] = None
    div_macd: Optional[IndicatorValue] = None
    regime: Optional[IndicatorValue] = None
    vae_regime: Optional[IndicatorValue] = None
    microprice_delta: Optional[IndicatorValue] = None
    liq_exhaustion: Optional[IndicatorValue] = None
    # Optional mapping of arbitrary indicator names to values.  This allows
    # downstream users to supply dynamically evolved indicators without
    # requiring explicit fields for each one.
    evolved: Optional[Dict[str, IndicatorValue]] = None

    def vector_length(self) -> Optional[int]:
        """Return the vector length when any field contains array data."""

        length: Optional[int] = None
        for field in fields(self):
            if field.name == "evolved":
                value = getattr(self, field.name)
                if value:
                    raise NotImplementedError("Vectorized evolved indicators are not supported")
                continue
            value = getattr(self, field.name)
            if isinstance(value, pd.Series):
                curr_len = len(value)
            elif isinstance(value, np.ndarray):
                curr_len = int(value.size)
            elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                curr_len = len(value)
            else:
                continue
            if length is None:
                length = curr_len
            elif curr_len != length:
                raise ValueError("Indicator arrays must share the same length")
        return length


@dataclass
class RiskProfile:
    """User-defined risk preferences used to scale signals."""

    tolerance: float = 1.0
    leverage_cap: float = 1.0
    drawdown_limit: float = 0.0


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
        risk_profile: Optional[RiskProfile] = None,
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
        self.risk_profile = risk_profile or RiskProfile()
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
    # Batch evaluation helpers
    # ------------------------------------------------------------------
    def batch_compute(
        self,
        price: Sequence[float] | pd.Series,
        indicators: IndicatorBundle,
        *,
        session: Optional[str] = None,
        cross_confirm: Optional[Dict[str, Sequence[float] | pd.Series]] = None,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Compute signals and stops over a vector of prices.

        Parameters
        ----------
        price:
            Price series aligned with the indicator arrays.
        indicators:
            Indicator bundle.  When vectorized arrays are supplied a fully
            vectorized execution path is used.  Otherwise the strategy falls
            back to sequential evaluation mirroring :meth:`update`.
        session:
            Optional session identifier; unsupported in vectorized mode.
        cross_confirm:
            Optional confirmation signals; unsupported in vectorized mode.
        """

        price_series = self._ensure_price_series(price)
        try:
            return self._batch_vectorized(price_series, indicators, session=session, cross_confirm=cross_confirm)
        except NotImplementedError:
            return self._batch_sequential(price_series, indicators, session=session, cross_confirm=cross_confirm)

    def _batch_vectorized(
        self,
        price_series: pd.Series,
        indicators: IndicatorBundle,
        *,
        session: Optional[str],
        cross_confirm: Optional[Dict[str, Sequence[float] | pd.Series]],
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        if session is not None or cross_confirm is not None:
            raise NotImplementedError("Vectorized batch processing does not support sessions or cross confirmations")
        if self.use_kalman_smoothing:
            raise NotImplementedError("Kalman smoothing is unavailable in vectorized mode")
        if self.scale_pos_by_atr:
            raise NotImplementedError("ATR position scaling is unavailable in vectorized mode")
        if self.session_position_limits:
            raise NotImplementedError("Session-specific position limits are not supported in vectorized mode")
        if self.default_position_limit != 1:
            raise NotImplementedError("Vectorized mode currently assumes a unit position limit")
        if indicators.evolved:
            raise NotImplementedError("Vectorized mode does not yet support evolved indicators")

        index = price_series.index
        series_map = self._prepare_indicator_series(price_series, indicators)

        short_ma = series_map["short_ma"]
        long_ma = series_map["long_ma"]
        boll_upper = series_map["boll_upper"]
        boll_lower = series_map["boll_lower"]
        rsi_series = series_map["rsi"].fillna(50)
        atr_series = series_map["atr_val"]

        prev_short = short_ma.shift(1)
        prev_long = long_ma.shift(1)
        prev_short = prev_short.where(prev_long.notna(), 0.0).fillna(0.0)
        prev_long = prev_long.fillna(0.0)
        price_lt_upper = price_series < boll_upper
        price_gt_lower = price_series > boll_lower

        long_entries = (
            (short_ma > long_ma)
            & (prev_short <= prev_long)
            & (rsi_series < 70)
            & price_lt_upper
        )
        short_entries = (
            (short_ma < long_ma)
            & (prev_short >= prev_long)
            & (rsi_series > 30)
            & price_gt_lower
        )

        raw_signal = pd.Series(
            np.where(long_entries, 1, np.where(short_entries, -1, 0)),
            index=index,
            dtype=int,
        )
        valid = (~short_ma.isna()) & (~long_ma.isna()) & (~atr_series.isna())
        signal = raw_signal.where(valid, 0)

        # High timeframe filters
        htf_ma = series_map["htf_ma"]
        htf_rsi = series_map["htf_rsi"]
        mask_long = signal == 1
        invalid_long = (htf_ma.notna() & (price_series <= htf_ma)) | (
            htf_rsi.notna() & (htf_rsi <= 50)
        )
        signal = signal.where(~(mask_long & invalid_long), 0)
        mask_short = signal == -1
        invalid_short = (htf_ma.notna() & (price_series >= htf_ma)) | (
            htf_rsi.notna() & (htf_rsi >= 50)
        )
        signal = signal.where(~(mask_short & invalid_short), 0)

        # OBV/MFI confirmation
        obv = series_map["obv"]
        mfi = series_map["mfi"]
        if obv.notna().any() and mfi.notna().any():
            prev_obv = obv.where(obv.notna()).ffill().shift(1)
            mask = (signal == 1) & obv.notna() & mfi.notna()
            valid_mask = (mfi > 50) & ((prev_obv.isna()) | (obv > prev_obv))
            signal = signal.where(~mask | valid_mask, 0)
            mask = (signal == -1) & obv.notna() & mfi.notna()
            valid_mask = (mfi < 50) & ((prev_obv.isna()) | (obv < prev_obv))
            signal = signal.where(~mask | valid_mask, 0)

        # CVD confirmation
        cvd = series_map["cvd"]
        if cvd.notna().any():
            prev_cvd = cvd.where(cvd.notna()).ffill().shift(1)
            mask = (signal == 1) & cvd.notna()
            valid_mask = (cvd > 0) & ((prev_cvd.isna()) | (cvd > prev_cvd))
            signal = signal.where(~mask | valid_mask, 0)
            mask = (signal == -1) & cvd.notna()
            valid_mask = (cvd < 0) & ((prev_cvd.isna()) | (cvd < prev_cvd))
            signal = signal.where(~mask | valid_mask, 0)

        # Microprice and liquidity filters
        microprice = series_map["microprice_delta"]
        if microprice.notna().any():
            mask = (signal == 1) & microprice.notna() & (microprice <= 0)
            signal = signal.where(~mask, 0)
            mask = (signal == -1) & microprice.notna() & (microprice >= 0)
            signal = signal.where(~mask, 0)

        liq_exhaustion = series_map["liq_exhaustion"]
        if liq_exhaustion.notna().any():
            mismatch = (signal != 0) & liq_exhaustion.notna() & (liq_exhaustion != signal)
            signal = signal.where(~mismatch, 0)

        # RAM/Hurst gating
        ram = series_map["ram"]
        if ram.notna().any():
            if self.ram_long_threshold != 0:
                mask = (signal == 1) & ram.notna() & (ram < self.ram_long_threshold)
                signal = signal.where(~mask, 0)
            if self.ram_short_threshold != 0:
                mask = (signal == -1) & ram.notna() & (ram > self.ram_short_threshold)
                signal = signal.where(~mask, 0)

        hurst = series_map["hurst"]
        if hurst.notna().any():
            mask = (signal == 1) & hurst.notna() & (hurst < self.hurst_trend_min)
            signal = signal.where(~mask, 0)
            mask = (signal == -1) & hurst.notna() & (hurst > self.hurst_mean_reversion_max)
            signal = signal.where(~mask, 0)

        # Misc cross filters
        for name in [
            "vwap_cross",
            "supertrend_break",
            "kama_cross",
            "kma_cross",
            "macd_cross",
            "squeeze_break",
            "div_rsi",
            "div_macd",
        ]:
            series = series_map[name]
            if series.notna().any():
                mask = (signal != 0) & series.notna() & (series != signal)
                signal = signal.where(~mask, 0)

        price_arr = price_series.to_numpy()
        atr_arr = atr_series.to_numpy()
        regime_series = series_map["regime"]
        vae_regime_series = series_map["vae_regime"]

        result_signal = np.zeros(len(price_series), dtype=float)
        long_stop = np.full(len(price_series), np.nan, dtype=float)
        short_stop = np.full(len(price_series), np.nan, dtype=float)

        idx = 0
        drawdown_limit = float(self.risk_profile.drawdown_limit)
        while idx < len(price_series):
            sig_val = int(signal.iat[idx])
            if sig_val == 1:
                regime_val = regime_series.iat[idx]
                if (
                    self.long_regimes is not None
                    and pd.notna(regime_val)
                    and int(regime_val) not in self.long_regimes
                ):
                    idx += 1
                    continue
                vae_val = vae_regime_series.iat[idx]
                if (
                    self.long_vae_regimes is not None
                    and pd.notna(vae_val)
                    and int(vae_val) not in self.long_vae_regimes
                ):
                    idx += 1
                    continue
                entry_atr = atr_arr[idx]
                if not np.isfinite(entry_atr):
                    idx += 1
                    continue

                entry_idx = idx
                price_segment = price_arr[entry_idx:]
                peak = np.maximum.accumulate(price_segment)
                stop_initial = price_arr[entry_idx] - entry_atr * self.atr_stop_long
                if drawdown_limit > 0:
                    stop_initial = max(
                        stop_initial, price_arr[entry_idx] * (1 - drawdown_limit)
                    )
                trailing_component = peak * (1 - self.trailing_stop_pct)
                stop_series = np.maximum(stop_initial, trailing_component)
                take_profit_level = price_arr[entry_idx] + entry_atr * self.atr_stop_long
                tp_armed = peak >= take_profit_level
                tp_trigger = peak * (1 - self.trailing_take_profit_pct)
                exit_mask = (price_segment <= stop_series) | (
                    tp_armed & (price_segment <= tp_trigger)
                )
                exit_indices = np.flatnonzero(exit_mask)
                if exit_indices.size:
                    exit_rel = int(exit_indices[0])
                    exit_idx = entry_idx + exit_rel
                    active_len = exit_rel
                else:
                    exit_idx = len(price_arr)
                    active_len = len(price_segment)

                if active_len > 0:
                    long_stop[entry_idx : entry_idx + active_len] = stop_series[:active_len]
                result_signal[entry_idx] = 1.0
                if exit_idx < len(price_arr):
                    result_signal[exit_idx] = -1.0
                    idx = exit_idx + 1
                else:
                    idx = len(price_arr)
            elif sig_val == -1:
                regime_val = regime_series.iat[idx]
                if (
                    self.short_regimes is not None
                    and pd.notna(regime_val)
                    and int(regime_val) not in self.short_regimes
                ):
                    idx += 1
                    continue
                vae_val = vae_regime_series.iat[idx]
                if (
                    self.short_vae_regimes is not None
                    and pd.notna(vae_val)
                    and int(vae_val) not in self.short_vae_regimes
                ):
                    idx += 1
                    continue
                entry_atr = atr_arr[idx]
                if not np.isfinite(entry_atr):
                    idx += 1
                    continue

                entry_idx = idx
                price_segment = price_arr[entry_idx:]
                trough = np.minimum.accumulate(price_segment)
                stop_initial = price_arr[entry_idx] + entry_atr * self.atr_stop_short
                if drawdown_limit > 0:
                    stop_initial = min(
                        stop_initial, price_arr[entry_idx] * (1 + drawdown_limit)
                    )
                trailing_component = trough * (1 + self.trailing_stop_pct)
                stop_series = np.minimum(stop_initial, trailing_component)
                take_profit_level = price_arr[entry_idx] - entry_atr * self.atr_stop_short
                tp_armed = trough <= take_profit_level
                tp_trigger = trough * (1 + self.trailing_take_profit_pct)
                exit_mask = (price_segment >= stop_series) | (
                    tp_armed & (price_segment >= tp_trigger)
                )
                exit_indices = np.flatnonzero(exit_mask)
                if exit_indices.size:
                    exit_rel = int(exit_indices[0])
                    exit_idx = entry_idx + exit_rel
                    active_len = exit_rel
                else:
                    exit_idx = len(price_arr)
                    active_len = len(price_segment)

                if active_len > 0:
                    short_stop[entry_idx : entry_idx + active_len] = stop_series[:active_len]
                result_signal[entry_idx] = -1.0
                if exit_idx < len(price_arr):
                    result_signal[exit_idx] = 1.0
                    idx = exit_idx + 1
                else:
                    idx = len(price_arr)
            else:
                idx += 1

        rp = self.risk_profile
        if rp.tolerance != 1.0 or rp.leverage_cap != 1.0:
            result_signal *= rp.tolerance
            if rp.leverage_cap > 0:
                np.clip(result_signal, -rp.leverage_cap, rp.leverage_cap, out=result_signal)

        return (
            pd.Series(result_signal, index=index, dtype=float),
            pd.Series(long_stop, index=index, dtype=float),
            pd.Series(short_stop, index=index, dtype=float),
        )

    def _batch_sequential(
        self,
        price_series: pd.Series,
        indicators: IndicatorBundle,
        *,
        session: Optional[str],
        cross_confirm: Optional[Dict[str, Sequence[float] | pd.Series]],
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        strat = self._clone_for_batch()
        if session is not None:
            strat.set_session(session)
        if cross_confirm is not None:
            raise NotImplementedError(
                "Sequential fallback does not support cross confirmation data"
            )

        index = price_series.index
        series_map = self._prepare_indicator_series(price_series, indicators)
        signals: list[float] = []
        long_stops: list[float] = []
        short_stops: list[float] = []

        for i, price_val in enumerate(price_series):
            bundle = IndicatorBundle(
                high=series_map["high"].iat[i],
                low=series_map["low"].iat[i],
                short_ma=series_map["short_ma"].iat[i],
                long_ma=series_map["long_ma"].iat[i],
                rsi=series_map["rsi"].iat[i],
                atr_val=series_map["atr_val"].iat[i],
                boll_upper=series_map["boll_upper"].iat[i],
                boll_lower=series_map["boll_lower"].iat[i],
                obv=series_map["obv"].iat[i],
                mfi=series_map["mfi"].iat[i],
                cvd=series_map["cvd"].iat[i],
                ram=series_map["ram"].iat[i],
                hurst=series_map["hurst"].iat[i],
                htf_ma=series_map["htf_ma"].iat[i],
                htf_rsi=series_map["htf_rsi"].iat[i],
                supertrend_break=series_map["supertrend_break"].iat[i],
                kama_cross=series_map["kama_cross"].iat[i],
                kma_cross=series_map["kma_cross"].iat[i],
                vwap_cross=series_map["vwap_cross"].iat[i],
                macd_cross=series_map["macd_cross"].iat[i],
                squeeze_break=series_map["squeeze_break"].iat[i],
                div_rsi=series_map["div_rsi"].iat[i],
                div_macd=series_map["div_macd"].iat[i],
                regime=series_map["regime"].iat[i],
                vae_regime=series_map["vae_regime"].iat[i],
                microprice_delta=series_map["microprice_delta"].iat[i],
                liq_exhaustion=series_map["liq_exhaustion"].iat[i],
            )
            sig = strat.update(price_val, bundle)
            signals.append(sig)

            if strat.position == 1 and strat.entry_price is not None and strat.entry_atr is not None:
                peak = strat.peak_price if strat.peak_price is not None else price_val
                long_stop = max(
                    strat.entry_price - strat.entry_atr * strat.atr_stop_long,
                    peak * (1 - strat.trailing_stop_pct),
                )
            else:
                long_stop = float("nan")

            if strat.position == -1 and strat.entry_price is not None and strat.entry_atr is not None:
                trough = strat.trough_price if strat.trough_price is not None else price_val
                short_stop = min(
                    strat.entry_price + strat.entry_atr * strat.atr_stop_short,
                    trough * (1 + strat.trailing_stop_pct),
                )
            else:
                short_stop = float("nan")

            long_stops.append(long_stop)
            short_stops.append(short_stop)

        return (
            pd.Series(signals, index=index, dtype=float),
            pd.Series(long_stops, index=index, dtype=float),
            pd.Series(short_stops, index=index, dtype=float),
        )

    def _clone_for_batch(self) -> "BaselineStrategy":
        return BaselineStrategy(
            short_window=self.short_window,
            long_window=self.long_window,
            rsi_window=self.rsi_window,
            atr_window=self.atr_window,
            atr_stop_long=self.atr_stop_long,
            atr_stop_short=self.atr_stop_short,
            trailing_stop_pct=self.trailing_stop_pct,
            trailing_take_profit_pct=self.trailing_take_profit_pct,
            session_position_limits=self.session_position_limits,
            default_position_limit=self.default_position_limit,
            scale_pos_by_atr=self.scale_pos_by_atr,
            long_regimes=self.long_regimes,
            short_regimes=self.short_regimes,
            long_vae_regimes=self.long_vae_regimes,
            short_vae_regimes=self.short_vae_regimes,
            ram_long_threshold=self.ram_long_threshold,
            ram_short_threshold=self.ram_short_threshold,
            hurst_trend_min=self.hurst_trend_min,
            hurst_mean_reversion_max=self.hurst_mean_reversion_max,
            use_kalman_smoothing=self.use_kalman_smoothing,
            risk_profile=RiskProfile(
                tolerance=self.risk_profile.tolerance,
                leverage_cap=self.risk_profile.leverage_cap,
                drawdown_limit=self.risk_profile.drawdown_limit,
            ),
        )

    @staticmethod
    def _ensure_price_series(price: Sequence[float] | pd.Series) -> pd.Series:
        if isinstance(price, pd.Series):
            return price.astype(float)
        if isinstance(price, np.ndarray):
            return pd.Series(price.astype(float))
        return pd.Series(list(price), dtype=float)

    def _prepare_indicator_series(
        self,
        price_series: pd.Series,
        indicators: IndicatorBundle,
    ) -> Dict[str, pd.Series]:
        index = price_series.index
        nan_series = pd.Series(np.nan, index=index, dtype=float)

        def ensure_series(value: IndicatorValue | None, name: str, fill_value: float = np.nan) -> pd.Series:
            if isinstance(value, pd.Series):
                if not value.index.equals(index):
                    return value.reindex(index)
                return value.astype(float, copy=False)
            if value is None:
                return pd.Series(fill_value, index=index, dtype=float, name=name)
            if isinstance(value, np.ndarray):
                arr = value
                if arr.ndim == 0:
                    arr = np.full(len(index), float(arr))
                elif len(arr) != len(index):
                    raise ValueError(f"Indicator '{name}' length mismatch")
                return pd.Series(arr, index=index, dtype=float, name=name)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                seq = list(value)
                if len(seq) != len(index):
                    raise ValueError(f"Indicator '{name}' length mismatch")
                return pd.Series(seq, index=index, dtype=float, name=name)
            if isinstance(value, (int, float, np.integer, np.floating)):
                return pd.Series([value] * len(index), index=index, dtype=float, name=name)
            raise TypeError(f"Unsupported indicator type for '{name}'")

        def optional_series(value: IndicatorValue | None, name: str) -> pd.Series:
            if value is None:
                return nan_series
            return ensure_series(value, name)

        high = ensure_series(indicators.high, "high").fillna(price_series)
        low = ensure_series(indicators.low, "low").fillna(price_series)
        short_ma = ensure_series(indicators.short_ma, "short_ma")
        long_ma = ensure_series(indicators.long_ma, "long_ma")
        boll_upper = ensure_series(indicators.boll_upper, "boll_upper")
        boll_lower = ensure_series(indicators.boll_lower, "boll_lower")
        rsi_series = ensure_series(indicators.rsi, "rsi")
        atr_series = ensure_series(indicators.atr_val, "atr_val")

        computed_short = sma(price_series, self.short_window)
        short_ma = short_ma.combine_first(computed_short)
        ma, upper, lower = bollinger(price_series, self.long_window)
        long_ma = long_ma.combine_first(ma)
        boll_upper = boll_upper.combine_first(upper)
        boll_lower = boll_lower.combine_first(lower)
        computed_rsi = calc_rsi(price_series, self.rsi_window)
        rsi_series = rsi_series.combine_first(computed_rsi)
        computed_atr = calc_atr(high, low, price_series, self.atr_window)
        atr_series = atr_series.combine_first(computed_atr)

        series_map: Dict[str, pd.Series] = {
            "price": price_series.astype(float, copy=False),
            "high": high.astype(float, copy=False),
            "low": low.astype(float, copy=False),
            "short_ma": short_ma.astype(float, copy=False),
            "long_ma": long_ma.astype(float, copy=False),
            "boll_upper": boll_upper.astype(float, copy=False),
            "boll_lower": boll_lower.astype(float, copy=False),
            "rsi": rsi_series.astype(float, copy=False),
            "atr_val": atr_series.astype(float, copy=False),
            "obv": optional_series(indicators.obv, "obv"),
            "mfi": optional_series(indicators.mfi, "mfi"),
            "cvd": optional_series(indicators.cvd, "cvd"),
            "ram": optional_series(indicators.ram, "ram"),
            "hurst": optional_series(indicators.hurst, "hurst"),
            "htf_ma": optional_series(indicators.htf_ma, "htf_ma"),
            "htf_rsi": optional_series(indicators.htf_rsi, "htf_rsi"),
            "supertrend_break": optional_series(indicators.supertrend_break, "supertrend_break"),
            "kama_cross": optional_series(indicators.kama_cross, "kama_cross"),
            "kma_cross": optional_series(indicators.kma_cross, "kma_cross"),
            "vwap_cross": optional_series(indicators.vwap_cross, "vwap_cross"),
            "macd_cross": optional_series(indicators.macd_cross, "macd_cross"),
            "squeeze_break": optional_series(indicators.squeeze_break, "squeeze_break"),
            "div_rsi": optional_series(indicators.div_rsi, "div_rsi"),
            "div_macd": optional_series(indicators.div_macd, "div_macd"),
            "regime": optional_series(indicators.regime, "regime"),
            "vae_regime": optional_series(indicators.vae_regime, "vae_regime"),
            "microprice_delta": optional_series(indicators.microprice_delta, "microprice_delta"),
            "liq_exhaustion": optional_series(indicators.liq_exhaustion, "liq_exhaustion"),
        }

        return series_map

    # ------------------------------------------------------------------
    # Main update routine
    # ------------------------------------------------------------------
    def update(
        self,
        price: float,
        indicators: Optional[IndicatorBundle] = None,
        session: Optional[str] = None,
        cross_confirm: Optional[Dict[str, float]] = None,
    ) -> float:
        """Process a new bar and return a trading signal.

        Trades are only permitted when ``cross_confirm`` is supplied and all
        provided confirmation values are positive, indicating the traded asset
        remains sufficiently correlated with its peers.
        """

        if indicators is None:
            indicators = IndicatorBundle()
        if session is not None:
            self.set_session(session)

        raw_price = price
        if self.use_kalman_smoothing:
            price, self._kf_state = smooth_price(price, self._kf_state)

        signal = self._compute_signal(price, indicators)
        signal = self._apply_filters(signal, indicators, price, cross_confirm)
        sig = self._manage_position(
            raw_price, signal, indicators.regime, indicators.vae_regime
        )
        return self._apply_risk(sig)

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

        if ind.atr_val is not None:
            self.latest_atr = ind.atr_val
        elif len(self._closes) >= self.atr_window + 1:
            self.latest_atr = calc_atr(
                self._highs, self._lows, self._closes, self.atr_window
            )
            ind.atr_val = self.latest_atr

        if len(self._long) < self.long_window or self.latest_atr is None:
            return 0

        if ind.short_ma is not None:
            short_ma_val = ind.short_ma
        else:
            short_ma_val = sma(self._short, self.short_window)
            ind.short_ma = short_ma_val
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
            if ind.long_ma is None:
                ind.long_ma = lm
            if ind.boll_upper is None:
                ind.boll_upper = ub
            if ind.boll_lower is None:
                ind.boll_lower = lb
            long_ma_val = ind.long_ma
            upper_band = ind.boll_upper
            lower_band = ind.boll_lower

        if ind.rsi is not None:
            rsi_val = ind.rsi
        else:
            rsi_val = calc_rsi(self._long, self.rsi_window)
            ind.rsi = rsi_val
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
        self,
        raw_signal: int,
        ind: IndicatorBundle,
        price: float,
        cross_confirm: Optional[Dict[str, float]] = None,
    ) -> int:
        signal = raw_signal

        if signal != 0 and cross_confirm is not None:
            if any(v <= 0 for v in cross_confirm.values()):
                return 0

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

        if (
            signal != 0
            and ind.liq_exhaustion is not None
            and ind.liq_exhaustion != signal
        ):
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
        regime: Optional[int] = None,
        vae_regime: Optional[int] = None,
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
        if self.risk_profile.drawdown_limit > 0:
            stop_loss = max(
                stop_loss, self.entry_price * (1 - self.risk_profile.drawdown_limit)
            )
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
        if self.risk_profile.drawdown_limit > 0:
            stop_loss = min(
                stop_loss, self.entry_price * (1 + self.risk_profile.drawdown_limit)
            )
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

    def _apply_risk(self, signal: int) -> float:
        rp = self.risk_profile
        if rp.tolerance == 1.0 and rp.leverage_cap == 1.0:
            return float(signal)
        adj = signal * rp.tolerance
        if rp.leverage_cap > 0:
            adj = max(min(adj, rp.leverage_cap), -rp.leverage_cap)
        return adj


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


__all__ = ["BaselineStrategy", "IndicatorBundle", "RiskProfile", "run_backtest"]
