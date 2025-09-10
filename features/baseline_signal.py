"""Run BaselineStrategy over historical bars as a feature."""

from __future__ import annotations

import numpy as np
import pandas as pd

from strategies.baseline import BaselineStrategy, IndicatorBundle
from indicators import sma, rsi, atr, bollinger


def compute(
    df: pd.DataFrame,
    short_window: int = 5,
    long_window: int = 20,
    rsi_window: int = 14,
    atr_window: int = 14,
    atr_stop_long: float = 3.0,
    atr_stop_short: float = 3.0,
    trailing_stop_pct: float = 0.01,
    trailing_take_profit_pct: float = 0.02,
) -> pd.DataFrame:
    """Append baseline strategy signals and stops.

    Parameters
    ----------
    df:
        DataFrame containing ``Close`` prices and optionally ``High`` and ``Low``.
    short_window, long_window, rsi_window, atr_window:
        Indicator windows mirroring :class:`~strategies.baseline.BaselineStrategy`.
    atr_stop_long, atr_stop_short, trailing_stop_pct, trailing_take_profit_pct:
        Risk management parameters passed to :class:`BaselineStrategy`.

    Returns
    -------
    pandas.DataFrame
        Original dataframe with ``baseline_signal``, ``baseline_long_stop`` and
        ``baseline_short_stop`` columns.
    """

    df = df.copy()

    price = df.get("Close", df.get("mid"))
    if price is None:
        raise KeyError("DataFrame must contain 'Close' or 'mid' column")
    high = df.get("High", price)
    low = df.get("Low", price)

    # indicators, use precomputed columns when present
    short_ma = df.get("short_ma")
    if short_ma is None:
        short_ma = sma(price, short_window)
    long_ma = df.get("long_ma")
    boll_upper = df.get("boll_upper")
    boll_lower = df.get("boll_lower")
    if long_ma is None or boll_upper is None or boll_lower is None:
        ma, upper, lower = bollinger(price, long_window)
        if long_ma is None:
            long_ma = ma
        if boll_upper is None:
            boll_upper = upper
        if boll_lower is None:
            boll_lower = lower
    rsi_vals = df.get("rsi")
    if rsi_vals is None:
        rsi_vals = rsi(price, rsi_window)
    atr_vals = df.get("atr")
    if atr_vals is None:
        atr_vals = atr(high, low, price, atr_window)

    if "short_ma" not in df:
        df["short_ma"] = short_ma
    if "long_ma" not in df:
        df["long_ma"] = long_ma
    if "rsi" not in df:
        df["rsi"] = rsi_vals
    if "atr" not in df:
        df["atr"] = atr_vals
    if "boll_upper" not in df:
        df["boll_upper"] = boll_upper
    if "boll_lower" not in df:
        df["boll_lower"] = boll_lower

    strat = BaselineStrategy(
        short_window=short_window,
        long_window=long_window,
        rsi_window=rsi_window,
        atr_window=atr_window,
        atr_stop_long=atr_stop_long,
        atr_stop_short=atr_stop_short,
        trailing_stop_pct=trailing_stop_pct,
        trailing_take_profit_pct=trailing_take_profit_pct,
    )

    signals: list[int] = []
    long_stops: list[float] = []
    short_stops: list[float] = []

    for row in df.itertuples():
        price_val = getattr(row, "Close", getattr(row, "mid", np.nan))
        ind = IndicatorBundle(
            high=getattr(row, "High", price_val),
            low=getattr(row, "Low", price_val),
            short_ma=row.short_ma if not pd.isna(row.short_ma) else None,
            long_ma=row.long_ma if not pd.isna(row.long_ma) else None,
            rsi=row.rsi if not pd.isna(row.rsi) else None,
            atr=row.atr if not pd.isna(row.atr) else None,
            boll_upper=row.boll_upper if not pd.isna(row.boll_upper) else None,
            boll_lower=row.boll_lower if not pd.isna(row.boll_lower) else None,
            obv=(
                getattr(row, "obv", None)
                if not pd.isna(getattr(row, "obv", np.nan))
                else None
            ),
            mfi=(
                getattr(row, "mfi", None)
                if not pd.isna(getattr(row, "mfi", np.nan))
                else None
            ),
            cvd=(
                getattr(row, "cvd", None)
                if not pd.isna(getattr(row, "cvd", np.nan))
                else None
            ),
            ram=(
                getattr(row, "ram", None)
                if not pd.isna(getattr(row, "ram", np.nan))
                else None
            ),
            hurst=(
                getattr(row, "hurst", None)
                if not pd.isna(getattr(row, "hurst", np.nan))
                else None
            ),
            htf_ma=(
                getattr(row, "htf_ma", None)
                if not pd.isna(getattr(row, "htf_ma", np.nan))
                else None
            ),
            htf_rsi=(
                getattr(row, "htf_rsi", None)
                if not pd.isna(getattr(row, "htf_rsi", np.nan))
                else None
            ),
            supertrend_break=(
                getattr(row, "supertrend_break", None)
                if not pd.isna(getattr(row, "supertrend_break", np.nan))
                else None
            ),
            kama_cross=(
                getattr(row, "kama_cross", None)
                if not pd.isna(getattr(row, "kama_cross", np.nan))
                else None
            ),
            kma_cross=(
                getattr(row, "kma_cross", None)
                if not pd.isna(getattr(row, "kma_cross", np.nan))
                else None
            ),
            vwap_cross=(
                getattr(row, "vwap_cross", None)
                if not pd.isna(getattr(row, "vwap_cross", np.nan))
                else None
            ),
            macd_cross=(
                getattr(row, "macd_cross", None)
                if not pd.isna(getattr(row, "macd_cross", np.nan))
                else None
            ),
            squeeze_break=(
                getattr(row, "squeeze_break", None)
                if not pd.isna(getattr(row, "squeeze_break", np.nan))
                else None
            ),
            regime=(
                getattr(row, "regime", None)
                if not pd.isna(getattr(row, "regime", np.nan))
                else None
            ),
            microprice_delta=(
                getattr(row, "microprice_delta", None)
                if not pd.isna(getattr(row, "microprice_delta", np.nan))
                else None
            ),
        )
        sig = strat.update(price_val, ind)
        signals.append(sig)

        if (
            strat.position == 1
            and strat.entry_price is not None
            and strat.entry_atr is not None
        ):
            peak = strat.peak_price if strat.peak_price is not None else price_val
            long_stop = max(
                strat.entry_price - strat.entry_atr * strat.atr_stop_long,
                peak * (1 - strat.trailing_stop_pct),
            )
        else:
            long_stop = np.nan
        if (
            strat.position == -1
            and strat.entry_price is not None
            and strat.entry_atr is not None
        ):
            trough = strat.trough_price if strat.trough_price is not None else price_val
            short_stop = min(
                strat.entry_price + strat.entry_atr * strat.atr_stop_short,
                trough * (1 + strat.trailing_stop_pct),
            )
        else:
            short_stop = np.nan

        long_stops.append(long_stop)
        short_stops.append(short_stop)

    df["baseline_signal"] = signals
    df["baseline_long_stop"] = long_stops
    df["baseline_short_stop"] = short_stops
    return df


__all__ = ["compute"]
