"""Run BaselineStrategy over historical bars as a feature.

The function precomputes moving averages, RSI and ATR once and stores
them in the resulting feature matrix so that downstream models can
reuse these baseline indicators without recomputing them."""

from __future__ import annotations

import pandas as pd

try:  # pragma: no cover - polars optional
    import polars as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore

from strategies.baseline import BaselineStrategy, IndicatorBundle
from indicators.common import atr, bollinger, rsi, sma


def _compute_pandas(
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
    atr_vals = df.get("atr_val")
    if atr_vals is None:
        atr_vals = atr(high, low, price, atr_window)

    if "short_ma" not in df:
        df["short_ma"] = short_ma
    if "long_ma" not in df:
        df["long_ma"] = long_ma
    if "rsi" not in df:
        df["rsi"] = rsi_vals
    if "atr_val" not in df:
        df["atr_val"] = atr_vals
    if "boll_upper" not in df:
        df["boll_upper"] = boll_upper
    if "boll_lower" not in df:
        df["boll_lower"] = boll_lower

    bundle = IndicatorBundle(
        high=high,
        low=low,
        short_ma=df["short_ma"],
        long_ma=df["long_ma"],
        rsi=df["rsi"],
        atr_val=df["atr_val"],
        boll_upper=df["boll_upper"],
        boll_lower=df["boll_lower"],
        obv=df.get("obv"),
        mfi=df.get("mfi"),
        cvd=df.get("cvd"),
        ram=df.get("ram"),
        hurst=df.get("hurst"),
        htf_ma=df.get("htf_ma"),
        htf_rsi=df.get("htf_rsi"),
        supertrend_break=df.get("supertrend_break"),
        kama_cross=df.get("kama_cross"),
        kma_cross=df.get("kma_cross"),
        vwap_cross=df.get("vwap_cross"),
        macd_cross=df.get("macd_cross"),
        squeeze_break=df.get("squeeze_break"),
        div_rsi=df.get("div_rsi"),
        div_macd=df.get("div_macd"),
        regime=df.get("regime"),
        vae_regime=df.get("vae_regime"),
        microprice_delta=df.get("microprice_delta"),
        liq_exhaustion=df.get("liq_exhaustion"),
    )

    strat = BaselineStrategy(
        short_window=short_window,
        long_window=long_window,
        rsi_window=rsi_window,
        atr_window=atr_window,
        atr_stop_long=atr_stop_long,
        atr_stop_short=atr_stop_short,
        trailing_stop_pct=trailing_stop_pct,
        trailing_take_profit_pct=trailing_take_profit_pct,
        session_position_limits={},
        default_position_limit=1,
    )

    signals, long_stops, short_stops, confidence = strat.batch_update(price, bundle)

    index = df.index
    df["baseline_signal"] = pd.Series(signals, index=index, dtype=float)
    df["long_stop"] = pd.Series(long_stops, index=index, dtype=float)
    df["short_stop"] = pd.Series(short_stops, index=index, dtype=float)
    df["baseline_confidence"] = pd.Series(confidence, index=index, dtype=float)
    return df


def compute(
    df,
    short_window: int = 5,
    long_window: int = 20,
    rsi_window: int = 14,
    atr_window: int = 14,
    atr_stop_long: float = 3.0,
    atr_stop_short: float = 3.0,
    trailing_stop_pct: float = 0.01,
    trailing_take_profit_pct: float = 0.02,
):
    if pl is not None and isinstance(df, pl.DataFrame):
        pdf = df.to_pandas()
        result = _compute_pandas(
            pdf,
            short_window=short_window,
            long_window=long_window,
            rsi_window=rsi_window,
            atr_window=atr_window,
            atr_stop_long=atr_stop_long,
            atr_stop_short=atr_stop_short,
            trailing_stop_pct=trailing_stop_pct,
            trailing_take_profit_pct=trailing_take_profit_pct,
        )
        return pl.from_pandas(result)
    return _compute_pandas(
        df,
        short_window=short_window,
        long_window=long_window,
        rsi_window=rsi_window,
        atr_window=atr_window,
        atr_stop_long=atr_stop_long,
        atr_stop_short=atr_stop_short,
        trailing_stop_pct=trailing_stop_pct,
        trailing_take_profit_pct=trailing_take_profit_pct,
    )


compute.supports_polars = True  # type: ignore[attr-defined]


__all__ = ["compute"]
