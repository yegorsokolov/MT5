"""Price based technical indicators."""

from __future__ import annotations

import numpy as np
import pandas as pd

from indicators import atr, bollinger, rsi, sma

try:  # pragma: no cover - decorator optional when imported standalone
    from . import validate_module
except Exception:  # pragma: no cover - fallback without validation
    def validate_module(func):
        return func


@validate_module
def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard price/volume technical features."""
    df = df.copy()
    df["spread"] = df["Ask"] - df["Bid"]
    df["mid"] = (df["Ask"] + df["Bid"]) / 2
    df["return"] = df["mid"].pct_change().fillna(0)

    # moving averages
    for win in [5, 10, 30, 60]:
        df[f"ma_{win}"] = sma(df["mid"], win)
    df["ma_h4"] = sma(df["mid"], 4 * 60)

    # Bollinger bands
    _, df["boll_upper"], df["boll_lower"] = bollinger(df["mid"], 20)
    df["boll_break"] = (
        (df["mid"] > df["boll_upper"]).astype(int)
        - (df["mid"] < df["boll_lower"]).astype(int)
    )

    # ATR (simplified using high/low/mid)
    hl = df.get("High", df["mid"])
    ll = df.get("Low", df["mid"])
    df["atr_14"] = atr(hl, ll, df["mid"], 14)
    df["atr_stop_long"] = df["mid"] - 3 * df["atr_14"]
    df["atr_stop_short"] = df["mid"] + 3 * df["atr_14"]

    # Donchian channel
    roll = df["mid"].rolling(20)
    df["donchian_high"] = roll.max()
    df["donchian_low"] = roll.min()
    df["donchian_break"] = (
        (df["mid"] > df["donchian_high"]).astype(int)
        - (df["mid"] < df["donchian_low"]).astype(int)
    )

    df["volatility_30"] = df["return"].rolling(30).std()
    df["rsi_14"] = rsi(df["mid"], 14)

    df["mid_change"] = df["mid"].diff().fillna(0)
    df["spread_change"] = df["spread"].diff().fillna(0)
    df["trade_rate"] = 0.0
    df["quote_revision"] = 0.0

    hours = pd.to_datetime(df["Timestamp"]).dt.hour
    df["hour"] = hours
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    df["ma_cross"] = (df["ma_5"] > df["ma_10"]).astype(int)
    df["market_regime"] = 0

    return df

__all__ = ["compute"]
