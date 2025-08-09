"""Pandera schemas for validating dataframes."""

from __future__ import annotations

import pandera as pa

TICK_SCHEMA = pa.DataFrameSchema(
    {
        "Timestamp": pa.Column(pa.DateTime),
        "Bid": pa.Column(float),
        "Ask": pa.Column(float),
        "BidVolume": pa.Column(float, required=False, nullable=True),
        "AskVolume": pa.Column(float, required=False, nullable=True),
        "Symbol": pa.Column(str, required=False, nullable=True),
    },
    coerce=True,
)

FEATURE_SCHEMA = pa.DataFrameSchema(
    {
        "Timestamp": pa.Column(pa.DateTime),
        "return": pa.Column(float),
        "ma_5": pa.Column(float),
        "ma_10": pa.Column(float),
        "ma_30": pa.Column(float),
        "ma_60": pa.Column(float),
        "volatility_30": pa.Column(float),
        "rsi_14": pa.Column(float),
        "market_regime": pa.Column(int),
    },
    coerce=True,
)

LABEL_SCHEMA = pa.DataFrameSchema(
    {
        "target": pa.Column(int),
    },
    coerce=True,
)

__all__ = ["TICK_SCHEMA", "FEATURE_SCHEMA", "LABEL_SCHEMA"]
