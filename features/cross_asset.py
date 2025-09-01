"""Cross asset and market wide features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def _merge_asof(left: pd.DataFrame, right: pd.DataFrame, **kwargs) -> pd.DataFrame:
    return pd.merge_asof(left, right, **kwargs)


def load_index_ohlc(source: str) -> pd.DataFrame:
    try:
        if str(source).startswith("http"):
            df = pd.read_csv(source)
        else:
            df = pd.read_csv(Path(source))
    except Exception:
        return pd.DataFrame()
    if "Date" not in df.columns:
        df.columns = [c.strip() for c in df.columns]
    return df


def add_index_features(df: pd.DataFrame) -> pd.DataFrame:
    try:
        from utils import load_config

        cfg = load_config()
        index_sources = cfg.get("index_data", {})
    except Exception:  # pragma: no cover - config issues shouldn't fail
        index_sources = {}

    df = df.sort_values("Timestamp")
    for name, src in index_sources.items():
        idx_df = load_index_ohlc(src)
        if {"Date", "Close"}.issubset(idx_df.columns):
            idx_df["Date"] = pd.to_datetime(idx_df["Date"])
            idx_df = idx_df.sort_values("Date")
            idx_df["return"] = idx_df["Close"].pct_change()
            idx_df["vol"] = idx_df["return"].rolling(21).std()
            idx_df = idx_df[["Date", "return", "vol"]]
            idx_df = idx_df.rename(
                columns={
                    "Date": "index_date",
                    "return": f"{name}_ret",
                    "vol": f"{name}_vol",
                }
            )
            df = _merge_asof(
                df,
                idx_df,
                left_on="Timestamp",
                right_on="index_date",
                direction="backward",
            ).drop(columns=["index_date"])
        else:
            df[f"{name}_ret"] = 0.0
            df[f"{name}_vol"] = 0.0

    if not index_sources:
        for col in ["sp500_ret", "sp500_vol", "vix_ret", "vix_vol"]:
            if col not in df.columns:
                df[col] = 0.0
    return df


def add_cross_asset_features(df: pd.DataFrame) -> pd.DataFrame:
    # Placeholder for more advanced cross-asset relationships
    return df


def compute(df: pd.DataFrame) -> pd.DataFrame:
    from data import features as base

    df = base.add_index_features(df)
    df = base.add_cross_asset_features(df)

    if "Symbol" in df.columns:
        symbols = sorted(df["Symbol"].unique())
        size = len(symbols)
        eye = np.eye(size)
        matrices = {}
        for ts in pd.to_datetime(df["Timestamp"]).unique():
            matrices[ts] = eye
        df.attrs["adjacency_matrices"] = matrices
    return df


__all__ = ["add_index_features", "add_cross_asset_features", "compute"]
