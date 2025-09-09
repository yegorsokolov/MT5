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


def add_cross_asset_features(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """Add simple cross-asset interaction features.

    For every pair of symbols sharing the same ``Timestamp`` this function
    computes:

    - Rolling correlation of their returns over ``window`` periods, appended as
      ``corr_<sym1>_<sym2>`` for rows where ``Symbol`` is ``sym1``.
    - The ratio of their instantaneous returns, appended as
      ``relret_<sym1>_<sym2>`` for the same rows.

    Missing values (for example during the warm up period of the rolling
    correlation) are filled with ``0.0`` to keep downstream models simple.
    """

    required = {"Symbol", "Timestamp", "return"}
    if not required.issubset(df.columns):
        return df

    df = df.copy().sort_values("Timestamp")
    pivot = df.pivot(index="Timestamp", columns="Symbol", values="return")
    symbols = list(pivot.columns)

    # ------------------------------------------------------------------
    # Cross-sectional relative strength
    # ------------------------------------------------------------------
    cs_mean = pivot.mean(axis=1)
    cs_std = pivot.std(axis=1, ddof=0).replace(0, np.nan)
    rel_strength = pivot.sub(cs_mean, axis=0).div(cs_std, axis=0)

    for sym in symbols:
        ts_map = df.loc[df["Symbol"] == sym, "Timestamp"]
        df.loc[df["Symbol"] == sym, f"rel_strength_{sym}"] = ts_map.map(
            rel_strength[sym]
        ).fillna(0.0)

    # ------------------------------------------------------------------
    # Pairwise interactions
    # ------------------------------------------------------------------
    for sym1 in symbols:
        for sym2 in symbols:
            if sym1 == sym2:
                continue

            corr_series = pivot[sym1].rolling(window).corr(pivot[sym2])
            ratio_series = (pivot[sym1] / pivot[sym2]).replace(
                [np.inf, -np.inf], np.nan
            )

            ts_map = df.loc[df["Symbol"] == sym1, "Timestamp"]
            df.loc[df["Symbol"] == sym1, f"corr_{sym1}_{sym2}"] = ts_map.map(
                corr_series
            ).fillna(0.0)
            df.loc[df["Symbol"] == sym1, f"relret_{sym1}_{sym2}"] = ts_map.map(
                ratio_series
            ).fillna(0.0)

    return df


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Enrich ``df`` with index and cross-asset features.

    The following columns are appended:

    - Market index returns/volatility via :func:`add_index_features`.
    - Pairwise rolling correlations (``corr_<sym1>_<sym2>``) and relative
      return ratios (``relret_<sym1>_<sym2>``) via
      :func:`add_cross_asset_features`.

    Additionally, rolling adjacency matrices describing the cross-symbol
    relationships are stored in ``df.attrs['adjacency_matrices']``.
    """

    from data.graph_builder import build_rolling_adjacency

    df = add_index_features(df)
    df = add_cross_asset_features(df)

    if {"Symbol", "Timestamp"}.issubset(df.columns):
        try:
            matrices = build_rolling_adjacency(df)
        except Exception:  # pragma: no cover - fallback to identity
            symbols = sorted(df["Symbol"].unique())
            eye = np.eye(len(symbols))
            matrices = {ts: eye for ts in pd.to_datetime(df["Timestamp"]).unique()}
        df.attrs["adjacency_matrices"] = matrices
    return df


__all__ = ["add_index_features", "add_cross_asset_features", "compute"]
