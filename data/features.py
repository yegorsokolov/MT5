"""Lightweight feature engineering orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from features import get_feature_pipeline
from features.news import (
    add_economic_calendar_features,
    add_news_sentiment_features,
)
from features.cross_asset import (
    add_index_features,
    add_cross_asset_features,
)
from utils.resource_monitor import monitor, ResourceCapabilities
from analysis import feature_gate

logger = logging.getLogger(__name__)


def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """Downcast common numeric dtypes to reduce memory usage.

    ``float64`` columns are converted to ``float32`` and ``int64`` columns
    to ``int32``.  The transformation is applied in-place and the original
    DataFrame is returned for convenience.
    """

    float_cols = df.select_dtypes(include=["float64"]).columns
    int_cols = df.select_dtypes(include=["int64"]).columns

    if len(float_cols) > 0:
        df[float_cols] = df[float_cols].astype("float32")
    if len(int_cols) > 0:
        df[int_cols] = df[int_cols].astype("int32")

    return df


def add_alt_features(df: pd.DataFrame) -> pd.DataFrame:
    """Merge alternative datasets into ``df``.

    This function loads shipping metrics, retail transactions and
    weather series (as well as legacy options/on-chain/ESG metrics) using
    :func:`data.alt_data_loader.load_alt_data` and performs a
    ``merge_asof`` on ``Timestamp`` and ``Symbol``.  Missing columns are
    forward filled and remaining gaps replaced with ``0`` so downstream
    models can rely on their presence.
    """

    if "Symbol" not in df.columns:
        return df

    from .alt_data_loader import load_alt_data

    alt = load_alt_data(sorted(df["Symbol"].unique()))
    if not alt.empty:
        alt = alt.rename(columns={"Date": "alt_date"})
        df = pd.merge_asof(
            df.sort_values("Timestamp"),
            alt.sort_values("alt_date"),
            left_on="Timestamp",
            right_on="alt_date",
            by="Symbol",
            direction="backward",
        ).drop(columns=["alt_date"])

    for col in [
        "implied_vol",
        "active_addresses",
        "esg_score",
        "shipping_metric",
        "retail_sales",
        "temperature",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)

    return df


def add_corporate_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Merge dividend, split and ownership data into ``df``.

    The function loads corporate action datasets via
    :func:`data.corporate_actions.load_corporate_actions` and performs a
    ``merge_asof`` on ``Timestamp`` and ``Symbol``.  Missing columns are
    forward filled with ``0`` to provide stable inputs for downstream
    models.  The loader is marked with ``min_capability='standard'`` so
    constrained environments can omit these heavier features.
    """

    if "Symbol" not in df.columns:
        return df

    from .corporate_actions import load_corporate_actions

    actions = load_corporate_actions(sorted(df["Symbol"].unique()))
    if not actions.empty:
        actions = actions.rename(columns={"Date": "corp_date"})
        df = pd.merge_asof(
            df.sort_values("Timestamp"),
            actions.sort_values("corp_date"),
            left_on="Timestamp",
            right_on="corp_date",
            by="Symbol",
            direction="backward",
        ).drop(columns=["corp_date"])

    for col in [
        "dividend",
        "split",
        "insider_trades",
        "institutional_holdings",
    ]:
        if col not in df.columns:
            df[col] = 0.0
        else:
            df[col] = df[col].ffill().fillna(0.0)

    return df


# Minimum capability required for corporate actions
add_corporate_actions.min_capability = "standard"  # type: ignore[attr-defined]


def make_features(df: pd.DataFrame, validate: bool = False) -> pd.DataFrame:
    """Generate model features by executing registered modules sequentially.

    In addition to the lightweight feature modules registered in
    :func:`features.get_feature_pipeline`, this function optionally merges
    external datasets such as fundamentals, options implied volatility and
    on-chain metrics.  These heavier data sources are only loaded when the
    :class:`utils.resource_monitor.ResourceMonitor` reports sufficient
    capabilities to avoid overwhelming constrained environments.  The final
    feature set is then passed through :func:`analysis.feature_gate.select` to
    drop low-importance or heavy features for the current capability tier and
    market regime, ensuring consistent behaviour across runs.
    """

    for compute in get_feature_pipeline():
        df = compute(df)

    # Allow runtime plugins to extend the feature set
    adjacency = df.attrs.get("adjacency_matrices")
    try:
        import dataset  # type: ignore

        plugins = getattr(dataset, "FEATURE_PLUGINS", [])
    except Exception:
        plugins = []
    for plugin in plugins:
        df = plugin(df, adjacency_matrices=adjacency)

    # Optionally compute cross-spectral coherence features on capable machines
    try:
        from analysis import cross_spectral

        caps = getattr(
            monitor,
            "capabilities",
            ResourceCapabilities(cpus=1, memory_gb=0.0, has_gpu=False, gpu_count=0),
        )
        req = getattr(cross_spectral, "REQUIREMENTS", caps)
        if (
            caps.cpus >= req.cpus
            and caps.memory_gb >= req.memory_gb
            and (caps.has_gpu or not req.has_gpu)
        ):
            df = cross_spectral.compute(df)
    except Exception:
        logger.debug("cross spectral computation failed", exc_info=True)

    tier = getattr(monitor, "capability_tier", "lite")
    if tier in {"standard", "gpu", "hpc"}:
        try:
            from .fundamental_loader import load_fundamental_data

            if "Symbol" in df.columns:
                fundamentals = load_fundamental_data(sorted(df["Symbol"].unique()))
            else:
                fundamentals = pd.DataFrame()
            if not fundamentals.empty and "Symbol" in df.columns:
                fundamentals = fundamentals.rename(columns={"Date": "fund_date"})
                df = pd.merge_asof(
                    df.sort_values("Timestamp"),
                    fundamentals.sort_values("fund_date"),
                    left_on="Timestamp",
                    right_on="fund_date",
                    by="Symbol",
                    direction="backward",
                ).drop(columns=["fund_date"])
            for col in [
                "revenue",
                "net_income",
                "pe_ratio",
                "dividend_yield",
            ]:
                if col not in df.columns:
                    df[col] = 0.0
                else:
                    df[col] = df[col].ffill().fillna(0.0)
        except Exception:  # pragma: no cover - optional dependency failures
            logger.debug("fundamental data merge failed", exc_info=True)
            for col in [
                "revenue",
                "net_income",
                "pe_ratio",
                "dividend_yield",
            ]:
                if col not in df.columns:
                    df[col] = 0.0
        try:
            from .macro_features import load_macro_features

            df = load_macro_features(df)
        except Exception:
            logger.debug("macro feature merge failed", exc_info=True)
            for col in ["macro_gdp", "macro_cpi", "macro_interest_rate"]:
                if col not in df.columns:
                    df[col] = 0.0
        try:
            from news.stock_headlines import load_scores as load_headline_scores

            if "Symbol" in df.columns:
                headlines = load_headline_scores()
            else:
                headlines = pd.DataFrame()
            if not headlines.empty and "Symbol" in df.columns:
                headlines = headlines.rename(
                    columns={"symbol": "Symbol", "timestamp": "headline_time"}
                )
                df = pd.merge_asof(
                    df.sort_values("Timestamp"),
                    headlines.sort_values("headline_time"),
                    left_on="Timestamp",
                    right_on="headline_time",
                    by="Symbol",
                    direction="backward",
                ).drop(columns=["headline_time"])
            if "news_movement_score" not in df.columns:
                df["news_movement_score"] = 0.0
        except Exception:
            logger.debug("news movement merge failed", exc_info=True)
            df["news_movement_score"] = 0.0
        try:
            from news.sentiment_score import load_vectors as load_news_vectors

            vectors = load_news_vectors()
            if not vectors.empty and "Symbol" in df.columns:
                vectors = vectors.rename(
                    columns={"symbol": "Symbol", "timestamp": "news_time"}
                )
                df = pd.merge_asof(
                    df.sort_values("Timestamp"),
                    vectors.sort_values("news_time"),
                    left_on="Timestamp",
                    right_on="news_time",
                    by="Symbol",
                    direction="backward",
                ).drop(columns=["news_time"])
            for k in range(3):
                if f"news_sentiment_{k}" not in df.columns:
                    df[f"news_sentiment_{k}"] = 0.0
                if f"news_impact_{k}" not in df.columns:
                    df[f"news_impact_{k}"] = 0.0
        except Exception:
            logger.debug("news sentiment merge failed", exc_info=True)
            for k in range(3):
                df[f"news_sentiment_{k}"] = 0.0
                df[f"news_impact_{k}"] = 0.0
        try:
            df = add_corporate_actions(df)
        except Exception:
            logger.debug("corporate actions merge failed", exc_info=True)
            for col in [
                "dividend",
                "split",
                "insider_trades",
                "institutional_holdings",
            ]:
                if col not in df.columns:
                    df[col] = 0.0
    if tier in {"gpu", "hpc"}:
        try:
            df = add_alt_features(df)
        except Exception:
            logger.debug("alternative data merge failed", exc_info=True)
            for col in [
                "implied_vol",
                "active_addresses",
                "esg_score",
                "shipping_metric",
                "retail_sales",
                "temperature",
            ]:
                if col not in df.columns:
                    df[col] = 0.0

    if validate:
        try:
            from .validators import FEATURE_SCHEMA

            FEATURE_SCHEMA.validate(df, lazy=True)
        except Exception:
            logger.debug("feature validation failed", exc_info=True)

    # Drop features based on capability tier and regime specific gating.  The
    # gate is computed offline and persisted for deterministic behaviour, so at
    # runtime we simply apply the stored selection without recomputing
    # importances.
    regime_id = int(df["market_regime"].iloc[-1]) if "market_regime" in df.columns else 0
    df, _ = feature_gate.select(df, tier, regime_id, persist=False)

    df = optimize_dtypes(df)

    return df


def make_features_memmap(path: str | Path, chunk_size: int = 1000) -> pd.DataFrame:
    """Load history from ``path`` and compute features."""
    df = pd.read_parquet(path)
    return make_features(df, validate=False)


# -- Technical helpers -------------------------------------------------

def compute_rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def ma_cross_signal(df: pd.DataFrame, short: str = "ma_10", long: str = "ma_30") -> pd.Series:
    cross_up = (df[short] > df[long]) & (df[short].shift(1) <= df[long].shift(1))
    cross_down = (df[short] < df[long]) & (df[short].shift(1) >= df[long].shift(1))
    signal = pd.Series(0, index=df.index)
    signal[cross_up] = 1
    signal[cross_down] = -1
    return signal


def train_test_split(df: pd.DataFrame, n_train: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if "Symbol" in df.columns:
        trains: List[pd.DataFrame] = []
        tests: List[pd.DataFrame] = []
        for _, g in df.groupby("Symbol"):
            trains.append(g.iloc[:n_train].copy())
            tests.append(g.iloc[n_train:].copy())
        return pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)
    train = df.iloc[:n_train].copy()
    test = df.iloc[n_train:].copy()
    return train, test


def make_sequence_arrays(
    df: pd.DataFrame, features: List[str], seq_len: int, label_col: str = "return"
) -> Tuple[np.ndarray, np.ndarray]:
    X_list: List[np.ndarray] = []
    y_list: List[int] = []
    groups = [df]
    if "Symbol" in df.columns:
        groups = [g for _, g in df.groupby("Symbol")]
    for g in groups:
        values = g[features].values
        if label_col == "return":
            targets = (g["return"].shift(-1) > 0).astype(int).values
            limit = len(g) - 1
        else:
            targets = g[label_col].values
            limit = len(g)
        for i in range(seq_len, limit):
            X_list.append(values[i - seq_len : i])
            y_list.append(targets[i])
    X = np.stack(X_list)
    y = np.array(y_list)
    return X, y


def apply_evolved_features(
    df: pd.DataFrame, store_dir: str | Path | None = None
) -> pd.DataFrame:
    """Apply evolved features stored in ``feature_store`` to ``df``."""

    from analysis.feature_evolver import FeatureEvolver

    evolver = FeatureEvolver(store_dir=store_dir)
    return evolver.apply_stored_features(df)


__all__ = [
    "add_index_features",
    "add_economic_calendar_features",
    "add_news_sentiment_features",
    "add_cross_asset_features",
    "make_features",
    "make_features_memmap",
    "compute_rsi",
    "ma_cross_signal",
    "train_test_split",
    "make_sequence_arrays",
    "apply_evolved_features",
]

# Evolved features will be appended below by ``FeatureEvolver``.
