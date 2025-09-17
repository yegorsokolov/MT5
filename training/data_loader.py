"""Utilities for loading training datasets.

This module centralises the logic that loads historical market data for the
training pipeline.  The helpers are intentionally lightweight wrappers around
existing data-access utilities so that they can be reused by both the standard
trainer and alternative entry points such as the parallel trainer.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Tuple

import pandas as pd

from config_models import AppConfig

__all__ = ["load_training_frame"]


def _symbol_history_chunks(
    symbol: str,
    cfg: AppConfig,
    root: Path,
    *,
    chunk_size: int,
    validate: bool,
) -> Iterator[pd.DataFrame]:
    """Yield chunks of raw history for ``symbol`` with symbol column attached."""

    from data.history import load_history_config, load_history_iter

    pq_path = root / "data" / f"{symbol}_history.parquet"
    if pq_path.exists():
        for chunk in load_history_iter(pq_path, chunk_size):
            out = chunk.copy()
            out["Symbol"] = symbol
            yield out
    else:
        frame = load_history_config(symbol, cfg, root, validate=validate)
        frame["Symbol"] = symbol
        yield frame


def _collect_streaming_features(
    symbols: list[str],
    cfg: AppConfig,
    root: Path,
    *,
    chunk_size: int,
    feature_lookback: int,
    validate: bool,
) -> pd.DataFrame:
    """Return feature dataframe assembled from streaming chunks."""

    from data.history import save_history_parquet
    from data.streaming import stream_features

    feature_chunks: list[pd.DataFrame] = []
    for sym in symbols:
        chunks = _symbol_history_chunks(
            sym, cfg, root, chunk_size=chunk_size, validate=validate
        )
        for feat_chunk in stream_features(
            chunks,
            validate=validate,
            feature_lookback=feature_lookback,
        ):
            feature_chunks.append(feat_chunk)
    if not feature_chunks:
        return pd.DataFrame()
    df = pd.concat(feature_chunks, ignore_index=True)
    save_history_parquet(df, root / "data" / "history.parquet")
    return df


def _collect_offline_features(
    symbols: list[str],
    cfg: AppConfig,
    root: Path,
    *,
    validate: bool,
) -> pd.DataFrame:
    """Return feature dataframe built from offline history downloads."""

    from data.features import make_features
    from data.history import load_history_config, save_history_parquet

    raw_frames: list[pd.DataFrame] = []
    for sym in symbols:
        df_sym = load_history_config(sym, cfg, root, validate=validate)
        df_sym["Symbol"] = sym
        raw_frames.append(df_sym)
    if not raw_frames:
        return pd.DataFrame()
    df_raw = pd.concat(raw_frames, ignore_index=True)
    save_history_parquet(df_raw, root / "data" / "history.parquet")
    return make_features(df_raw, validate=validate)


def load_training_frame(
    cfg: AppConfig,
    root: Path,
    *,
    df_override: pd.DataFrame | None = None,
    stream: bool = False,
    chunk_size: int = 100_000,
    feature_lookback: int = 512,
    validate: bool = False,
) -> Tuple[pd.DataFrame, str]:
    """Return the dataframe used for training along with its provenance.

    Parameters
    ----------
    cfg:
        The resolved application configuration.
    root:
        Base path used to persist cached artefacts.
    df_override:
        Optional pre-computed feature dataframe.  When supplied it is returned
        directly and the provenance string is ``"override"``.
    stream:
        Whether features should be generated via the streaming pipeline.
    chunk_size:
        Chunk size used when iterating historical parquet files.
    feature_lookback:
        Lookback window used by the streaming feature generator.
    validate:
        When ``True`` additional validation checks are run while constructing
        features.

    Returns
    -------
    Tuple[pd.DataFrame, str]
        The feature dataframe and a string describing where the data originated
        from (``"override"`` or ``"config"``).
    """

    if df_override is not None:
        return df_override, "override"

    symbols = cfg.strategy.symbols
    if stream:
        df = _collect_streaming_features(
            symbols,
            cfg,
            root,
            chunk_size=chunk_size,
            feature_lookback=feature_lookback,
            validate=validate,
        )
    else:
        df = _collect_offline_features(symbols, cfg, root, validate=validate)
    return df, "config"
