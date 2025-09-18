"""Utilities for loading training datasets.

This module centralises the logic that loads historical market data for the
training pipeline.  The helpers are intentionally lightweight wrappers around
existing data-access utilities so that they can be reused by both the standard
trainer and alternative entry points such as the parallel trainer.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

import pandas as pd

from config_models import AppConfig

__all__ = ["load_training_frame", "StreamingTrainingFrame"]


@dataclass
class StreamingTrainingFrame:
    """Container for lazily materialised feature chunks."""

    chunks: Iterable[pd.DataFrame] | Iterator[pd.DataFrame]
    metadata: dict[str, object] = field(default_factory=dict)
    persist_path: Path | None = None
    _cache: list[pd.DataFrame] = field(default_factory=list, init=False, repr=False)
    _iterator: Iterator[pd.DataFrame] = field(init=False, repr=False)
    _materialised: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _persisted: bool = field(default=False, init=False, repr=False)
    materialise_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._iterator = iter(self.chunks)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        for chunk in self._cache:
            yield chunk
        for chunk in self._iterator:
            self._cache.append(chunk)
            yield chunk
        self._iterator = iter(())

    def materialise(self) -> pd.DataFrame:
        """Return the concatenated dataframe, caching the result."""

        if self._materialised is None:
            self.materialise_count += 1
            frames = list(self)
            if frames:
                self._materialised = pd.concat(frames, ignore_index=True)
            else:
                self._materialised = pd.DataFrame()
            if self.persist_path is not None and not self._persisted:
                try:
                    from data.history import save_history_parquet

                    save_history_parquet(self._materialised, self.persist_path)
                    self._persisted = True
                except Exception:  # pragma: no cover - best effort persistence
                    pass
        return self._materialised if self._materialised is not None else pd.DataFrame()


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
) -> Iterator[pd.DataFrame]:
    """Yield feature-engineered chunks assembled from streaming history."""

    from data.streaming import stream_features

    for sym in symbols:
        chunks = _symbol_history_chunks(
            sym, cfg, root, chunk_size=chunk_size, validate=validate
        )
        yield from stream_features(
            chunks,
            validate=validate,
            feature_lookback=feature_lookback,
        )


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
) -> Tuple[pd.DataFrame | StreamingTrainingFrame, str]:
    """Return the dataframe or streaming iterator used for training.

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
    Tuple[pd.DataFrame | StreamingTrainingFrame, str]
        Either a fully materialised feature dataframe or a
        :class:`StreamingTrainingFrame` when ``stream=True`` along with a string
        describing where the data originated from (``"override"`` or
        ``"config"``).
    """

    if df_override is not None:
        return df_override, "override"

    symbols = cfg.strategy.symbols
    if stream:
        iterator = _collect_streaming_features(
            symbols,
            cfg,
            root,
            chunk_size=chunk_size,
            feature_lookback=feature_lookback,
            validate=validate,
        )
        metadata = {
            "symbols": list(symbols),
            "chunk_size": int(chunk_size),
            "feature_lookback": int(feature_lookback),
            "validate": bool(validate),
        }
        frame = StreamingTrainingFrame(
            iterator,
            metadata=metadata,
            persist_path=root / "data" / "history.parquet",
        )
    else:
        frame = _collect_offline_features(symbols, cfg, root, validate=validate)
    return frame, "config"
