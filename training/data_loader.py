"""Utilities for loading training datasets.

This module centralises the logic that loads historical market data for the
training pipeline.  The helpers are intentionally lightweight wrappers around
existing data-access utilities so that they can be reused by both the standard
trainer and alternative entry points such as the parallel trainer.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
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
    _chunk_lengths: list[int] = field(default_factory=list, init=False, repr=False)
    _iterator: Iterator[pd.DataFrame] = field(init=False, repr=False)
    _materialised: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _persisted: bool = field(default=False, init=False, repr=False)
    _columns: set[str] = field(default_factory=set, init=False, repr=False)
    _column_order: list[str] = field(default_factory=list, init=False, repr=False)
    _post_materialise: list[Callable[[pd.DataFrame], pd.DataFrame]] = field(
        default_factory=list, init=False, repr=False
    )
    materialise_count: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self._iterator = iter(self.chunks)

    def __iter__(self) -> Iterator[pd.DataFrame]:
        for chunk in self._cache:
            self._record_columns(chunk.columns)
            yield chunk
        for chunk in self._iterator:
            self._cache.append(chunk)
            self._chunk_lengths.append(len(chunk))
            self._record_columns(chunk.columns)
            yield chunk
        self._iterator = iter(())

    def _record_columns(self, columns: Iterable[str]) -> None:
        for col in columns:
            if col not in self._columns:
                self._columns.add(col)
                self._column_order.append(col)

    def _refresh_columns_from_cache(self) -> None:
        seen: set[str] = set()
        ordered: list[str] = []
        for chunk in self._cache:
            for col in chunk.columns:
                if col not in seen:
                    seen.add(col)
                    ordered.append(col)
        self._columns = seen
        self._column_order = ordered

    def materialise(self) -> pd.DataFrame:
        """Return the concatenated dataframe, caching the result."""

        if self._materialised is None:
            self.materialise_count += 1
            frames = list(self)
            if frames:
                self._materialised = pd.concat(frames, ignore_index=True)
            else:
                self._materialised = pd.DataFrame()
            for fn in list(self._post_materialise):
                self._materialised = fn(self._materialised)
            self._post_materialise.clear()
            if self._materialised is not None:
                if self._chunk_lengths and sum(self._chunk_lengths) == len(self._materialised):
                    new_cache: list[pd.DataFrame] = []
                    start = 0
                    for length in self._chunk_lengths:
                        stop = start + length
                        new_cache.append(
                            self._materialised.iloc[start:stop].reset_index(drop=True)
                        )
                        start = stop
                    self._cache = new_cache
                else:
                    self._cache = [self._materialised]
                    self._chunk_lengths = [len(self._materialised)]
                self._refresh_columns_from_cache()
            if self.persist_path is not None and not self._persisted:
                try:
                    from data.history import save_history_parquet

                    save_history_parquet(self._materialised, self.persist_path)
                    self._persisted = True
                except Exception:  # pragma: no cover - best effort persistence
                    pass
        return self._materialised if self._materialised is not None else pd.DataFrame()

    def apply_chunk(
        self, func: Callable[[pd.DataFrame], pd.DataFrame | None], *, copy: bool = True
    ) -> "StreamingTrainingFrame":
        """Apply ``func`` to each chunk without materialising the full frame."""

        def _transform(chunk: pd.DataFrame) -> pd.DataFrame:
            data = chunk.copy(deep=True) if copy else chunk
            result = func(data)
            return data if result is None else result

        self._cache = [_transform(chunk) for chunk in self._cache]
        self._chunk_lengths = [len(chunk) for chunk in self._cache]
        iterator = self._iterator

        def _iterator_wrapper() -> Iterator[pd.DataFrame]:
            for chunk in iterator:
                yield _transform(chunk)

        self._iterator = _iterator_wrapper()
        self._materialised = None
        self._persisted = False
        self._refresh_columns_from_cache()
        return self

    def apply_full(
        self, func: Callable[[pd.DataFrame], pd.DataFrame]
    ) -> "StreamingTrainingFrame":
        """Register ``func`` to run when the frame materialises."""

        self._post_materialise.append(func)
        self._materialised = None
        self._persisted = False
        return self

    def collect_chunks(self) -> list[pd.DataFrame]:
        """Ensure all chunks are cached and return them."""

        for _ in self:
            pass
        return list(self._cache)

    def collect_columns(self) -> list[str]:
        """Return column names encountered so far."""

        if not self._column_order:
            for _ in self:
                pass
        return list(self._column_order)

    def chunk_lengths(self) -> list[int]:
        """Return the lengths of cached chunks."""

        self.collect_chunks()
        return list(self._chunk_lengths)

    def peek(self, rows: int = 5) -> pd.DataFrame:
        """Return the first chunk (optionally truncated) without materialising."""

        for chunk in self:
            return chunk.head(rows).copy()
        return pd.DataFrame()

    def __len__(self) -> int:
        lengths = self.chunk_lengths()
        return int(sum(lengths))

    @property
    def columns(self) -> list[str]:
        return self.collect_columns()


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
