"""Utilities for streaming feature engineering and label generation.

This module provides lightweight helpers that mirror :func:`data.features.make_features`
and :func:`data.labels.multi_horizon_labels` but operate on iterators of
``pandas`` dataframes.  The helpers maintain small in-memory buffers so that
rolling statistics can be computed across chunk boundaries without loading the
entire dataset at once.  They emit processed chunks sequentially which allows
callers to update incremental trainers or progressively persist intermediate
results while keeping memory usage bounded.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field

import pandas as pd

from .features import make_features
from .labels import multi_horizon_labels


@dataclass
class _FeatureStreamState:
    """Track buffered state required when streaming feature computation."""

    validate: bool = False
    feature_lookback: int = 512
    buffer: pd.DataFrame = field(default_factory=pd.DataFrame)
    pending_start: int = 0

    def update(self, chunk: pd.DataFrame) -> Iterator[pd.DataFrame]:
        """Consume ``chunk`` and yield feature dataframes ready for use."""

        if chunk is None:
            return iter(())
        if chunk.empty and self.buffer.empty:
            return iter(())

        chunk = chunk.reset_index(drop=True)
        if self.buffer.empty:
            combined = chunk
        else:
            combined = pd.concat([self.buffer, chunk], ignore_index=True)

        features = make_features(combined, validate=self.validate)
        emit_start = self.pending_start
        emit_end = len(combined)
        if emit_end <= emit_start:
            ready: list[pd.DataFrame] = []
        else:
            ready = [features.iloc[emit_start:emit_end].reset_index(drop=True)]

        new_start = max(emit_end - self.feature_lookback, 0)
        self.buffer = combined.iloc[new_start:].reset_index(drop=True)
        self.pending_start = emit_end - new_start
        return iter(ready)

    def flush(self) -> Iterator[pd.DataFrame]:
        """Emit any remaining buffered rows when the stream ends."""

        if self.buffer.empty:
            return iter(())
        features = make_features(self.buffer, validate=self.validate)
        emit_start = self.pending_start
        if emit_start >= len(self.buffer):
            self.buffer = pd.DataFrame()
            self.pending_start = 0
            return iter(())
        ready = features.iloc[emit_start:].reset_index(drop=True)
        self.buffer = pd.DataFrame()
        self.pending_start = 0
        return iter([ready])


def stream_features(
    frames: Iterable[pd.DataFrame],
    *,
    validate: bool = False,
    feature_lookback: int = 512,
) -> Iterator[pd.DataFrame]:
    """Yield feature-engineered chunks from ``frames`` sequentially."""

    state = _FeatureStreamState(validate=validate, feature_lookback=feature_lookback)
    for frame in frames:
        yield from state.update(frame)
    yield from state.flush()


@dataclass
class _LabelStreamState:
    """Maintain context for streaming multi-horizon label computation."""

    horizons: Sequence[int]
    buffer: pd.DataFrame = field(default_factory=pd.DataFrame)
    pending_start: int = 0

    @property
    def label_context(self) -> int:
        return max((int(h) for h in self.horizons if int(h) > 0), default=0)

    def update(self, chunk: pd.DataFrame) -> Iterator[pd.DataFrame]:
        if chunk is None:
            return iter(())
        if chunk.empty and self.buffer.empty:
            return iter(())

        chunk = chunk.reset_index(drop=True)
        if self.buffer.empty:
            combined = chunk
        else:
            combined = pd.concat([self.buffer, chunk], ignore_index=True)

        if "mid" not in combined.columns:
            raise KeyError("Expected 'mid' column for label computation")

        labels = multi_horizon_labels(combined["mid"], list(self.horizons))
        emit_start = self.pending_start
        emit_end = len(combined) - self.label_context
        if emit_end < emit_start:
            emit_end = emit_start
        if emit_end > emit_start:
            ready = [labels.iloc[emit_start:emit_end].reset_index(drop=True)]
        else:
            ready = []

        if emit_end < len(combined):
            self.buffer = combined.iloc[emit_end:].reset_index(drop=True)
        else:
            self.buffer = pd.DataFrame()
        self.pending_start = 0
        return iter(ready)

    def flush(self) -> Iterator[pd.DataFrame]:
        if self.buffer.empty:
            return iter(())
        if "mid" not in self.buffer.columns:
            raise KeyError("Expected 'mid' column for label computation")
        labels = multi_horizon_labels(self.buffer["mid"], list(self.horizons))
        emit_start = self.pending_start
        if emit_start >= len(self.buffer):
            self.buffer = pd.DataFrame()
            self.pending_start = 0
            return iter(())
        ready = labels.iloc[emit_start:].reset_index(drop=True)
        self.buffer = pd.DataFrame()
        self.pending_start = 0
        return iter([ready])


def stream_labels(
    frames: Iterable[pd.DataFrame],
    horizons: Sequence[int],
) -> Iterator[pd.DataFrame]:
    """Yield label dataframes computed incrementally from ``frames``."""

    state = _LabelStreamState(horizons=horizons)
    for frame in frames:
        yield from state.update(frame)
    yield from state.flush()


__all__ = ["stream_features", "stream_labels"]
