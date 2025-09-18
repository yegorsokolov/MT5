"""Label generation utilities for the training pipeline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd
import numpy as np

from training.data_loader import StreamingTrainingFrame

__all__ = ["generate_training_labels"]


def generate_training_labels(
    df: pd.DataFrame | StreamingTrainingFrame,
    *,
    stream: bool,
    horizons: Iterable[int],
    chunk_size: int,
) -> pd.DataFrame:
    """Create the training label dataframe for ``df``.

    When ``stream`` is ``True`` the function mirrors the behaviour of the
    original trainer by iterating over chunks of ``df`` and computing labels for
    each chunk individually.  Otherwise labels are produced from the full
    ``df`` in one go.
    """

    horizons = [int(h) for h in horizons]
    from data.labels import multi_horizon_labels
    from data.streaming import stream_labels

    if isinstance(df, StreamingTrainingFrame):
        lengths = df.chunk_lengths()
        total_length = sum(lengths)
        if not horizons:
            df.apply_chunk(lambda chunk: chunk)
            return pd.DataFrame(index=pd.RangeIndex(total_length))

        label_context = max((h for h in horizons if h > 0), default=0)
        sample_size = max(label_context + 1, 1)
        sample_series = pd.Series(np.zeros(sample_size, dtype=float))
        label_template = multi_horizon_labels(sample_series, horizons)
        label_columns = list(label_template.columns)

        chunks = df.collect_chunks()
        if total_length == 0:
            df.apply_chunk(lambda chunk: chunk)
            return pd.DataFrame(columns=label_columns)

        label_frames = list(
            stream_labels((chunk.reset_index(drop=True) for chunk in chunks), horizons)
        )
        if label_frames:
            labels = pd.concat(label_frames, ignore_index=True)
        else:
            labels = pd.DataFrame(columns=label_columns)
        if labels.empty and label_columns:
            labels = pd.DataFrame(columns=label_columns)
        if label_columns and set(labels.columns) != set(label_columns):
            labels = labels.reindex(columns=label_columns, fill_value=0)
        labels = labels.reset_index(drop=True)
        labels = labels.reindex(range(total_length), fill_value=0)

        aligned: list[pd.DataFrame] = []
        start = 0
        for length in lengths:
            end = start + length
            chunk_labels = labels.iloc[start:end].reset_index(drop=True)
            if len(chunk_labels) < length:
                filler = pd.DataFrame(0, index=range(length - len(chunk_labels)), columns=labels.columns)
                chunk_labels = pd.concat([chunk_labels, filler], ignore_index=True)
            chunk_labels = chunk_labels.reindex(columns=labels.columns, fill_value=0)
            aligned.append(chunk_labels)
            start = end

        aligned_iter = iter(aligned)

        def _attach(chunk: pd.DataFrame) -> pd.DataFrame:
            labels_chunk = next(aligned_iter, pd.DataFrame(columns=labels.columns))
            labels_chunk = labels_chunk.reindex(range(len(chunk)), fill_value=0)
            labels_chunk.index = chunk.index
            result = chunk.copy()
            for name in labels.columns:
                result[name] = labels_chunk.get(name, 0)
            return result

        if aligned:
            df.apply_chunk(_attach)
        else:
            df.apply_chunk(lambda chunk: chunk)
        return labels

    horizons = list(horizons)
    if not horizons:
        return pd.DataFrame(index=df.index)
    if stream:
        if df.empty:
            return pd.DataFrame(index=df.index)
        step = max(int(chunk_size), 1)
        label_chunks = list(
            stream_labels(
                (df.iloc[start : start + step].copy() for start in range(0, len(df), step)),
                horizons,
            )
        )
        if not label_chunks:
            return pd.DataFrame(index=pd.RangeIndex(len(df)))
        labels = pd.concat(label_chunks, ignore_index=True)
    else:
        labels = multi_horizon_labels(df["mid"], horizons)
    labels = labels.reset_index(drop=True)
    if len(labels) != len(df):
        labels = labels.reindex(range(len(df)), fill_value=0)
    labels.index = df.index
    return labels
