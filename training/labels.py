"""Label generation utilities for the training pipeline."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

__all__ = ["generate_training_labels"]


def generate_training_labels(
    df: pd.DataFrame,
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

    horizons = list(horizons)
    if not horizons:
        return pd.DataFrame(index=df.index)
    from data.labels import multi_horizon_labels
    from data.streaming import stream_labels
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
