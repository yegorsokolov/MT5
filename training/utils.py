"""Lightweight utilities shared across training components."""

from __future__ import annotations

import numpy as np

__all__ = ["combined_sample_weight"]


def combined_sample_weight(
    y: np.ndarray,
    timestamps: np.ndarray,
    t_max: int,
    balance: bool,
    half_life: int | None,
    dq_w: np.ndarray | None = None,
) -> np.ndarray | None:
    """Compute sample weights blending data quality, class balance and decay."""

    weights = np.ones(len(y), dtype=float)
    applied = False
    if dq_w is not None:
        weights *= dq_w
        applied = True
    if balance:
        unique, counts = np.unique(y, return_counts=True)
        class_weight = {cls: len(y) / (len(unique) * count) for cls, count in zip(unique, counts)}
        weights *= np.array([class_weight[val] for val in y], dtype=float)
        applied = True
    if half_life:
        decay = 0.5 ** ((t_max - timestamps) / half_life)
        weights *= decay
        applied = True
    return weights if applied else None
