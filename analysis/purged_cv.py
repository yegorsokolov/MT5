"""Cross-validation utilities for time series.

This module provides :class:`PurgedTimeSeriesSplit`, a splitter similar to
:class:`sklearn.model_selection.TimeSeriesSplit` but with an additional
*embargo* window.  Samples falling within the embargo window directly
preceding a validation fold are excluded from the training indices to reduce
look-ahead bias.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence


class PurgedTimeSeriesSplit:
    """Time series cross-validator with an embargo window.

    Parameters
    ----------
    n_splits : int, default 5
        Number of splits.  Must be at least 1 and less than the number of
        samples in the dataset.
    embargo : int, default 0
        Number of observations to exclude immediately before each validation
        fold.  This removes potential leakage from labels that extend into the
        validation period.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        self.n_splits = n_splits
        self.embargo = max(0, int(embargo))

    def split(
        self,
        X: Sequence,
        y: Sequence | None = None,
        groups: Sequence | None = None,
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Generate indices to split ``X`` into train and validation sets.

        The returned training indices contain only samples that occur before
        the validation fold and are at least ``embargo`` steps away from the
        start of the validation period.
        """
        n_samples = len(X)
        if self.n_splits >= n_samples:
            raise ValueError("Cannot have number of splits >= number of samples")

        test_size = n_samples // (self.n_splits + 1)
        if test_size == 0:
            raise ValueError("Too many splits for number of samples")

        indices = list(range(n_samples))
        for i in range(self.n_splits):
            start = (i + 1) * test_size
            end = (i + 2) * test_size if i < self.n_splits - 1 else n_samples
            val_idx = indices[start:end]
            train_end = max(0, start - self.embargo)
            train_idx = indices[:train_end]
            yield train_idx, val_idx
