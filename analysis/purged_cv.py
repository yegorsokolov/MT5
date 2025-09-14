"""Cross-validation utilities for time series.

This module provides :class:`PurgedTimeSeriesSplit`, a splitter similar to
:class:`sklearn.model_selection.TimeSeriesSplit` but with two key
enhancements:

* An *embargo* window where samples immediately preceding a validation fold
  are excluded from the training indices to reduce look-ahead bias.
* Optional *group* based exclusion so that samples sharing a group with the
  validation fold (e.g., the same asset symbol) are also removed from the
  training indices.
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

        Parameters
        ----------
        X : Sequence
            Data to split. Only its length is used.
        y : Sequence, optional
            Included for compatibility with scikit-learn's splitter API.
        groups : Sequence, optional
            Group labels for each sample. Any sample sharing a group with the
            validation fold is excluded from the training indices.

        Invariants
        ----------
        * Training indices lie strictly before the validation fold and are
          separated from it by at least ``embargo`` observations.
        * Samples sharing a group with the validation fold are excluded from
          the training indices.

        These constraints prevent information leakage when performing
        time-series cross-validation.
        """
        n_samples = len(X)
        if self.n_splits >= n_samples:
            raise ValueError("Cannot have number of splits >= number of samples")

        if groups is not None and len(groups) != n_samples:
            raise ValueError("groups must be the same length as X")

        test_size = n_samples // (self.n_splits + 1)
        if test_size == 0:
            raise ValueError("Too many splits for number of samples")

        indices = list(range(n_samples))
        for i in range(self.n_splits):
            start = (i + 1) * test_size
            end = (i + 2) * test_size if i < self.n_splits - 1 else n_samples
            val_idx = indices[start:end]
            val_groups = set()
            if groups is not None:
                val_groups = {groups[j] for j in val_idx}
            train_end = max(0, start - self.embargo)
            train_idx = [
                j
                for j in indices[:train_end]
                if groups is None or groups[j] not in val_groups
            ]
            yield train_idx, val_idx
