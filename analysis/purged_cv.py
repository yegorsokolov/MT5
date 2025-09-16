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
    """Time series cross-validator with embargo and group awareness.

    Parameters
    ----------
    n_splits:
        Number of validation folds to generate.
    embargo:
        Number of observations immediately preceding a validation window to
        drop from the training set.  This is the *purging* step described by
        López de Prado.
    test_size:
        Explicit validation window size.  When omitted, the window is inferred
        from ``len(X) // (n_splits + 1)`` similar to scikit-learn's
        ``TimeSeriesSplit``.
    min_train_size:
        Optional lower bound on the amount of training data required before a
        validation fold.  When provided, folds violating the constraint raise a
        :class:`ValueError`.
    group_gap:
        Additional samples on either side of the validation fold whose group
        labels should also be excluded from the training indices.  This mirrors
        :class:`sklearn.model_selection.GroupTimeSeriesSplit` semantics and
        prevents leakage when groups overlap in time.
    """

    def __init__(
        self,
        n_splits: int = 5,
        *,
        embargo: int = 0,
        test_size: int | None = None,
        min_train_size: int | None = None,
        group_gap: int = 0,
    ) -> None:
        if n_splits < 1:
            raise ValueError("n_splits must be at least 1")
        if embargo < 0:
            raise ValueError("embargo must be non-negative")
        if test_size is not None and test_size < 1:
            raise ValueError("test_size must be at least 1")
        if min_train_size is not None and min_train_size < 1:
            raise ValueError("min_train_size must be at least 1")
        if group_gap < 0:
            raise ValueError("group_gap must be non-negative")

        self.n_splits = int(n_splits)
        self.embargo = int(embargo)
        self.test_size = int(test_size) if test_size is not None else None
        self.min_train_size = (
            int(min_train_size) if min_train_size is not None else None
        )
        self.group_gap = int(group_gap)

    def get_n_splits(self, X=None, y=None, groups=None) -> int:  # noqa: D401 - sklearn parity
        """Return the number of splits."""

        return self.n_splits

    def split(
        self,
        X: Sequence,
        y: Sequence | None = None,
        groups: Sequence | None = None,
    ) -> Iterator[tuple[list[int], list[int]]]:
        """Generate indices to split ``X`` into train and validation sets."""

        del y  # Unused but included for scikit-learn compatibility
        n_samples = len(X)
        if self.n_splits >= n_samples:
            raise ValueError("Cannot have number of splits >= number of samples")

        if groups is not None and len(groups) != n_samples:
            raise ValueError("groups must be the same length as X")

        test_size = self.test_size or n_samples // (self.n_splits + 1)
        if test_size <= 0:
            raise ValueError("Too many splits for number of samples")

        min_train = self.min_train_size
        start_offset = n_samples - test_size * self.n_splits
        if start_offset <= 0:
            raise ValueError("Not enough samples to create the requested splits")

        test_starts = range(start_offset, n_samples, test_size)
        for fold_idx, start in enumerate(test_starts):
            stop = start + test_size
            if fold_idx == self.n_splits - 1 or stop > n_samples:
                stop = n_samples

            val_idx = list(range(start, stop))
            train_end = max(0, start - self.embargo)
            if min_train is not None and train_end < min_train:
                raise ValueError(
                    "Not enough training data available before validation window"
                )

            exclusion_groups = set()
            if groups is not None:
                exclusion_groups = {groups[i] for i in val_idx}
                if self.group_gap:
                    gap_start = max(0, start - self.group_gap)
                    gap_stop = min(n_samples, stop + self.group_gap)
                    exclusion_groups.update(groups[i] for i in range(gap_start, gap_stop))

            train_idx = list(range(train_end))
            if groups is not None and exclusion_groups:
                train_idx = [
                    idx for idx in train_idx if groups[idx] not in exclusion_groups
                ]

            yield train_idx, val_idx

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        params = (
            f"n_splits={self.n_splits}",
            f"embargo={self.embargo}",
            f"test_size={self.test_size}",
            f"min_train_size={self.min_train_size}",
            f"group_gap={self.group_gap}",
        )
        return f"PurgedTimeSeriesSplit({', '.join(params)})"
