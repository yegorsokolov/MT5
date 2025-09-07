"""Tests for PurgedTimeSeriesSplit."""

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.purged_cv import PurgedTimeSeriesSplit


def test_no_overlap_with_embargo() -> None:
    data = list(range(30))
    embargo = 3
    splitter = PurgedTimeSeriesSplit(n_splits=3, embargo=embargo)
    n = len(data)
    for train_idx, val_idx in splitter.split(data):
        # Train and validation indices should be disjoint
        assert set(train_idx).isdisjoint(val_idx)
        # Training indices must also avoid the embargo window around validation
        start, end = val_idx[0], val_idx[-1] + 1
        embargo_range = set(range(max(0, start - embargo), min(n, end + embargo)))
        assert embargo_range.isdisjoint(train_idx)
