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
        if train_idx:
            assert max(train_idx) < val_idx[0]
        # Training indices must also avoid the embargo window around validation
        start, end = val_idx[0], val_idx[-1] + 1
        embargo_range = set(range(max(0, start - embargo), min(n, end + embargo)))
        assert embargo_range.isdisjoint(train_idx)


def test_group_exclusion() -> None:
    data = list(range(12))
    groups = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]
    splitter = PurgedTimeSeriesSplit(n_splits=3, embargo=0)
    for train_idx, val_idx in splitter.split(data, groups=groups):
        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        assert train_groups.isdisjoint(val_groups)


def test_group_gap_exclusion() -> None:
    data = list(range(20))
    groups = [0, 0, 1, 1, 2] * 4
    splitter = PurgedTimeSeriesSplit(n_splits=4, embargo=0, group_gap=2)
    n = len(data)
    for train_idx, val_idx in splitter.split(data, groups=groups):
        start, end = val_idx[0], val_idx[-1] + 1
        gap_groups = {
            groups[i]
            for i in range(max(0, start - 2), min(n, end + 2))
        }
        assert all(groups[i] not in gap_groups for i in train_idx)


def test_no_time_or_group_overlap() -> None:
    data = list(range(24))
    groups = [0] * 6 + [1] * 6 + [0] * 6 + [1] * 6
    splitter = PurgedTimeSeriesSplit(n_splits=4, embargo=2)
    n = len(data)
    for train_idx, val_idx in splitter.split(data, groups=groups):
        train_set = set(train_idx)
        val_set = set(val_idx)
        assert train_set.isdisjoint(val_set)
        start, end = val_idx[0], val_idx[-1] + 1
        embargo_range = set(range(max(0, start - 2), min(n, end + 2)))
        assert embargo_range.isdisjoint(train_set)
        train_groups = {groups[i] for i in train_idx}
        val_groups = {groups[i] for i in val_idx}
        assert train_groups.isdisjoint(val_groups)
