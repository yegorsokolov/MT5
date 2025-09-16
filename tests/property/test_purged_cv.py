import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hypothesis import assume, given, strategies as st

from analysis.purged_cv import PurgedTimeSeriesSplit


@given(
    n_splits=st.integers(min_value=1, max_value=5),
    embargo=st.integers(min_value=0, max_value=3),
    group_gap=st.integers(min_value=0, max_value=3),
    data=st.data(),
)
def test_purged_cv_invariants(n_splits, embargo, group_gap, data) -> None:
    """Training indices exclude embargoed and same-group samples."""
    n_samples = data.draw(
        st.integers(min_value=n_splits + 2, max_value=50), label="n_samples"
    )
    groups = data.draw(
        st.lists(
            st.integers(min_value=0, max_value=3),
            min_size=n_samples,
            max_size=n_samples,
        ),
        label="groups",
    )

    splitter = PurgedTimeSeriesSplit(
        n_splits=n_splits, embargo=embargo, group_gap=group_gap
    )
    X = list(range(n_samples))
    try:
        splits = list(splitter.split(X, groups=groups))
    except ValueError:
        assume(False)

    assert splits

    for train_idx, val_idx in splits:
        # training indices precede the validation fold and avoid the embargo window
        start = val_idx[0]
        embargo_range = set(range(max(0, start - embargo), start))
        assert embargo_range.isdisjoint(train_idx)

        # no sample in the training set shares a group with the validation set
        val_groups = {groups[j] for j in val_idx}
        if group_gap:
            end = val_idx[-1] + 1
            gap_start = max(0, start - group_gap)
            gap_stop = min(n_samples, end + group_gap)
            val_groups.update(groups[j] for j in range(gap_start, gap_stop))
        assert all(groups[j] not in val_groups for j in train_idx)

        # indices are sorted and unique to avoid accidental duplication
        assert train_idx == sorted(set(train_idx))
        assert val_idx == sorted(set(val_idx))

