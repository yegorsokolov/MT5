import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hypothesis import given, strategies as st

from analysis.purged_cv import PurgedTimeSeriesSplit


@given(
    n_splits=st.integers(min_value=1, max_value=5),
    embargo=st.integers(min_value=0, max_value=3),
    data=st.data(),
)
def test_purged_cv_invariants(n_splits, embargo, data) -> None:
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

    splitter = PurgedTimeSeriesSplit(n_splits=n_splits, embargo=embargo)
    X = list(range(n_samples))
    for train_idx, val_idx in splitter.split(X, groups=groups):
        # training indices precede the validation fold and avoid the embargo window
        start = val_idx[0]
        assert all(j < start - embargo for j in train_idx)

        # no sample in the training set shares a group with the validation set
        val_groups = {groups[j] for j in val_idx}
        assert all(groups[j] not in val_groups for j in train_idx)

