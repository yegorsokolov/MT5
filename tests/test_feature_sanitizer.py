from math import inf, isclose
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from training.preprocessing import FeatureSanitizer


def test_feature_sanitizer_handles_missing_and_new_columns():
    df = pd.DataFrame({
        "a": [1.0, np.nan, inf],
        "b": ["1", "2", None],
    })
    sanitizer = FeatureSanitizer()
    transformed = sanitizer.fit_transform(df)
    assert list(transformed.columns) == ["a", "b"]
    assert not transformed.isna().any().any()

    new_df = pd.DataFrame({"b": ["3", "bad"], "a": [np.nan, 5.0]})
    new_transformed = sanitizer.transform(new_df)
    assert list(new_transformed.columns) == ["a", "b"]
    assert not new_transformed.isna().any().any()
    assert isclose(new_transformed.iloc[0, 0], sanitizer.fill_values_["a"])


def test_feature_sanitizer_state_roundtrip():
    df = pd.DataFrame({"x": [np.nan, 1.0, 2.0], "y": [5.0, 6.0, 7.0]})
    sanitizer = FeatureSanitizer(fill_method="zero")
    sanitizer.fit(df)
    state = sanitizer.state_dict()

    clone = FeatureSanitizer()
    clone.load_state_dict(state)
    result = clone.transform(pd.DataFrame({"x": [inf], "y": [np.nan]}))

    assert not result.isna().any().any()
    assert clone.fill_values_ == sanitizer.fill_values_
