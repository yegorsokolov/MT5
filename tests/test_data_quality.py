import pandas as pd
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    "analysis.data_quality", Path(__file__).resolve().parents[1] / "analysis" / "data_quality.py"
)
data_quality = importlib.util.module_from_spec(spec)
sys.modules["analysis.data_quality"] = data_quality
spec.loader.exec_module(data_quality)
apply_quality_checks = data_quality.apply_quality_checks


def make_df_with_gap():
    return pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                [
                    "2024-01-01 00:00:00",
                    "2024-01-01 00:00:01",
                    "2024-01-01 00:00:04",
                ]
            ),
            "Bid": [1.0, 1.01, 1.04],
            "Ask": [1.1, 1.11, 1.14],
        }
    )


def test_gap_detection_and_interpolation():
    df = make_df_with_gap()
    clean, report = apply_quality_checks(df, max_gap="1s")
    assert report["gaps"] == 1
    assert len(clean) == 5
    assert clean["Bid"].isna().sum() == 0
    assert clean.loc[2, "Bid"] == pytest.approx(1.02, rel=1e-3)


def test_outlier_filters():
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=5, freq="s"),
            "Bid": [1, 1, 10, 1, 1],
            "Ask": [1.1, 1.1, 10.1, 1.1, 1.1],
        }
    )
    clean, report = apply_quality_checks(df, max_gap="1s", z_threshold=1.5, med_window=3)
    assert report["zscore"] > 0
    assert report["median"] >= 0
    assert clean.loc[2, "Bid"] == pytest.approx(1, rel=1e-3)
