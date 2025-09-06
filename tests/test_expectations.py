import importlib.util
import sys
from pathlib import Path

import pandas as pd

spec = importlib.util.spec_from_file_location(
    "data.expectations",
    Path(__file__).resolve().parents[1] / "data" / "expectations" / "__init__.py",
)
expectations = importlib.util.module_from_spec(spec)
sys.modules["data.expectations"] = expectations
spec.loader.exec_module(expectations)
validate_dataframe = expectations.validate_dataframe


def test_raw_ticks_expectations(tmp_path):
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
            "Bid": [1.0, 1.1, 1.2],
            "Ask": [1.1, 1.2, 1.3],
        }
    )
    assert validate_dataframe(df, "raw_ticks", quarantine=False) is None


def test_engineered_features_expectations():
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024-01-01", periods=3, freq="s"),
            "feature": [0.1, 0.2, 0.3],
        }
    )
    assert validate_dataframe(df, "engineered_features", quarantine=False) is None


def test_labels_expectations():
    df = pd.DataFrame({"Label": [1, 0, -1]})
    assert validate_dataframe(df, "labels", quarantine=False) is None
