from __future__ import annotations

import sys
import types
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

data_stub = types.ModuleType("data")
data_stub.__path__ = [str(ROOT / "data")]
sys.modules.setdefault("data", data_stub)

from data.macro_features import load_macro_features


def test_load_macro_features_respects_data_delay():
    df = pd.DataFrame(
        {
            "Timestamp": [
                pd.Timestamp("2020-01-02 09:00", tz="UTC"),
                pd.Timestamp("2020-01-02 15:00", tz="UTC"),
            ],
            "Region": ["US", "US"],
            "Symbol": ["AAA", "AAA"],
        }
    )
    macro_df = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2020-01-01", tz="UTC"),
                pd.Timestamp("2020-01-02", tz="UTC"),
            ],
            "Region": ["US", "US"],
            "gdp": [1.0, 2.0],
        }
    )

    out = load_macro_features(df, macro_df=macro_df, data_delay=pd.Timedelta(hours=12))
    assert list(out["macro_gdp"]) == [pytest.approx(1.0), pytest.approx(2.0)]


def test_load_macro_features_supports_column_specific_delay():
    df = pd.DataFrame(
        {
            "Timestamp": [
                pd.Timestamp("2020-01-02 09:00", tz="UTC"),
                pd.Timestamp("2020-01-02 09:30", tz="UTC"),
                pd.Timestamp("2020-01-02 11:30", tz="UTC"),
            ],
            "Region": ["US", "US", "US"],
            "Symbol": ["AAA", "AAA", "AAA"],
        }
    )
    macro_df = pd.DataFrame(
        {
            "Date": [
                pd.Timestamp("2020-01-01 09:00", tz="UTC"),
                pd.Timestamp("2020-01-02 09:00", tz="UTC"),
            ],
            "Region": ["US", "US"],
            "gdp": [1.0, 2.0],
            "cpi": [4.0, 5.0],
        }
    )

    delays = {"default": pd.Timedelta(hours=1), "gdp": pd.Timedelta(0), "cpi": pd.Timedelta(hours=2)}
    out = load_macro_features(df, macro_df=macro_df, data_delay=delays)

    assert list(out["macro_gdp"]) == [pytest.approx(2.0), pytest.approx(2.0), pytest.approx(2.0)]
    assert list(out["macro_cpi"]) == [pytest.approx(4.0), pytest.approx(4.0), pytest.approx(5.0)]
