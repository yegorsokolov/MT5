import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

FEATURES_PATH = Path(__file__).resolve().parents[1] / "features" / "cross_asset.py"
spec = importlib.util.spec_from_file_location("cross_asset", FEATURES_PATH)
cross_asset = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(cross_asset)
add_cross_asset_features = cross_asset.add_cross_asset_features


def _sample_df():
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=5, freq="D").tolist() * 2,
            "Symbol": ["AAA"] * 5 + ["BBB"] * 5,
            "mid": [1, 2, 3, 4, 5, 2, 1, 2, 3, 4],
        }
    )
    df["return"] = df.groupby("Symbol")["mid"].pct_change()
    return df


def test_cross_asset_features_creation():
    df = _sample_df()
    out = add_cross_asset_features(df, window=3)

    for col in [
        "corr_AAA_BBB",
        "corr_BBB_AAA",
        "relret_AAA_BBB",
        "relret_BBB_AAA",
    ]:
        assert col in out.columns

    assert out.shape[0] == df.shape[0]
    assert out.shape[1] == df.shape[1] + 4

    wide = df.pivot(index="Timestamp", columns="Symbol", values="return")
    expected_corr = wide["AAA"].rolling(3).corr(wide["BBB"]).iloc[-1]
    expected_ratio = (wide["AAA"] / wide["BBB"]).iloc[-1]

    last_row = out[
        (out["Symbol"] == "AAA") & (out["Timestamp"] == df["Timestamp"].max())
    ]
    assert last_row["corr_AAA_BBB"].iloc[0] == pytest.approx(expected_corr)
    assert last_row["relret_AAA_BBB"].iloc[0] == pytest.approx(expected_ratio)
