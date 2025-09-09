import pandas as pd
import numpy as np
import importlib.util
from pathlib import Path
import sys

# Load cross_asset module without importing the full package (avoids heavy deps)
_cross_asset_path = Path(__file__).resolve().parents[1] / "features" / "cross_asset.py"
spec = importlib.util.spec_from_file_location("cross_asset", _cross_asset_path)
cross_asset = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cross_asset)
add_cross_asset_features = cross_asset.add_cross_asset_features

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategies.pairs_baseline import PairsBaselineConfig, generate_signals


def build_df() -> pd.DataFrame:
    ts = pd.date_range("2020-01-01", periods=3, freq="D")
    a = pd.DataFrame({"Timestamp": ts, "Symbol": "A", "return": [0.01, 0.02, 0.03]})
    b = pd.DataFrame({"Timestamp": ts, "Symbol": "B", "return": [-0.01, -0.02, -0.01]})
    return pd.concat([a, b], ignore_index=True)


def test_relative_strength_and_signals():
    df = build_df()
    feat = add_cross_asset_features(df)

    # Relative strength should be +/-1 given symmetric returns
    ts0 = feat["Timestamp"].iloc[0]
    rs_a = feat[(feat["Timestamp"] == ts0) & (feat["Symbol"] == "A")]["rel_strength_A"].iloc[0]
    rs_b = feat[(feat["Timestamp"] == ts0) & (feat["Symbol"] == "B")]["rel_strength_B"].iloc[0]
    assert np.isclose(rs_a, 1.0)
    assert np.isclose(rs_b, -1.0)

    cfg = PairsBaselineConfig(universe=["A", "B"], hedge_ratio=1.0)
    out = generate_signals(feat, cfg)

    # A should always be long and B short with net zero exposure
    assert (out[out["Symbol"] == "A"]["signal"] == 1.0).all()
    assert (out[out["Symbol"] == "B"]["signal"] == -1.0).all()
    grouped = out.groupby("Timestamp")["signal"].sum()
    assert (grouped.round(6) == 0).all()
