import json
from pathlib import Path
import sys

import numpy as np

# Ensure real pandas and scipy are available for sklearn
sys.modules.pop("pandas", None)
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)
import pandas as pd  # type: ignore  # pylint: disable=unused-import
import scipy  # type: ignore  # pylint: disable=unused-import
import scipy.stats  # type: ignore  # pylint: disable=unused-import

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis import feature_gate


def _make_df(f2_correlated: bool) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    ret = rng.normal(size=200)
    y = (np.roll(ret, -1) > 0).astype(float)
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2024", periods=200, freq="T"),
            "return": ret,
            # f1 perfectly correlated with target
            "f1": y + rng.normal(scale=0.01, size=200),
            # f2 is either noise or also correlated
            "f2": (y + rng.normal(scale=0.01, size=200)) if f2_correlated else rng.normal(size=200),
            # heavy feature should always be removed on lite tier
            "heavy_feat": rng.normal(size=200),
            "market_regime": 0,
        }
    )
    return df


def test_feature_gate_persistence(tmp_path: Path):
    df = _make_df(f2_correlated=False)
    filtered1, selected1 = feature_gate.select(df, "lite", 0, store_dir=tmp_path)

    # heavy feature dropped and only f1 retained
    assert "heavy_feat" not in filtered1.columns
    assert selected1 == ["f1"]

    gate_file = tmp_path / "regime_0.json"
    assert json.loads(gate_file.read_text()) == selected1

    # make f2 informative but ensure gate uses persisted list
    df2 = _make_df(f2_correlated=True)
    filtered2, selected2 = feature_gate.select(df2, "lite", 0, store_dir=tmp_path)

    assert selected2 == selected1
    assert list(filtered1.columns) == list(filtered2.columns)
