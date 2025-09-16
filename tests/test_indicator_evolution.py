import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evolved_indicators", ROOT / "features" / "evolved_indicators.py"
)
evolved_indicators = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(evolved_indicators)


def _synthetic() -> pd.DataFrame:
    n = 100
    price = np.arange(n) + np.sin(np.arange(n) / 5)
    volume = np.linspace(1.0, 2.0, n)
    df = pd.DataFrame({"price": price, "volume": volume})
    target = (df["price"].diff() / df["volume"]).rolling(2).mean().shift(-1)
    df["target"] = target.fillna(0.0)
    return df


def test_deterministic_output():
    df = _synthetic()
    out1 = evolved_indicators.compute(df)
    out2 = evolved_indicators.compute(df)
    assert out1["evo_ind_0"].equals(out2["evo_ind_0"])


def test_indicator_improves_sharpe():
    df = _synthetic()
    out = evolved_indicators.compute(df)
    signal_ind = np.sign(out["evo_ind_0"])[:-1]
    signal_base = np.sign(df["price"].diff().fillna(0.0))[:-1]
    ret = df["target"][1:]
    sharpe_ind = (signal_ind * ret).mean() / (signal_ind * ret).std()
    sharpe_base = (signal_base * ret).mean() / (signal_base * ret).std()
    assert sharpe_ind > sharpe_base


def test_compute_handles_series_and_arrays(tmp_path):
    df = pd.DataFrame({"price": np.linspace(1.0, 2.0, 6)})
    path = tmp_path / "evolved.json"
    path.write_text(
        json.dumps(
            [
                {"name": "array_feature", "formula": "np.linspace(0, 1, len(df))"},
                {
                    "name": "series_feature",
                    "formula": "pd.Series(df['price']).rolling(3).mean().fillna(0.0)",
                },
            ]
        )
    )

    out = evolved_indicators.compute(df.copy(), path=path)
    assert "array_feature" in out.columns
    assert "series_feature" in out.columns
    expected_array = np.linspace(0, 1, len(df))
    np.testing.assert_allclose(out["array_feature"].to_numpy(), expected_array)
    assert out["series_feature"].index.equals(df.index)
