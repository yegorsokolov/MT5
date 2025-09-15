import sys
import types
from pathlib import Path
import numpy as np
import pandas as pd

# Stub minimal data and feature submodules to satisfy imports in features.__init__
_data = types.ModuleType("data")
_data_features = types.ModuleType("data.features")
_data_features.make_features = lambda df, *a, **k: df
_data.features = _data_features
_data_expectations = types.ModuleType("data.expectations")
_data_expectations.validate_dataframe = lambda *a, **k: None
sys.modules.update({
    "data": _data,
    "data.features": _data_features,
    "data.expectations": _data_expectations,
})

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def _make_stub():
    def compute(df, *_, **__):
        return df
    return compute

for name in [
    "price",
    "news",
    "cross_asset",
    "orderbook",
    "order_flow",
    "microprice",
    "liquidity_exhaustion",
    "auto_indicator",
    "volume",
    "multi_timeframe",
    "supertrend",
    "keltner_squeeze",
    "adaptive_ma",
    "kalman_ma",
    "regime",
    "macd",
    "ram",
    "cointegration",
    "vwap",
    "baseline_signal",
    "divergence",
    "evolved_indicators",
    "evolved_symbols",
]:
    mod = types.ModuleType(f"features.{name}")
    mod.compute = _make_stub()
    sys.modules[f"features.{name}"] = mod

import features
from analysis import cross_spectral


def test_cross_spectral_registration_and_config(monkeypatch):
    monkeypatch.setattr(features, "load_config", lambda: {"features": ["cross_spectral"]})
    monkeypatch.setattr(
        cross_spectral, "load_config", lambda: {"cross_spectral": {"window": 16}}
    )

    class Caps:
        cpus = 8
        memory_gb = 16.0
        has_gpu = False
        gpu_count = 0

    monkeypatch.setattr(features.monitor, "capabilities", Caps())
    features._update_status()
    pipeline = features.get_feature_pipeline()
    assert cross_spectral.compute in pipeline

    n = 20
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=n, freq="D").tolist() * 2,
            "Symbol": ["AAA"] * n + ["BBB"] * n,
            "Close": np.random.rand(2 * n),
        }
    )
    out = pipeline[0](df)
    assert "coh_BBB" in out.columns
    coh = out[out["Symbol"] == "AAA"]["coh_BBB"]
    assert coh.notna().sum() == n - 15
