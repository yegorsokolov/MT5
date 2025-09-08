import sys
from pathlib import Path
import types
import importlib.machinery

import numpy as np
import pandas as pd

# Stub heavy optional dependencies before importing the project modules
ge_mod = types.ModuleType("great_expectations")
ge_mod.__spec__ = importlib.machinery.ModuleSpec("great_expectations", loader=None)

class _DummyResult(dict):
    def to_json_dict(self):  # pragma: no cover - simple stub
        return {}


class _DummyDataset:
    def __init__(self, df):
        self.df = df

    def validate(self, expectation_suite=None):  # pragma: no cover - simple stub
        return _DummyResult(success=True)


ge_mod.dataset = types.SimpleNamespace(PandasDataset=_DummyDataset)
ge_core_mod = types.ModuleType("great_expectations.core")
ge_suite_mod = types.ModuleType("great_expectations.core.expectation_suite")
ge_suite_mod.ExpectationSuite = object
ge_mod.core = ge_core_mod
sys.modules.setdefault("great_expectations", ge_mod)
sys.modules.setdefault("great_expectations.core", ge_core_mod)
sys.modules.setdefault("great_expectations.core.expectation_suite", ge_suite_mod)

# Stub analysis helpers to avoid heavy dependencies
feature_gate_mod = types.ModuleType("analysis.feature_gate")
feature_gate_mod.select = lambda df, tier, regime_id, persist=False: (df, [])
data_lineage_mod = types.ModuleType("analysis.data_lineage")
data_lineage_mod.log_lineage = lambda *a, **k: None
fractal_mod = types.ModuleType("analysis.fractal_features")
fractal_mod.rolling_fractal_features = lambda s: pd.DataFrame(
    {"hurst": pd.Series(0.0, index=s.index), "fractal_dim": pd.Series(0.0, index=s.index)}
)
cross_spec_mod = types.ModuleType("analysis.cross_spectral")
cross_spec_mod.compute = lambda df, window=64: df
cross_spec_mod.REQUIREMENTS = types.SimpleNamespace(cpus=1, memory_gb=0.0, has_gpu=False)
sys.modules.setdefault("analysis.feature_gate", feature_gate_mod)
sys.modules.setdefault("analysis.data_lineage", data_lineage_mod)
sys.modules.setdefault("analysis.fractal_features", fractal_mod)
sys.modules.setdefault("analysis.cross_spectral", cross_spec_mod)

# Ensure project root on path for package imports
root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

# Load data.features without executing heavy data.__init__
data_pkg = types.ModuleType("data")
data_pkg.__path__ = [str(root / "data")]
sys.modules.setdefault("data", data_pkg)
spec = importlib.util.spec_from_file_location(
    "data.features", root / "data" / "features.py"
)
features_mod = importlib.util.module_from_spec(spec)
sys.modules["data.features"] = features_mod
spec.loader.exec_module(features_mod)
add_garch_volatility = features_mod.add_garch_volatility


def test_garch_fallback_matches_rolling_std(monkeypatch):
    rng = pd.date_range("2020", periods=100, freq="D")
    returns = pd.Series(np.random.randn(100), index=rng)
    df = pd.DataFrame({"return": returns})

    # Simulate missing ``arch`` library to trigger fallback path
    monkeypatch.setitem(sys.modules, "arch", None)

    out = add_garch_volatility(df)
    expected = returns.rolling(30).std()
    assert "garch_vol" in out.columns
    pd.testing.assert_series_equal(out["garch_vol"], expected, check_names=False)

