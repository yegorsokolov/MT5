import importlib.util
import sys
import types
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA_PATH = ROOT / "data"

# Stub minimal features and analytics packages required by data.features
features_stub = types.ModuleType("features")
features_stub.get_feature_pipeline = lambda: []
features_news = types.ModuleType("features.news")
features_news.add_economic_calendar_features = lambda df: df
features_news.add_news_sentiment_features = lambda df: df
features_cross = types.ModuleType("features.cross_asset")
features_cross.add_index_features = lambda df: df
features_cross.add_cross_asset_features = lambda df: df
sys.modules["features"] = features_stub
sys.modules["features.news"] = features_news
sys.modules["features.cross_asset"] = features_cross

analytics_stub = types.ModuleType("analytics.metrics_store")
analytics_stub.record_metric = lambda *a, **k: None
sys.modules["analytics.metrics_store"] = analytics_stub

feature_gate_stub = types.ModuleType("analysis.feature_gate")
feature_gate_stub.select = lambda df, tier, regime, persist=False: (df, [])
analysis_pkg = types.ModuleType("analysis")
analysis_pkg.feature_gate = feature_gate_stub
analysis_pkg.cross_spectral = types.ModuleType("analysis.cross_spectral")
analysis_pkg.cross_spectral.compute = lambda df, window=64: df
analysis_pkg.cross_spectral.REQUIREMENTS = types.SimpleNamespace(
    cpus=0, memory_gb=0.0, has_gpu=False
)
analysis_pkg.data_lineage = types.ModuleType("analysis.data_lineage")
analysis_pkg.data_lineage.log_lineage = lambda *a, **k: None
analysis_pkg.fractal_features = types.ModuleType("analysis.fractal_features")
analysis_pkg.fractal_features.rolling_fractal_features = (
    lambda s: pd.DataFrame({"hurst": s, "fractal_dim": s})
)
analysis_pkg.frequency_features = types.ModuleType("analysis.frequency_features")
analysis_pkg.frequency_features.spectral_features = (
    lambda s: pd.DataFrame({"spec_energy": s})
)
analysis_pkg.frequency_features.wavelet_energy = (
    lambda s: pd.DataFrame({"wavelet_energy": s})
)
analysis_pkg.garch_vol = types.ModuleType("analysis.garch_vol")
analysis_pkg.garch_vol.garch_volatility = lambda s: s
sys.modules["analysis"] = analysis_pkg
sys.modules["analysis.feature_gate"] = feature_gate_stub
sys.modules["analysis.cross_spectral"] = analysis_pkg.cross_spectral
sys.modules["analysis.data_lineage"] = analysis_pkg.data_lineage
sys.modules["analysis.fractal_features"] = analysis_pkg.fractal_features
sys.modules["analysis.frequency_features"] = analysis_pkg.frequency_features
sys.modules["analysis.garch_vol"] = analysis_pkg.garch_vol

package = types.ModuleType("data")
package.__path__ = [str(DATA_PATH)]
sys.modules["data"] = package
spec = importlib.util.spec_from_file_location("data.features", DATA_PATH / "features.py")
features = importlib.util.module_from_spec(spec)
sys.modules["data.features"] = features
spec.loader.exec_module(features)


def test_factor_exposures_merged(tmp_path, monkeypatch):
    exp = pd.DataFrame(
        {
            "factor_1": [0.1, 0.2],
            "factor_2": [0.3, 0.4],
        },
        index=["AAA", "BBB"],
    )
    exp.to_csv(tmp_path / "exposures_20200101.csv")

    monkeypatch.setattr(features, "FACTOR_EXPOSURE_DIR", tmp_path)

    df = pd.DataFrame({"Symbol": ["AAA", "BBB", "CCC"]})
    merged = features.add_factor_exposure_features(df)

    assert set(c for c in merged.columns if c.startswith("factor_")) == {"factor_1", "factor_2"}
    a = merged.loc[merged.Symbol == "AAA", "factor_1"].iloc[0]
    b = merged.loc[merged.Symbol == "BBB", "factor_2"].iloc[0]
    c = merged.loc[merged.Symbol == "CCC", "factor_1"].iloc[0]
    assert a == 0.1 and b == 0.4 and c == 0.0
