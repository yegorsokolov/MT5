import numpy as np
import pandas as pd
import sys
import types
from pathlib import Path

sys.modules.setdefault(
    "mlflow",
    types.SimpleNamespace(
        set_tracking_uri=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        start_run=lambda *a, **k: types.SimpleNamespace(
            __enter__=lambda self: None, __exit__=lambda self, *exc: None
        ),
        log_dict=lambda *a, **k: None,
    ),
)
sys.modules.setdefault(
    "utils.environment", types.SimpleNamespace(ensure_environment=lambda: None)
)
sys.modules.setdefault("duckdb", types.SimpleNamespace(connect=lambda *a, **k: None))
sys.modules.setdefault("networkx", types.SimpleNamespace())
sys.modules.setdefault("requests", types.SimpleNamespace(get=lambda *a, **k: None))
sys.modules.setdefault(
    "dateutil",
    types.SimpleNamespace(parser=types.SimpleNamespace(parse=lambda *a, **k: None)),
)
sys.modules.setdefault(
    "pydantic",
    types.SimpleNamespace(
        ValidationError=Exception,
        BaseModel=object,
        Field=lambda *a, **k: None,
        ConfigDict=dict,
    ),
)
sys.modules.setdefault(
    "psutil",
    types.SimpleNamespace(
        cpu_percent=lambda *a, **k: 0,
        virtual_memory=lambda: types.SimpleNamespace(total=0),
        cpu_count=lambda *a, **k: 1,
    ),
)
ge_stub = types.ModuleType("great_expectations")
ge_core = types.ModuleType("great_expectations.core")
ge_stub.core = ge_core
sys.modules.setdefault("great_expectations", ge_stub)
sys.modules.setdefault("great_expectations.core", ge_core)
ge_expectation_suite = types.ModuleType("great_expectations.core.expectation_suite")
ge_expectation_suite.ExpectationSuite = object
sys.modules.setdefault(
    "great_expectations.core.expectation_suite", ge_expectation_suite
)
torch_stub = types.SimpleNamespace(
    cuda=types.SimpleNamespace(device_count=lambda: 0),
    nn=types.SimpleNamespace(Module=object),
    utils=types.SimpleNamespace(
        data=types.SimpleNamespace(DataLoader=None, TensorDataset=None)
    ),
)
sys.modules.setdefault("torch", torch_stub)
sys.modules.setdefault("torch.nn", torch_stub.nn)
sys.modules.setdefault("torch.utils", torch_stub.utils)
sys.modules.setdefault("torch.utils.data", torch_stub.utils.data)

features_pkg = types.ModuleType("features")
features_pkg.get_feature_pipeline = lambda: []
features_news = types.ModuleType("features.news")
features_news.add_economic_calendar_features = lambda df: df
features_news.add_news_sentiment_features = lambda df: df
features_cross = types.ModuleType("features.cross_asset")
features_cross.add_index_features = lambda df: df
features_cross.add_cross_asset_features = lambda df: df
features_validators = types.ModuleType("features.validators")
features_validators.validate_ge = lambda df, suite: df
sys.modules.setdefault("features", features_pkg)
sys.modules.setdefault("features.news", features_news)
sys.modules.setdefault("features.cross_asset", features_cross)
sys.modules.setdefault("features.validators", features_validators)
gplearn_genetic = types.ModuleType("gplearn.genetic")
gplearn_genetic.SymbolicTransformer = lambda *a, **k: None
sys.modules.setdefault("gplearn", types.ModuleType("gplearn"))
sys.modules.setdefault("gplearn.genetic", gplearn_genetic)
analysis_regime = types.ModuleType("analysis.regime_detection")
analysis_regime.periodic_reclassification = lambda df, step=500: df
sys.modules.setdefault("analysis.regime_detection", analysis_regime)
feature_gate_mod = types.ModuleType("analysis.feature_gate")
feature_gate_mod.select = lambda df, tier, regime_id, persist=False: (df, [])
sys.modules.setdefault("analysis.feature_gate", feature_gate_mod)
feature_evolver_mod = types.ModuleType("analysis.feature_evolver")
feature_evolver_mod.FeatureEvolver = lambda: types.SimpleNamespace(
    apply_stored_features=lambda df: df,
    apply_hypernet=lambda df, target_col=None: df,
    maybe_evolve=lambda df, **kw: df,
)
sys.modules.setdefault("analysis.feature_evolver", feature_evolver_mod)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import data.features as F
from analysis import anomaly_detector as AD
import utils


def _stub(df):
    return df


def test_anomalies_filtered(monkeypatch, caplog):
    # Patch heavy feature functions to no-ops
    monkeypatch.setattr(F, "get_feature_pipeline", lambda: [])
    monkeypatch.setattr(F, "add_news_sentiment_features", _stub)
    monkeypatch.setattr(F, "add_index_features", _stub)
    monkeypatch.setattr(F, "add_economic_calendar_features", _stub)
    monkeypatch.setattr(F, "add_cross_asset_features", _stub, raising=False)
    monkeypatch.setattr(F, "add_garch_volatility", _stub)
    monkeypatch.setattr(F, "add_cross_spectral_features", _stub)
    monkeypatch.setattr(F, "add_dtw_features", _stub)
    monkeypatch.setattr(F, "add_knowledge_graph_features", _stub)
    monkeypatch.setattr(F, "add_frequency_features", _stub)
    monkeypatch.setattr(F, "add_stl_features", _stub)
    monkeypatch.setattr(F, "add_fractal_features", _stub)
    monkeypatch.setattr(F, "add_factor_exposure_features", _stub)
    monkeypatch.setattr(F, "add_alt_features", _stub, raising=False)
    monkeypatch.setattr(F, "add_corporate_actions", _stub, raising=False)
    monkeypatch.setattr(F, "load_macro_features", _stub, raising=False)
    monkeypatch.setattr(
        F, "periodic_reclassification", lambda df, step=500: df.assign(market_regime=0)
    )
    fake_evolver = types.SimpleNamespace(
        apply_stored_features=lambda df: df,
        apply_hypernet=lambda df, target_col=None: df,
        maybe_evolve=lambda df, **kw: df,
    )
    monkeypatch.setattr(F, "FeatureEvolver", lambda: fake_evolver)
    monkeypatch.setattr(
        F.feature_gate, "select", lambda df, tier, regime_id, persist=False: (df, [])
    )
    monkeypatch.setattr(utils, "load_config", lambda: {})

    # Use a more aggressive contamination to simplify the test
    def detect(df, **kwargs):
        return AD.detect_anomalies(df, method="zscore", threshold=2.0)

    monkeypatch.setattr(F, "detect_anomalies", detect)

    n = 200
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=n, freq="T"),
            "Bid": np.random.normal(100, 1, n),
            "Ask": np.random.normal(100, 1, n) + 0.01,
            "Symbol": ["A"] * n,
        }
    )
    # Inject obvious anomalies
    df.loc[n - 10 :, ["Bid", "Ask"]] = 1000

    caplog.set_level("INFO")
    feats = F.make_features(df.copy())

    # Anomalies should be removed
    assert len(feats) < n
    assert feats["Bid"].max() < 1000
    assert "anomaly_rate" in caplog.text
