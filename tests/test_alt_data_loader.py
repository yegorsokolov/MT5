import sys
from pathlib import Path
import importlib.util

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _prepare_features(monkeypatch):
    import types
    import numpy as np

    stub_utils = types.ModuleType("utils")
    stub_utils.__path__ = []
    stub_utils.load_config = lambda: types.SimpleNamespace(
        use_feature_cache=False,
        use_atr=False,
        use_donchian=False,
        use_dask=False,
        features=types.SimpleNamespace(latency_threshold=0.0),
    )

    class _Mon:
        def start(self):
            pass

        capability_tier = "lite"

        class capabilities:
            cpus = 1
            memory_gb = 1
            has_gpu = False

            @staticmethod
            def capability_tier():
                return "lite"

    class _ResourceCaps:
        def __init__(self, cpus=1, memory_gb=1, has_gpu=False, gpu_count=0):
            self.cpus = cpus
            self.memory_gb = memory_gb
            self.has_gpu = has_gpu
            self.gpu_count = gpu_count

    stub_utils.resource_monitor = types.SimpleNamespace(
        monitor=_Mon(), ResourceCapabilities=_ResourceCaps
    )
    stub_utils.data_backend = types.SimpleNamespace(get_dataframe_module=lambda: pd)
    sys.modules["utils"] = stub_utils
    sys.modules["utils.resource_monitor"] = stub_utils.resource_monitor
    sys.modules["utils.data_backend"] = stub_utils.data_backend

    stub_features_pkg = types.ModuleType("features")
    stub_features_pkg.__path__ = []
    stub_features_pkg.get_feature_pipeline = lambda: []
    features_news = types.ModuleType("features.news")
    features_news.add_economic_calendar_features = lambda df: df
    features_news.add_news_sentiment_features = lambda df: df
    features_cross = types.ModuleType("features.cross_asset")
    features_cross.add_index_features = lambda df: df
    features_cross.add_cross_asset_features = lambda df: df
    stub_features_pkg.news = features_news
    stub_features_pkg.cross_asset = features_cross
    sys.modules["features"] = stub_features_pkg
    sys.modules["features.news"] = features_news
    sys.modules["features.cross_asset"] = features_cross
    features_validators = types.ModuleType("features.validators")
    features_validators.validate_ge = lambda df, *a, **k: df
    sys.modules["features.validators"] = features_validators

    alt_loader = types.ModuleType("data.alt_data_loader")
    alt_loader.load_alt_data = lambda symbols: pd.DataFrame()
    sys.modules["data.alt_data_loader"] = alt_loader

    sklearn_stub = types.SimpleNamespace(
        decomposition=types.SimpleNamespace(PCA=lambda *a, **k: None),
        feature_selection=types.SimpleNamespace(
            mutual_info_classif=lambda *a, **k: np.zeros(a[0].shape[1] if a else 0)
        ),
        cluster=types.SimpleNamespace(KMeans=lambda *a, **k: None),
    )
    sys.modules["sklearn"] = sklearn_stub
    sys.modules["sklearn.decomposition"] = sklearn_stub.decomposition
    sys.modules["sklearn.feature_selection"] = sklearn_stub.feature_selection
    sys.modules["sklearn.cluster"] = sklearn_stub.cluster

    stub_events = types.ModuleType("data.events")
    stub_events.get_events = lambda *a, **k: pd.DataFrame()
    sys.modules["data.events"] = stub_events

    stub_history = types.ModuleType("data.history")
    for name in [
        "load_history_from_urls",
        "load_history_mt5",
        "load_history_config",
        "load_history",
        "load_history_parquet",
        "load_history_memmap",
        "save_history_parquet",
        "load_multiple_histories",
    ]:
        setattr(stub_history, name, lambda *a, **k: pd.DataFrame())
    sys.modules["data.history"] = stub_history

    stub_expectations = types.ModuleType("data.expectations")
    stub_expectations.validate_dataframe = lambda df, name: df
    sys.modules["data.expectations"] = stub_expectations

    stub_graph = types.ModuleType("data.graph_builder")
    stub_graph.build_correlation_graph = lambda *a, **k: None
    stub_graph.build_rolling_adjacency = lambda *a, **k: None
    sys.modules["data.graph_builder"] = stub_graph

    stub_kg = types.ModuleType("analysis.knowledge_graph")
    stub_kg.load_knowledge_graph = lambda *a, **k: None
    stub_kg.risk_score = lambda *a, **k: 0.0
    stub_kg.opportunity_score = lambda *a, **k: 0.0
    sys.modules["analysis.knowledge_graph"] = stub_kg

    stub_freq = types.ModuleType("analysis.frequency_features")
    stub_freq.spectral_features = lambda df: df
    stub_freq.wavelet_energy = lambda df: df
    stub_freq.stl_decompose = lambda df: (df, df)
    sys.modules["analysis.frequency_features"] = stub_freq

    stub_fractal = types.ModuleType("analysis.fractal_features")
    stub_fractal.rolling_fractal_features = lambda df: df
    sys.modules["analysis.fractal_features"] = stub_fractal

    stub_cross = types.ModuleType("analysis.cross_spectral")
    stub_cross.compute = lambda df, window=None: df
    stub_cross.REQUIREMENTS = stub_utils.resource_monitor.ResourceCapabilities()
    sys.modules["analysis.cross_spectral"] = stub_cross

    stub_garch = types.ModuleType("analysis.garch_vol")
    stub_garch.garch_volatility = lambda df: df
    sys.modules["analysis.garch_vol"] = stub_garch

    sys.modules["analysis.data_lineage"] = types.SimpleNamespace(
        log_lineage=lambda *a, **k: None
    )

    sys.modules["analysis.regime_detection"] = types.SimpleNamespace(
        periodic_reclassification=lambda df, **k: df
    )

    stub_evolver = types.ModuleType("analysis.feature_evolver")

    class _FE:
        def __init__(self, *a, **k):
            pass

    stub_evolver.FeatureEvolver = _FE
    sys.modules["analysis.feature_evolver"] = stub_evolver

    sys.path.append(str(ROOT))
    sys.modules.pop("data", None)
    sys.modules.pop("data.features", None)
    from data import features as feature_mod  # type: ignore

    return feature_mod


def _load_validators():
    import types

    class _Schema:
        def validate(self, df):
            return df

    module = types.SimpleNamespace(FEATURE_SCHEMA=_Schema())
    sys.modules["data.validators"] = module
    return module


def _load_alt_loader():
    spec = importlib.util.spec_from_file_location(
        "data.alt_data_loader", ROOT / "data" / "alt_data_loader.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    sys.modules["data.alt_data_loader"] = module
    return module


def _write_csv(path: Path, header, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=header).to_csv(path, index=False)


def test_alt_data_loader_alignment(tmp_path, monkeypatch):
    feature_mod = _prepare_features(monkeypatch)
    alt_loader = _load_alt_loader()
    FEATURE_SCHEMA = _load_validators().FEATURE_SCHEMA

    _write_csv(
        tmp_path / "dataset" / "shipping" / "AAA.csv",
        ["Date", "shipping_metric"],
        [["2020-01-01", 1.2]],
    )
    _write_csv(
        tmp_path / "dataset" / "retail" / "AAA.csv",
        ["Date", "retail_sales"],
        [["2020-01-01", 10]],
    )
    _write_csv(
        tmp_path / "dataset" / "weather" / "AAA.csv",
        ["Date", "temperature"],
        [["2020-01-01", 25.0]],
    )
    _write_csv(
        tmp_path / "dataset" / "macro" / "AAA.csv",
        ["Date", "gdp", "cpi", "interest_rate"],
        [["2020-01-01", 1.0, 2.0, 0.5]],
    )
    _write_csv(
        tmp_path / "dataset" / "news" / "AAA.csv",
        ["Date", "news_sentiment"],
        [["2020-01-01", 0.8]],
    )

    monkeypatch.chdir(tmp_path)

    alt = alt_loader.load_alt_data(["AAA"])
    assert set(
        [
            "shipping_metric",
            "retail_sales",
            "temperature",
            "gdp",
            "cpi",
            "interest_rate",
            "news_sentiment",
        ]
    ).issubset(alt.columns)
    assert (alt["Symbol"] == "AAA").all()
    assert alt["Date"].dtype.tz is not None

    base = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2020-01-02"], utc=True),
            "Symbol": ["AAA"],
            "return": [0.0],
            "ma_5": [0.0],
            "ma_10": [0.0],
            "ma_30": [0.0],
            "ma_60": [0.0],
            "volatility_30": [0.0],
            "rsi_14": [0.0],
            "market_regime": [0],
        }
    )

    out = feature_mod.add_alt_features(base)

    assert out.loc[0, "shipping_metric"] == 1.2
    assert out.loc[0, "retail_sales"] == 10
    assert out.loc[0, "temperature"] == 25.0
    assert out.loc[0, "gdp"] == 1.0
    assert out.loc[0, "cpi"] == 2.0
    assert out.loc[0, "interest_rate"] == 0.5
    assert out.loc[0, "news_sentiment"] == 0.8
    FEATURE_SCHEMA.validate(out)
