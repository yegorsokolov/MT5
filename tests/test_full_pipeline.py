import numpy as np
import contextlib
import importlib.machinery
import types
import sys
from pathlib import Path

# ensure repository root on path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Minimal stubs for optional dependencies used during import
yaml_mod = types.ModuleType("yaml")
yaml_mod.safe_load = lambda *a, **k: {}
yaml_mod.safe_dump = lambda *a, **k: ""
yaml_mod.__spec__ = importlib.machinery.ModuleSpec("yaml", loader=None)
sys.modules.setdefault("yaml", yaml_mod)

mlflow_mod = types.ModuleType("mlflow")
mlflow_mod.set_tracking_uri = lambda *a, **k: None
mlflow_mod.set_experiment = lambda *a, **k: None
mlflow_mod.start_run = lambda *a, **k: contextlib.nullcontext()
mlflow_mod.log_dict = lambda *a, **k: None
mlflow_mod.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
sys.modules.setdefault("mlflow", mlflow_mod)

prom_mod = types.ModuleType("prometheus_client")
prom_mod.Counter = lambda *a, **k: None
prom_mod.Gauge = lambda *a, **k: None
prom_mod.generate_latest = lambda: b""
prom_mod.CONTENT_TYPE_LATEST = "text/plain"
prom_mod.__spec__ = importlib.machinery.ModuleSpec("prometheus_client", loader=None)
sys.modules.setdefault("prometheus_client", prom_mod)

metrics_mod = types.ModuleType("analytics.metrics_store")
metrics_mod.record_metric = lambda *a, **k: None
metrics_mod.model_cache_hit = lambda: None
metrics_mod.model_unload = lambda: None
metrics_mod.__spec__ = importlib.machinery.ModuleSpec("analytics.metrics_store", loader=None)
sys.modules.setdefault("analytics.metrics_store", metrics_mod)

pydantic_mod = types.ModuleType("pydantic")
pydantic_mod.BaseModel = object
pydantic_mod.ValidationError = Exception
pydantic_mod.Field = lambda *a, **k: None
pydantic_mod.ConfigDict = dict
pydantic_mod.__spec__ = importlib.machinery.ModuleSpec("pydantic", loader=None)
sys.modules.setdefault("pydantic", pydantic_mod)

filelock_mod = types.ModuleType("filelock")
class _FileLock:
    def __init__(self, *a, **k):
        pass

    def acquire(self, *a, **k):
        return True

    def release(self):
        pass

filelock_mod.FileLock = _FileLock
filelock_mod.__spec__ = importlib.machinery.ModuleSpec("filelock", loader=None)
sys.modules.setdefault("filelock", filelock_mod)

psutil_mod = types.ModuleType("psutil")
psutil_mod.Process = lambda *a, **k: None
psutil_mod.virtual_memory = lambda: types.SimpleNamespace(total=0, available=0)
psutil_mod.cpu_count = lambda *a, **k: 1
psutil_mod.cpu_percent = lambda *a, **k: 0.0
psutil_mod.__spec__ = importlib.machinery.ModuleSpec("psutil", loader=None)
sys.modules.setdefault("psutil", psutil_mod)

ge_mod = types.ModuleType("great_expectations")
ge_dataset_mod = types.ModuleType("great_expectations.dataset")
ge_dataset_mod.PandasDataset = object
ge_expectation_suite_mod = types.ModuleType(
    "great_expectations.core.expectation_suite"
)
class _ExpectationSuite:
    pass

ge_expectation_suite_mod.ExpectationSuite = _ExpectationSuite
sys.modules.setdefault("great_expectations", ge_mod)
sys.modules.setdefault("great_expectations.dataset", ge_dataset_mod)
sys.modules.setdefault("great_expectations.core", types.ModuleType("great_expectations.core"))
sys.modules.setdefault(
    "great_expectations.core.expectation_suite", ge_expectation_suite_mod
)

nx_mod = types.ModuleType("networkx")
nx_mod.Graph = object
nx_mod.DiGraph = object
sys.modules.setdefault("networkx", nx_mod)

feature_gate_mod = types.ModuleType("analysis.feature_gate")
feature_gate_mod.select = lambda df, tier, regime_id, persist=False: (df, [])
sys.modules.setdefault("analysis.feature_gate", feature_gate_mod)

pywt_mod = types.ModuleType("pywt")
pywt_mod.Wavelet = object
pywt_mod.wavedec = lambda *a, **k: [0]
pywt_mod.__spec__ = importlib.machinery.ModuleSpec("pywt", loader=None)
sys.modules.setdefault("pywt", pywt_mod)

crypto_mod = types.ModuleType("cryptography")
hazmat_mod = types.ModuleType("cryptography.hazmat")
primitives_mod = types.ModuleType("cryptography.hazmat.primitives")
ciphers_mod = types.ModuleType("cryptography.hazmat.primitives.ciphers")
aead_mod = types.ModuleType(
    "cryptography.hazmat.primitives.ciphers.aead"
)

class _AESGCM:
    def __init__(self, *a, **k):
        pass

    def encrypt(self, *a, **k):
        return b""

    def decrypt(self, *a, **k):
        return b""

aead_mod.AESGCM = _AESGCM
sys.modules.setdefault("cryptography", crypto_mod)
sys.modules.setdefault("cryptography.hazmat", hazmat_mod)
sys.modules.setdefault("cryptography.hazmat.primitives", primitives_mod)
sys.modules.setdefault("cryptography.hazmat.primitives.ciphers", ciphers_mod)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.ciphers.aead", aead_mod
)

matplotlib_mod = types.ModuleType("matplotlib")
pyplot_mod = types.ModuleType("matplotlib.pyplot")
pyplot_mod.figure = lambda *a, **k: None
pyplot_mod.plot = lambda *a, **k: None
pyplot_mod.savefig = lambda *a, **k: None
pyplot_mod.close = lambda *a, **k: None
matplotlib_mod.pyplot = pyplot_mod
sys.modules.setdefault("matplotlib", matplotlib_mod)
sys.modules.setdefault("matplotlib.pyplot", pyplot_mod)

otel_mod = types.ModuleType("opentelemetry")
otel_metrics = types.ModuleType("opentelemetry.metrics")
otel_metrics.get_meter_provider = lambda *a, **k: None
otel_metrics.set_meter_provider = lambda *a, **k: None
class _Meter:
    def create_histogram(self, *a, **k):
        return types.SimpleNamespace(record=lambda *args, **kwargs: None)

    def create_counter(self, *a, **k):
        return types.SimpleNamespace(add=lambda *args, **kwargs: None)

otel_metrics.get_meter = lambda *a, **k: _Meter()
otel_trace = types.ModuleType("opentelemetry.trace")
otel_trace.get_tracer_provider = lambda *a, **k: None
otel_trace.set_tracer_provider = lambda *a, **k: None
otel_mod.metrics = otel_metrics
otel_mod.trace = otel_trace
sys.modules.setdefault("opentelemetry", otel_mod)
sys.modules.setdefault("opentelemetry.metrics", otel_metrics)
sys.modules.setdefault("opentelemetry.trace", otel_trace)
otel_sdk = types.ModuleType("opentelemetry.sdk")
otel_sdk_resources = types.ModuleType("opentelemetry.sdk.resources")
class _Resource:
    @staticmethod
    def create(attrs):
        return _Resource()

otel_sdk_resources.Resource = _Resource
sys.modules.setdefault("opentelemetry.sdk", otel_sdk)
sys.modules.setdefault("opentelemetry.sdk.resources", otel_sdk_resources)
otel_sdk_trace = types.ModuleType("opentelemetry.sdk.trace")
class _TracerProvider:
    def __init__(self, *a, **k):
        pass

    def add_span_processor(self, *a, **k):
        pass

otel_sdk_trace.TracerProvider = _TracerProvider
sys.modules.setdefault("opentelemetry.sdk.trace", otel_sdk_trace)
otel_sdk_trace_export = types.ModuleType("opentelemetry.sdk.trace.export")
class _BatchSpanProcessor:
    def __init__(self, *a, **k):
        pass

otel_sdk_trace_export.BatchSpanProcessor = _BatchSpanProcessor
sys.modules.setdefault("opentelemetry.sdk.trace.export", otel_sdk_trace_export)
otel_exporter = types.ModuleType("opentelemetry.exporter")
otel_exporter_jaeger = types.ModuleType("opentelemetry.exporter.jaeger")
otel_exporter_jaeger_thrift = types.ModuleType(
    "opentelemetry.exporter.jaeger.thrift"
)
class _JaegerExporter:
    def __init__(self, *a, **k):
        pass

otel_exporter_jaeger_thrift.JaegerExporter = _JaegerExporter
sys.modules.setdefault("opentelemetry.exporter", otel_exporter)
sys.modules.setdefault("opentelemetry.exporter.jaeger", otel_exporter_jaeger)
sys.modules.setdefault(
    "opentelemetry.exporter.jaeger.thrift", otel_exporter_jaeger_thrift
)
otel_sdk_metrics = types.ModuleType("opentelemetry.sdk.metrics")
class _MeterProvider:
    def __init__(self, *a, **k):
        pass

otel_sdk_metrics.MeterProvider = _MeterProvider
sys.modules.setdefault("opentelemetry.sdk.metrics", otel_sdk_metrics)
otel_instr = types.ModuleType("opentelemetry.instrumentation")
otel_instr_logging = types.ModuleType("opentelemetry.instrumentation.logging")
otel_instr_logging.LoggingInstrumentor = type(
    "LoggingInstrumentor", (), {"instrument": lambda *a, **k: None}
)
sys.modules.setdefault("opentelemetry.instrumentation", otel_instr)
sys.modules.setdefault("opentelemetry.instrumentation.logging", otel_instr_logging)

import numpy as np
import pandas as pd
import pytest

from data.features import make_features
from data.labels import triple_barrier

# ensure real scipy is used
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.sparse", None)
sys.modules.pop("scipy.stats", None)
from mt5 import train


@pytest.fixture(autouse=True)
def stub_dependencies(monkeypatch):
    """Stub heavy external dependencies for the pipeline."""
    import data.features as f
    from features import price
    # use only price features and bypass heavy extras
    monkeypatch.setattr(f, "get_feature_pipeline", lambda: [price.compute])
    monkeypatch.setattr(f, "add_garch_volatility", lambda df: df)
    monkeypatch.setattr(f, "add_cross_spectral_features", lambda df: df)
    monkeypatch.setattr(f, "add_knowledge_graph_features", lambda df: df)
    monkeypatch.setattr(f, "add_frequency_features", lambda df: df)
    monkeypatch.setattr(f, "add_fractal_features", lambda df: df)
    monkeypatch.setattr(f, "add_factor_exposure_features", lambda df: df)
    import analysis.feature_gate as fg
    monkeypatch.setattr(fg, "select", lambda df, tier, regime_id, persist=False: (df, []))
    import analysis.knowledge_graph as kg
    monkeypatch.setattr(kg, "load_knowledge_graph", lambda: None)
    monkeypatch.setattr(kg, "risk_score", lambda g, c: 0.0)
    monkeypatch.setattr(kg, "opportunity_score", lambda g, c: 0.0)
    import features.news as news
    monkeypatch.setattr(news, "compute", lambda df: df)
    import features.cross_asset as cross_asset
    monkeypatch.setattr(cross_asset, "compute", lambda df: df)
    import analysis.data_lineage as dl
    monkeypatch.setattr(f, "log_lineage", lambda *a, **k: None)
    import data.labels as labels_mod
    monkeypatch.setattr(labels_mod, "log_lineage", lambda *a, **k: None)
    from mt5 import train as train_mod
    monkeypatch.setattr(train_mod, "_lgbm_params", lambda cfg: {})


@pytest.fixture
def synthetic_df():
    """Create a tiny synthetic price dataset."""
    n = 120
    ts = pd.date_range("2024-01-01", periods=n, freq="min")
    mid = np.sin(np.linspace(0, 6 * np.pi, n)) + np.linspace(100, 101, n)
    df = pd.DataFrame(
        {
            "Timestamp": ts,
            "Ask": mid + 0.01,
            "Bid": mid - 0.01,
            "Symbol": "TEST",
        }
    )
    return df


def test_full_pipeline(synthetic_df):
    # feature generation
    feat = make_features(synthetic_df)
    labels = triple_barrier(feat["mid"], pt_mult=0.001, sl_mult=0.001, max_horizon=5)

    feat = feat.fillna(0)
    labels = labels.fillna(0).astype(int)
    y = labels.loc[feat.index].to_frame("label")
    X = feat.drop(columns=["Timestamp", "Symbol"])

    model, metrics = train.train_multi_output_model(
        X, y, {"n_estimators": 5, "use_scaler": False, "n_jobs": 1}
    )

    assert metrics["aggregate_f1"] >= 0.5
    preds = model.predict(X)
    assert preds.shape == y.shape
