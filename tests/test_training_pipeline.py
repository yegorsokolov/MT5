import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest
import types
import numpy as np

if "data.labels" not in sys.modules:
    def _label_fn(series, horizons):
        data = {f"direction_{h}": pd.Series(0, index=series.index, dtype=int) for h in horizons}
        return pd.DataFrame(data)

    labels_stub = types.SimpleNamespace(multi_horizon_labels=_label_fn)
    sys.modules["data.labels"] = labels_stub

if "data.streaming" not in sys.modules:
    def _stream_labels(chunks, horizons):
        for chunk in chunks:
            yield sys.modules["data.labels"].multi_horizon_labels(chunk["mid"], horizons)

    sys.modules["data.streaming"] = types.SimpleNamespace(stream_labels=_stream_labels)

from training.data_loader import StreamingTrainingFrame, load_training_frame
from training.labels import generate_training_labels
from training.features import (
    apply_domain_adaptation,
    append_risk_profile_features,
    build_feature_candidates,
)
from training.postprocess import build_model_metadata, summarise_predictions
from training.utils import combined_sample_weight


class _DummyStrategy:
    def __init__(self) -> None:
        self.symbols: list[str] = []


def test_load_training_frame_override_returns_frame():
    df = pd.DataFrame({"a": [1, 2, 3]})
    cfg = SimpleNamespace(strategy=_DummyStrategy())
    result, source = load_training_frame(cfg, Path("."), df_override=df)
    pd.testing.assert_frame_equal(result, df)
    assert source == "override"


def test_load_training_frame_stream_returns_lazy_iterator(monkeypatch, tmp_path):
    symbols = ["TEST"]
    cfg = SimpleNamespace(strategy=SimpleNamespace(symbols=symbols))

    chunks = [
        pd.DataFrame({"mid": [1.0, 1.1], "Symbol": "TEST"}),
        pd.DataFrame({"mid": [1.2], "Symbol": "TEST"}),
    ]

    def fake_symbol_history_chunks(symbol, _cfg, _root, *, chunk_size, validate):
        assert symbol == "TEST"
        assert chunk_size == 5
        assert validate is False
        return iter(chunks)

    monkeypatch.setattr(
        "training.data_loader._symbol_history_chunks", fake_symbol_history_chunks
    )

    def fake_stream_features(frames, **kwargs):
        for frame in frames:
            out = frame.copy()
            out["return"] = out["mid"].pct_change().fillna(0)
            yield out

    saves = {"count": 0}

    def fake_save_history(df, path):
        saves["count"] += 1
        assert len(df) == sum(len(chunk) for chunk in chunks)

    fake_stream_module = types.SimpleNamespace(stream_features=fake_stream_features)
    fake_history_module = types.SimpleNamespace(save_history_parquet=fake_save_history)
    fake_data_pkg = types.SimpleNamespace(
        streaming=fake_stream_module,
        history=fake_history_module,
    )

    monkeypatch.setitem(sys.modules, "data", fake_data_pkg)
    monkeypatch.setitem(sys.modules, "data.streaming", fake_stream_module)
    monkeypatch.setitem(sys.modules, "data.history", fake_history_module)

    frame, source = load_training_frame(
        cfg,
        tmp_path,
        stream=True,
        chunk_size=5,
        feature_lookback=3,
    )
    assert isinstance(frame, StreamingTrainingFrame)
    assert source == "config"
    assert frame.materialise_count == 0

    observed_lengths = [len(chunk) for chunk in frame]
    assert observed_lengths == [2, 1]
    assert frame.materialise_count == 0

    df = frame.materialise()
    assert frame.materialise_count == 1
    assert len(df) == 3
    assert saves["count"] == 1

    # Re-materialising should reuse the cached dataframe without re-saving
    df_again = frame.materialise()
    assert frame.materialise_count == 1
    pd.testing.assert_frame_equal(df, df_again)
    assert saves["count"] == 1


def test_generate_training_labels_stream_matches_offline():
    idx = pd.RangeIndex(10)
    mid = pd.Series(np.linspace(100, 101, len(idx)), index=idx)
    df = pd.DataFrame({"mid": mid})
    offline = generate_training_labels(df, stream=False, horizons=[1], chunk_size=5)
    streamed = generate_training_labels(df, stream=True, horizons=[1], chunk_size=4)
    pd.testing.assert_frame_equal(streamed, offline)


def test_generate_training_labels_streaming_frame(monkeypatch):
    chunks = [
        pd.DataFrame(
            {
                "mid": [1.0, 1.1, 1.2],
                "return": [0.0, 0.1, 0.2],
                "Symbol": ["TEST"] * 3,
            }
        ),
        pd.DataFrame(
            {
                "mid": [1.3, 1.4],
                "return": [0.05, -0.02],
                "Symbol": ["TEST"] * 2,
            }
        ),
    ]
    frame = StreamingTrainingFrame(iter(chunks))
    labels_df = generate_training_labels(frame, stream=True, horizons=[1], chunk_size=2)
    assert "direction_1" in labels_df.columns
    assert frame.materialise_count == 0
    cached = frame.collect_chunks()
    assert cached and all("direction_1" in chunk.columns for chunk in cached)
    df_full = frame.materialise()
    assert frame.materialise_count == 1
    assert "direction_1" in df_full.columns


def test_combined_sample_weight_respects_quality_and_decay():
    y = np.array([0, 0, 1, 1], dtype=int)
    timestamps = np.array([0, 1, 2, 3], dtype=np.int64)
    dq = np.array([1.0, 0.5, 0.5, 1.0])
    weights = combined_sample_weight(y, timestamps, timestamps.max(), True, 2, dq)
    assert weights is not None
    assert weights[-1] > weights[1]
    assert weights[1] < weights[0]


def test_postprocess_helpers_round_trip():
    meta = build_model_metadata({0: 0.5}, interval_alpha=0.1, interval_coverage=0.9)
    assert meta["regime_thresholds"][0] == 0.5
    assert meta["interval_alpha"] == 0.1
    frame = summarise_predictions([0, 1], [0, 1], [0.3, 0.7], [0, 1], lower=[0.1, 0.2], upper=[0.9, 0.8])
    assert list(frame.columns) == ["y_true", "pred", "prob", "market_regime", "lower", "upper"]


def test_streaming_feature_flow_preserves_lazy_materialisation(monkeypatch, tmp_path):
    import analysis.domain_adapter as domain_adapter

    chunks = [
        pd.DataFrame(
            {
                "mid": [1.0, 1.1],
                "return": [0.0, 0.1],
                "Symbol": ["TEST"] * 2,
            }
        ),
        pd.DataFrame(
            {
                "mid": [1.2, 1.3],
                "return": [0.2, -0.05],
                "Symbol": ["TEST"] * 2,
            }
        ),
    ]
    frame = StreamingTrainingFrame(iter(chunks))

    class _DummyAdapter:
        def __init__(self) -> None:
            self.transformed: list[int] = []

        def transform(self, X: pd.DataFrame) -> pd.DataFrame:
            self.transformed.append(len(X))
            return X + 1

        def save(self, path):  # pragma: no cover - behaviour not critical for assertions
            self.saved_path = path
            return path

    holder: dict[str, _DummyAdapter] = {}

    def _fake_load(cls, path=None):
        adapter = _DummyAdapter()
        holder["adapter"] = adapter
        return adapter

    monkeypatch.setattr(domain_adapter.DomainAdapter, "load", classmethod(_fake_load))

    reclass_calls = {"count": 0}

    def _fake_reclass(df: pd.DataFrame, *, step: int = 500, **_kwargs):
        reclass_calls["count"] += 1
        out = df.copy()
        out["market_regime"] = 0
        return out

    monkeypatch.setattr(
        "analysis.regime_detection.periodic_reclassification", _fake_reclass
    )

    adapted = apply_domain_adaptation(frame, tmp_path / "adapter.pkl", regime_step=2)
    assert isinstance(adapted, StreamingTrainingFrame)
    assert frame.materialise_count == 0
    assert "adapter" in holder

    risk_profile = types.SimpleNamespace(leverage_cap=2.0, drawdown_limit=0.1, tolerance=0.5)
    budget = append_risk_profile_features(frame, risk_profile)
    assert budget.max_leverage == 2.0

    features = build_feature_candidates(frame, budget)
    assert "risk_tolerance" in features

    labels_df = generate_training_labels(frame, stream=True, horizons=[1], chunk_size=2)
    assert "direction_1" in labels_df.columns
    assert frame.materialise_count == 0

    cached_chunks = frame.collect_chunks()
    assert holder["adapter"].transformed == [len(chunk) for chunk in cached_chunks if not chunk.empty]
    assert all("risk_tolerance" in chunk.columns for chunk in cached_chunks)

    df_full = frame.materialise()
    assert frame.materialise_count == 1
    assert "market_regime" in df_full.columns
    assert "direction_1" in df_full.columns
    assert reclass_calls["count"] == 1


def test_run_training_ends_mlflow_on_exception(monkeypatch):
    class _StubLGBM:
        def __init__(self, *args, **kwargs) -> None:
            pass

    monkeypatch.setitem(sys.modules, "lightgbm", types.SimpleNamespace(LGBMClassifier=_StubLGBM))
    fake_ge = types.SimpleNamespace()
    fake_suite = types.SimpleNamespace(ExpectationSuite=object)
    fake_ge.core = types.SimpleNamespace(expectation_suite=fake_suite)
    monkeypatch.setitem(sys.modules, "great_expectations", fake_ge)
    monkeypatch.setitem(sys.modules, "great_expectations.core", fake_ge.core)
    monkeypatch.setitem(
        sys.modules,
        "great_expectations.core.expectation_suite",
        fake_suite,
    )
    fake_features = types.ModuleType("features")
    fake_features.get_feature_pipeline = lambda *args, **kwargs: []
    fake_features.make_features = lambda *args, **kwargs: None
    fake_features.start_capability_watch = lambda: None
    fake_features.__path__ = []  # mark as package for submodule imports
    monkeypatch.setitem(sys.modules, "features", fake_features)

    news_module = types.ModuleType("features.news")
    news_module.add_economic_calendar_features = lambda *a, **k: None
    news_module.add_news_sentiment_features = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "features.news", news_module)

    cross_asset_module = types.ModuleType("features.cross_asset")
    cross_asset_module.add_index_features = lambda *a, **k: None
    cross_asset_module.add_cross_asset_features = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "features.cross_asset", cross_asset_module)

    validators_module = types.ModuleType("features.validators")
    validators_module.validate_ge = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "features.validators", validators_module)
    monkeypatch.setitem(sys.modules, "networkx", types.ModuleType("networkx"))
    monkeypatch.setitem(sys.modules, "pywt", types.ModuleType("pywt"))
    monkeypatch.setitem(sys.modules, "gplearn", types.ModuleType("gplearn"))
    gplearn_genetic = types.ModuleType("gplearn.genetic")
    gplearn_genetic.SymbolicTransformer = object
    monkeypatch.setitem(sys.modules, "gplearn.genetic", gplearn_genetic)
    scheduler_module = types.ModuleType("scheduler")
    scheduler_module.schedule_retrain = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "scheduler", scheduler_module)
    matplotlib_module = types.ModuleType("matplotlib")
    pyplot_module = types.SimpleNamespace(
        figure=lambda *a, **k: None,
        plot=lambda *a, **k: None,
        xlabel=lambda *a, **k: None,
        ylabel=lambda *a, **k: None,
        title=lambda *a, **k: None,
        legend=lambda *a, **k: None,
        tight_layout=lambda *a, **k: None,
        savefig=lambda *a, **k: None,
        close=lambda *a, **k: None,
    )
    matplotlib_module.pyplot = pyplot_module
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib_module)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_module)
    sklearn_module = types.ModuleType("sklearn")
    sklearn_module.__path__ = []
    cluster_module = types.ModuleType("sklearn.cluster")

    class _StubKMeans:
        def __init__(self, *args, **kwargs) -> None:
            pass

    cluster_module.KMeans = _StubKMeans
    monkeypatch.setitem(sys.modules, "sklearn", sklearn_module)
    monkeypatch.setitem(sys.modules, "sklearn.cluster", cluster_module)
    base_module = types.ModuleType("sklearn.base")
    base_module.BaseEstimator = type("BaseEstimator", (), {})
    base_module.ClassifierMixin = type("ClassifierMixin", (), {})
    monkeypatch.setitem(sys.modules, "sklearn.base", base_module)
    ensemble_module = types.ModuleType("sklearn.ensemble")
    ensemble_module.RandomForestClassifier = type("RandomForestClassifier", (), {})
    monkeypatch.setitem(sys.modules, "sklearn.ensemble", ensemble_module)
    feature_selection_module = types.ModuleType("sklearn.feature_selection")
    feature_selection_module.mutual_info_classif = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "sklearn.feature_selection", feature_selection_module)
    linear_model_module = types.ModuleType("sklearn.linear_model")
    class _StubLinearRegression:
        def fit(self, *args, **kwargs):
            return self

        def predict(self, X):
            return np.zeros(len(np.asarray(X)))

    class _StubLogisticRegression:
        def fit(self, *args, **kwargs):
            return self

        def predict_proba(self, X):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            n = len(arr)
            return np.column_stack([np.zeros(n), np.zeros(n)])

    linear_model_module.LinearRegression = _StubLinearRegression
    linear_model_module.LogisticRegression = _StubLogisticRegression
    monkeypatch.setitem(sys.modules, "sklearn.linear_model", linear_model_module)
    model_selection_module = types.ModuleType("sklearn.model_selection")
    model_selection_module.KFold = type("KFold", (), {})
    model_selection_module.cross_val_score = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "sklearn.model_selection", model_selection_module)
    calibration_module = types.ModuleType("sklearn.calibration")

    class _StubCalibratedClassifierCV:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return self

        def predict_proba(self, X):
            arr = np.asarray(X)
            if arr.ndim == 1:
                arr = arr.reshape(-1, 1)
            n = len(arr)
            return np.column_stack([np.zeros(n), np.zeros(n)])

    calibration_module.CalibratedClassifierCV = _StubCalibratedClassifierCV
    calibration_module.calibration_curve = lambda *a, **k: (
        np.array([0.0]),
        np.array([0.0]),
    )
    monkeypatch.setitem(sys.modules, "sklearn.calibration", calibration_module)
    isotonic_module = types.ModuleType("sklearn.isotonic")

    class _StubIsotonicRegression:
        def __init__(self, *args, **kwargs):
            pass

        def fit(self, *args, **kwargs):
            return self

        def predict(self, X):
            arr = np.asarray(X)
            return np.zeros(len(arr))

    isotonic_module.IsotonicRegression = _StubIsotonicRegression
    monkeypatch.setitem(sys.modules, "sklearn.isotonic", isotonic_module)
    metrics_module = types.ModuleType("sklearn.metrics")
    metrics_module.classification_report = (
        lambda *a, **k: {"weighted avg": {"f1-score": 0.0, "precision": 0.0, "recall": 0.0}}
    )
    metrics_module.precision_recall_curve = (
        lambda *a, **k: (np.array([0.0]), np.array([0.0]), np.array([0.0]))
    )
    metrics_module.precision_score = lambda *a, **k: 0.0
    metrics_module.recall_score = lambda *a, **k: 0.0
    metrics_module.f1_score = lambda *a, **k: 0.0
    metrics_module.brier_score_loss = lambda *a, **k: 0.0
    monkeypatch.setitem(sys.modules, "sklearn.metrics", metrics_module)
    pipeline_module = types.ModuleType("sklearn.pipeline")

    class _StubPipeline:
        def __init__(self, steps):
            self.steps = list(steps)
            self.named_steps = {name: obj for name, obj in self.steps}

        def fit(self, *args, **kwargs):
            return self

        def __getitem__(self, item):
            if isinstance(item, slice):
                return self
            return self.steps[item]

        def transform(self, X):
            return X

    pipeline_module.Pipeline = _StubPipeline
    monkeypatch.setitem(sys.modules, "sklearn.pipeline", pipeline_module)

    fake_torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch.utils")
    torch_utils_data = types.ModuleType("torch.utils.data")
    torch_utils_data.DataLoader = object
    torch_utils_data.TensorDataset = object
    torch_utils.data = torch_utils_data
    torch_nn = types.ModuleType("torch.nn")

    class _TorchModule:
        def __init__(self, *args, **kwargs) -> None:
            pass

    torch_nn.Module = _TorchModule
    torch_nn.Parameter = lambda value: value
    fake_torch.Tensor = object
    fake_torch.randn = lambda *args, **kwargs: 0
    fake_torch.cuda = types.SimpleNamespace(device_count=lambda: 0)
    fake_torch.utils = types.SimpleNamespace(data=torch_utils_data)
    fake_torch.nn = torch_nn
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn)

    from training import pipeline

    monkeypatch.setattr(pipeline, "init_logging", lambda: None)

    class StubMlflow:
        def __init__(self) -> None:
            self.started = 0
            self.ended = 0

        def start_run(self, *args, **kwargs):
            self.started += 1

        def end_run(self, *args, **kwargs):
            self.ended += 1

        def log_param(self, *args, **kwargs):
            pass

        def log_metric(self, *args, **kwargs):
            pass

        def log_artifact(self, *args, **kwargs):
            pass

    stub_mlflow = StubMlflow()
    monkeypatch.setattr(pipeline, "mlflow", stub_mlflow)
    monkeypatch.setattr(pipeline, "_subscribe_cpu_updates", lambda cfg: None)
    monkeypatch.setattr(pipeline, "ensure_environment", lambda: None)

    def explode(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline, "load_training_frame", explode)

    class DummyConfig:
        def __init__(self) -> None:
            self.training = SimpleNamespace(
                model_type="lgbm",
                use_pseudo_labels=False,
                seed=1,
                drift_method="ks",
                drift_delta=0.1,
                use_focal_loss=False,
                focal_alpha=0.25,
                focal_gamma=2.0,
                num_leaves=None,
                learning_rate=None,
                max_depth=None,
            )

        def model_dump(self):
            return {}

        def get(self, key, default=None):
            return getattr(self, key, default)

    cfg = DummyConfig()
    with pytest.raises(RuntimeError):
        pipeline._run_training(cfg=cfg)

    assert stub_mlflow.started == 1
    assert stub_mlflow.ended == 1
