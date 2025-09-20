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

if "lightgbm" not in sys.modules:

    class _StubLGBMClassifier:
        def __init__(self, **params):
            self._params = {"n_estimators": 100, "n_jobs": params.get("n_jobs", 1)}
            self._params.update(params)
            for key, value in self._params.items():
                setattr(self, key, value)
            self.fitted_ = False

        def get_params(self, deep: bool = True) -> dict:
            return dict(self._params)

        def set_params(self, **params):  # type: ignore[override]
            self._params.update(params)
            for key, value in params.items():
                setattr(self, key, value)
            return self

        def fit(self, X, y=None, **kwargs):  # type: ignore[override]
            self.fitted_ = True
            self.booster_ = kwargs.get("init_model")
            return self

    sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=_StubLGBMClassifier)

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


def test_active_classifier_tracker_releases_finished_estimators(monkeypatch):
    import asyncio
    import contextlib
    import gc
    import weakref
    import sys

    import types as _types

    class _MonitorStub:
        def __init__(self) -> None:
            self.capabilities = types.SimpleNamespace(cpus=2)
            self.queue: asyncio.Queue | None = None
            self.tasks: list[asyncio.Task] = []

        def subscribe(self) -> asyncio.Queue:
            if self.queue is None:
                self.queue = asyncio.Queue()
            return self.queue

        def create_task(self, coro):  # type: ignore[override]
            task = asyncio.create_task(coro)
            self.tasks.append(task)
            return task

    class _MiniPipeline:
        def __init__(self, clf):
            self.named_steps = {"clf": clf}

        def fit(self, X, y):  # pragma: no cover - simple passthrough
            if hasattr(self.named_steps["clf"], "fit"):
                self.named_steps["clf"].fit(X, y)
            return self

    async def _exercise() -> None:
        ge_mod = _types.ModuleType("great_expectations")
        ge_dataset_mod = _types.ModuleType("great_expectations.dataset")

        class _StubResult(dict):
            def __init__(self) -> None:
                super().__init__(success=True)

            def to_json_dict(self) -> dict:
                return dict(self)

        class _StubDataset:
            def __init__(self, df):
                self.df = df

            def validate(self, expectation_suite):  # pragma: no cover - simple stub
                return _StubResult()

        ge_dataset_mod.PandasDataset = _StubDataset
        ge_core_mod = _types.ModuleType("great_expectations.core")
        ge_expectation_mod = _types.ModuleType("great_expectations.core.expectation_suite")
        ge_expectation_mod.ExpectationSuite = object
        ge_mod.dataset = ge_dataset_mod
        ge_mod.core = ge_core_mod
        ge_core_mod.expectation_suite = ge_expectation_mod
        monkeypatch.setitem(sys.modules, "great_expectations", ge_mod)
        monkeypatch.setitem(sys.modules, "great_expectations.dataset", ge_dataset_mod)
        monkeypatch.setitem(sys.modules, "great_expectations.core", ge_core_mod)
        monkeypatch.setitem(
            sys.modules,
            "great_expectations.core.expectation_suite",
            ge_expectation_mod,
        )

        joblib_mod = sys.modules.get("joblib")
        if joblib_mod is None:
            joblib_mod = _types.ModuleType("joblib")
            monkeypatch.setitem(sys.modules, "joblib", joblib_mod)
        monkeypatch.setattr(joblib_mod, "Memory", lambda *a, **k: None, raising=False)

        features_mod = _types.ModuleType("features")
        features_mod.start_capability_watch = lambda: None
        features_mod.auto_indicators = types.SimpleNamespace(
            REGISTRY_PATH=Path("."),
            apply=lambda X, registry_path=None, formula_dir=None: X,
            generate=lambda X, hypernet, asset, reg, registry_path=None: (X, {}),
        )
        monkeypatch.setitem(sys.modules, "features", features_mod)

        mpl_mod = _types.ModuleType("matplotlib")
        pyplot_mod = types.SimpleNamespace(
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
        mpl_mod.pyplot = pyplot_mod
        monkeypatch.setitem(sys.modules, "matplotlib", mpl_mod)
        monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot_mod)

        sklearn_mod = _types.ModuleType("sklearn")
        sklearn_mod.__path__ = []
        monkeypatch.setitem(sys.modules, "sklearn", sklearn_mod)

        sklearn_base = _types.ModuleType("sklearn.base")

        class _StubBaseEstimator:
            pass

        class _StubClassifierMixin:
            pass

        sklearn_base.BaseEstimator = _StubBaseEstimator
        sklearn_base.ClassifierMixin = _StubClassifierMixin
        monkeypatch.setitem(sys.modules, "sklearn.base", sklearn_base)

        sklearn_calib = _types.ModuleType("sklearn.calibration")

        class _StubCalibratedClassifierCV:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def fit(self, *args, **kwargs):  # pragma: no cover - trivial stub
                return self

            def predict_proba(self, X):  # pragma: no cover - trivial stub
                arr = np.asarray(X)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                n = len(arr)
                return np.column_stack([np.zeros(n), np.zeros(n)])

        sklearn_calib.CalibratedClassifierCV = _StubCalibratedClassifierCV
        sklearn_calib.calibration_curve = (
            lambda *a, **k: (np.array([0.0]), np.array([0.0]))
        )
        monkeypatch.setitem(sys.modules, "sklearn.calibration", sklearn_calib)

        sklearn_isotonic = _types.ModuleType("sklearn.isotonic")

        class _StubIsotonicRegression:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def fit(self, *args, **kwargs):  # pragma: no cover - trivial stub
                return self

            def predict(self, X):  # pragma: no cover - trivial stub
                arr = np.asarray(X)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                return np.zeros(len(arr))

        sklearn_isotonic.IsotonicRegression = _StubIsotonicRegression
        monkeypatch.setitem(sys.modules, "sklearn.isotonic", sklearn_isotonic)

        sklearn_linear = _types.ModuleType("sklearn.linear_model")

        class _StubLogisticRegression:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def fit(self, *args, **kwargs):  # pragma: no cover - trivial stub
                return self

            def predict_proba(self, X):  # pragma: no cover - trivial stub
                arr = np.asarray(X)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                n = len(arr)
                return np.column_stack([np.zeros(n), np.zeros(n)])

        sklearn_linear.LogisticRegression = _StubLogisticRegression
        monkeypatch.setitem(sys.modules, "sklearn.linear_model", sklearn_linear)

        sklearn_metrics = _types.ModuleType("sklearn.metrics")
        sklearn_metrics.brier_score_loss = lambda *a, **k: 0.0
        sklearn_metrics.precision_score = lambda *a, **k: 0.0
        sklearn_metrics.recall_score = lambda *a, **k: 0.0
        sklearn_metrics.f1_score = lambda *a, **k: 0.0
        sklearn_metrics.classification_report = (
            lambda *a, **k: {"weighted avg": {"precision": 0.0, "recall": 0.0, "f1-score": 0.0}}
        )
        sklearn_metrics.precision_recall_curve = (
            lambda *a, **k: (np.array([0.0]), np.array([0.0]), np.array([0.0]))
        )
        monkeypatch.setitem(sys.modules, "sklearn.metrics", sklearn_metrics)

        scheduler_mod = _types.ModuleType("scheduler")
        scheduler_mod.schedule_retrain = lambda *a, **k: None
        monkeypatch.setitem(sys.modules, "scheduler", scheduler_mod)

        sklearn_ensemble = _types.ModuleType("sklearn.ensemble")

        class _StubRandomForestClassifier:
            def __init__(self, *args, **kwargs) -> None:
                pass

            def fit(self, *args, **kwargs):  # pragma: no cover - trivial stub
                return self

            def predict_proba(self, X):  # pragma: no cover - trivial stub
                arr = np.asarray(X)
                if arr.ndim == 1:
                    arr = arr.reshape(-1, 1)
                n = len(arr)
                return np.column_stack([np.zeros(n), np.zeros(n)])

        sklearn_ensemble.RandomForestClassifier = _StubRandomForestClassifier
        monkeypatch.setitem(sys.modules, "sklearn.ensemble", sklearn_ensemble)

        sklearn_pipeline = _types.ModuleType("sklearn.pipeline")

        class _StubPipeline:
            def __init__(self, steps):
                self.steps = list(steps)
                self.named_steps = {name: step for name, step in self.steps}

            def fit(self, X, y=None, **kwargs):  # pragma: no cover - trivial stub
                for _, step in self.steps:
                    if hasattr(step, "fit"):
                        step.fit(X, y)
                return self

            def __getitem__(self, item):  # pragma: no cover - trivial stub
                if isinstance(item, slice):
                    return self
                return self.steps[item]

        sklearn_pipeline.Pipeline = _StubPipeline
        monkeypatch.setitem(sys.modules, "sklearn.pipeline", sklearn_pipeline)

        from training import pipeline

        monitor_stub = _MonitorStub()
        monkeypatch.setattr(pipeline, "monitor", monitor_stub)
        pipeline._ACTIVE_CLFS.clear()
        cfg: dict[str, object] = {}
        pipeline._subscribe_cpu_updates(cfg)
        await asyncio.sleep(0)

        try:
            X = np.array([[0.0], [1.0], [2.0], [3.0]], dtype=float)
            y = np.array([0, 1, 0, 1], dtype=int)

            live_pipe = _MiniPipeline(pipeline.LGBMClassifier(n_estimators=5, n_jobs=1))
            live_pipe.fit(X, y)
            live_clf = live_pipe.named_steps["clf"]
            pipeline._register_clf(live_clf)

            stale_pipe = _MiniPipeline(pipeline.LGBMClassifier(n_estimators=5, n_jobs=1))
            stale_pipe.fit(X, y)
            stale_clf = stale_pipe.named_steps["clf"]
            pipeline._register_clf(stale_clf)

            assert len(pipeline._ACTIVE_CLFS) == 2

            stale_ref = weakref.ref(stale_clf)
            del stale_pipe
            del stale_clf
            for _ in range(3):
                gc.collect()
                await asyncio.sleep(0)
                if len(pipeline._ACTIVE_CLFS) == 1:
                    break

            assert stale_ref() is None
            assert len(pipeline._ACTIVE_CLFS) == 1

            live_clf.set_params(n_jobs=1)
            monitor_stub.capabilities.cpus = 7
            assert monitor_stub.queue is not None
            monitor_stub.queue.put_nowait("capability-update")
            for _ in range(5):
                await asyncio.sleep(0)
                if live_clf.get_params().get("n_jobs") == 7:
                    break
            assert live_clf.get_params().get("n_jobs") == 7
        finally:
            for task in monitor_stub.tasks:
                task.cancel()
            if monitor_stub.tasks:
                await asyncio.gather(*monitor_stub.tasks, return_exceptions=True)
            pipeline._ACTIVE_CLFS.clear()

    asyncio.run(_exercise())


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
