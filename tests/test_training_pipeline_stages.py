"""Unit tests for the modular training pipeline stages."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import types

proj_root = Path(__file__).resolve().parents[1]
if str(proj_root) not in sys.path:
    sys.path.insert(0, str(proj_root))

if "lightgbm" not in sys.modules:

    class _StubLGBMClassifier:
        def __init__(self, **_params) -> None:
            self.booster_ = None

        def fit(self, X, y=None, **_kwargs):
            self.booster_ = "stub"
            return self

        def predict_proba(self, X):
            return [[0.4, 0.6] for _ in range(len(X))]

    sys.modules["lightgbm"] = types.SimpleNamespace(LGBMClassifier=_StubLGBMClassifier)

if "mt5.crypto_utils" not in sys.modules:
    crypto_stub = types.ModuleType("mt5.crypto_utils")
    crypto_stub._load_key = lambda *a, **k: None
    crypto_stub.encrypt = lambda data, *a, **k: data
    crypto_stub.decrypt = lambda data, *a, **k: data
    sys.modules["mt5.crypto_utils"] = crypto_stub


class DummyConfig:
    """Minimal configuration stub mimicking :class:`AppConfig`."""

    def __init__(
        self,
        *,
        training: dict[str, Any] | None = None,
        strategy: dict[str, Any] | None = None,
        **extras: Any,
    ) -> None:
        training = training or {}
        strategy = strategy or {}
        self.training = SimpleNamespace(**training)
        risk = strategy.get("risk_profile", {})
        strategy_fields = {k: v for k, v in strategy.items() if k != "risk_profile"}
        self.strategy = SimpleNamespace(**strategy_fields)
        self.strategy.risk_profile = SimpleNamespace(**risk)
        self._extras = dict(extras)
        self._extras.setdefault("active_learning", {})
        self._extras.setdefault("model", {})
        for attr in ("num_leaves", "learning_rate", "max_depth"):
            if not hasattr(self.training, attr):
                setattr(self.training, attr, None)

    def get(self, key: str, default: Any | None = None) -> Any:
        if hasattr(self.training, key):
            return getattr(self.training, key)
        if hasattr(self.strategy, key):
            return getattr(self.strategy, key)
        return self._extras.get(key, default)

    def setdefault(self, key: str, default: Any) -> Any:
        return self._extras.setdefault(key, default)


if "mt5.config_models" not in sys.modules:
    config_stub = types.ModuleType("mt5.config_models")

    class _StubAppConfig(DummyConfig):
        def __init__(self, **data) -> None:
            super().__init__(**data)

        def model_dump(self) -> dict:
            return {}

        def update_from(self, other):  # pragma: no cover - compatibility shim
            return None

    config_stub.AppConfig = _StubAppConfig
    config_stub.ConfigError = RuntimeError
    sys.modules["mt5.config_models"] = config_stub

if "great_expectations" not in sys.modules:
    ge_mod = types.ModuleType("great_expectations")
    ge_dataset_mod = types.ModuleType("great_expectations.dataset")

    class _StubDataset:
        def __init__(self, df) -> None:
            self.df = df

        def validate(self, expectation_suite=None):  # pragma: no cover - stub
            return {"success": True}

    ge_dataset_mod.PandasDataset = _StubDataset
    ge_core_mod = types.ModuleType("great_expectations.core")
    ge_expectation_mod = types.ModuleType("great_expectations.core.expectation_suite")
    ge_expectation_mod.ExpectationSuite = object
    ge_core_mod.expectation_suite = ge_expectation_mod
    ge_mod.dataset = ge_dataset_mod
    ge_mod.core = ge_core_mod
    sys.modules["great_expectations"] = ge_mod
    sys.modules["great_expectations.dataset"] = ge_dataset_mod
    sys.modules["great_expectations.core"] = ge_core_mod
    sys.modules["great_expectations.core.expectation_suite"] = ge_expectation_mod

if "features" not in sys.modules:
    features_stub = types.ModuleType("features")
    features_stub.start_capability_watch = lambda: None
    features_stub.auto_indicators = SimpleNamespace(
        REGISTRY_PATH=Path("."),
        apply=lambda X, registry_path=None, formula_dir=None: X,
        generate=lambda X, hypernet, asset, reg, registry_path=None: (X, {}),
    )
    sys.modules["features"] = features_stub

if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = SimpleNamespace(
        figure=lambda *a, **k: None,
        plot=lambda *a, **k: None,
        xlabel=lambda *a, **k: None,
        ylabel=lambda *a, **k: None,
        title=lambda *a, **k: None,
        legend=lambda *a, **k: None,
        tight_layout=lambda *a, **k: None,
        savefig=lambda *a, **k: None,
        close=lambda *a, **k: None,
        show=lambda *a, **k: None,
    )
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

if "analysis.concept_drift" not in sys.modules:
    import importlib

    drift_stub = types.ModuleType("analysis.concept_drift")

    class _StubConceptDriftMonitor:
        def __init__(self, *a, **k) -> None:
            pass

        def update(self, *a, **k) -> None:
            return None

    drift_stub.ConceptDriftMonitor = _StubConceptDriftMonitor
    analysis_pkg = importlib.import_module("analysis")
    sys.modules["analysis.concept_drift"] = drift_stub
    setattr(analysis_pkg, "concept_drift", drift_stub)

if "training.features" not in sys.modules:
    features_mod_stub = types.ModuleType("training.features")
    features_mod_stub.apply_domain_adaptation = lambda df, adapter_path, regime_step=500: df
    features_mod_stub.append_risk_profile_features = lambda df, rp: None
    features_mod_stub.build_feature_candidates = lambda df, budget, cfg=None: list(getattr(df, "columns", []))
    features_mod_stub.select_model_features = lambda df, feats, target, **kwargs: list(feats)
    sys.modules["training.features"] = features_mod_stub

for mod_name in ["scipy", "scipy.sparse", "scipy.stats"]:
    sys.modules.pop(mod_name, None)

from training import pipeline


@pytest.fixture
def minimal_cfg() -> DummyConfig:
    """Return a lightweight configuration object used across tests."""

    return DummyConfig(
        strategy={
            "symbols": ["EURUSD"],
            "risk_per_trade": 0.01,
            "risk_profile": {
                "tolerance": 1.0,
                "leverage_cap": 2.0,
                "drawdown_limit": 0.1,
            },
        },
        training={
            "seed": 7,
            "model_type": "lgbm",
            "n_jobs": 1,
            "online_batch_size": 2,
            "use_scaler": False,
        },
    )


def test_load_histories_returns_expected_shape(monkeypatch, tmp_path: Path, minimal_cfg: DummyConfig) -> None:
    frame = pd.DataFrame({"mid": [1.0, 1.1, 1.2]})
    calls: dict[str, Any] = {}

    def fake_loader(cfg, root, *, df_override=None, stream, chunk_size, feature_lookback, validate):
        calls.update(
            {
                "cfg": cfg,
                "root": root,
                "stream": stream,
                "chunk_size": chunk_size,
                "feature_lookback": feature_lookback,
                "validate": validate,
            }
        )
        assert df_override is None
        return frame, "history_source"

    def fake_adapter(data, adapter_path, *, regime_step):
        calls["adapter_path"] = adapter_path
        calls["regime_step"] = regime_step
        assert data is frame
        return data

    monkeypatch.setattr(pipeline, "load_training_frame", fake_loader)
    monkeypatch.setattr(pipeline, "apply_domain_adaptation", fake_adapter)

    minimal_cfg.training.validate = True
    minimal_cfg.training.stream_chunk_size = 50
    minimal_cfg.training.stream_feature_lookback = 16

    result = pipeline.load_histories(minimal_cfg, tmp_path)

    assert isinstance(result, pipeline.HistoryLoadResult)
    pd.testing.assert_frame_equal(result.frame, frame)
    assert result.data_source == "history_source"
    assert result.stream is False
    assert result.chunk_size == 50
    assert result.feature_lookback == 16
    assert result.validate is True
    assert calls["root"] == tmp_path
    assert calls["adapter_path"] == tmp_path / "domain_adapter.pkl"
    assert calls["regime_step"] == 500


def test_prepare_features_returns_dataclass(monkeypatch, tmp_path: Path, minimal_cfg: DummyConfig) -> None:
    minimal_cfg.training.model_type = "lgbm"
    base = pd.DataFrame({"feature_a": [0.1, 0.2], "return": [0.0, 0.1]})
    history = pipeline.HistoryLoadResult(
        frame=base,
        data_source="synthetic",
        stream_metadata=None,
        stream=False,
        chunk_size=32,
        feature_lookback=8,
        validate=False,
    )

    budget = pipeline.RiskBudget(max_leverage=2.0, max_drawdown=0.1, cvar_limit=None)

    def fake_append(df, risk_profile):
        df = df.copy()
        df["risk_tolerance"] = risk_profile.tolerance
        df["leverage_cap"] = risk_profile.leverage_cap
        df["drawdown_limit"] = risk_profile.drawdown_limit
        return budget

    def fake_build(df, *_args, **_kwargs):
        return ["feature_a", "risk_tolerance"]

    labels = pd.DataFrame({"direction_1": [0, 1]}, index=base.index)

    monkeypatch.setattr(pipeline, "append_risk_profile_features", fake_append)
    monkeypatch.setattr(pipeline, "build_feature_candidates", fake_build)
    monkeypatch.setattr(pipeline, "generate_training_labels", lambda *a, **k: labels)
    monkeypatch.setattr(pipeline, "select_model_features", lambda df, feats, *_args, **_kwargs: list(dict.fromkeys(feats)))
    monkeypatch.setattr(
        pipeline,
        "add_similar_day_features",
        lambda df, feature_cols, return_col, k, index_path: (
            df.assign(nn_return_mean=0.0, nn_vol=0.0),
            {"k": k, "path": index_path},
        ),
    )
    monkeypatch.setattr(pipeline, "prepare_modal_arrays", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "CrossModalTransformer", None)
    monkeypatch.setattr(pipeline, "torch", None)

    result = pipeline.prepare_features(history, minimal_cfg, "lgbm", tmp_path)

    assert isinstance(result, pipeline.FeaturePreparationResult)
    pd.testing.assert_frame_equal(result.labels, labels)
    assert set(result.features) >= {"feature_a", "risk_tolerance", "nn_return_mean", "nn_vol"}
    assert result.use_multi_task_heads is False
    assert result.user_budget == budget
    assert (tmp_path / "selected_features.json").exists()


def test_build_datasets_merges_labels_and_risk_budget(monkeypatch, tmp_path: Path, minimal_cfg: DummyConfig) -> None:
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "feature_a": [0.1, 0.2],
            "return": [0.0, 0.1],
        }
    )
    labels = pd.DataFrame({"direction_1": [0, 1]})
    features_state = pipeline.FeaturePreparationResult(
        df=df,
        features=["feature_a"],
        labels=labels,
        label_cols=["direction_1"],
        abs_label_cols=[],
        vol_label_cols=[],
        sel_target=labels["direction_1"],
        use_multi_task_heads=False,
        user_budget=pipeline.RiskBudget(2.0, 0.1, cvar_limit=None),
    )

    merged_ids: list[int] = []

    def fake_merge(df_in, new_labels, label_col):
        merged_ids.extend(new_labels["id"].tolist())
        out = df_in.copy()
        out[label_col] = 1
        return out

    saved: dict[str, Any] = {}

    def fake_save_history(df_in, path):
        saved["path"] = path
        saved["rows"] = len(df_in)

    class DummyQueue:
        def __init__(self) -> None:
            self.popped = False

        def pop_labeled(self) -> pd.DataFrame:
            self.popped = True
            return pd.DataFrame({"id": [0], "label": [1]})

        def stats(self) -> dict[str, int]:
            return {"awaiting_label": 3, "ready_for_merge": 1}

    monkeypatch.setattr(pipeline, "ActiveLearningQueue", DummyQueue)
    monkeypatch.setattr(pipeline, "merge_labels", fake_merge)
    monkeypatch.setattr(pipeline, "save_history_parquet", fake_save_history)

    dataset = pipeline.build_datasets(
        features_state,
        minimal_cfg,
        tmp_path,
        use_pseudo_labels=False,
        risk_target={"max_leverage": 1.5, "max_drawdown": 0.2, "cvar": 0.1},
    )

    assert isinstance(dataset, pipeline.DatasetBuildResult)
    assert dataset.risk_budget is not None
    assert dataset.risk_budget.max_leverage == pytest.approx(1.5)
    assert set(dataset.features) >= {"feature_a", "risk_max_leverage", "risk_max_drawdown", "risk_cvar_limit"}
    assert list(dataset.y["direction_1"]) == [1, 1]
    assert merged_ids == [0]
    assert saved["path"].parent == tmp_path / "data"
    assert saved["rows"] == len(df)
    assert dataset.queue_stats == {"awaiting_label": 3, "ready_for_merge": 1}


def test_train_models_resume_online(monkeypatch, tmp_path: Path, minimal_cfg: DummyConfig) -> None:
    X = pd.DataFrame({"feature_a": [0.1, 0.2, 0.3, 0.4]})
    y = pd.DataFrame({"direction_1": [0, 1, 0, 1]})
    class _QueueStub:
        def __len__(self) -> int:
            return 0

    dataset = pipeline.DatasetBuildResult(
        df=pd.concat([X, y], axis=1),
        X=X,
        y=y,
        features=["feature_a"],
        label_cols=["direction_1"],
        abs_label_cols=[],
        vol_label_cols=[],
        groups=pd.Series(["EURUSD"] * len(X)),
        timestamps=np.arange(len(X)),
        al_queue=_QueueStub(),
        al_threshold=0.6,
        queue_stats={"awaiting_label": 0, "ready_for_merge": 0},
        risk_budget=None,
        user_budget=None,
    )

    class StubSanitizer:
        def __init__(self) -> None:
            self._fitted = False

        def fit(self, X_in):
            self._fitted = True
            return self

        def transform(self, X_in):
            return X_in

    class StubClassifier:
        def __init__(self, **_kwargs) -> None:
            self.booster_ = None
            self.fit_calls: list[tuple[int, int]] = []

        def fit(self, X_in, y_in, **_kwargs):
            self.booster_ = "trained"
            self.fit_calls.append((len(X_in), len(y_in)))
            return self

    saved_batches: list[int] = []

    monkeypatch.setattr(pipeline, "_make_sanitizer", lambda cfg: StubSanitizer())
    monkeypatch.setattr(pipeline, "FeatureScaler", SimpleNamespace(load=lambda path: None))
    monkeypatch.setattr(pipeline, "dq_score_samples", lambda X_in: [1.0] * len(X_in))
    monkeypatch.setattr(pipeline, "combined_sample_weight", lambda *a, **k: None)
    monkeypatch.setattr(pipeline, "save_checkpoint", lambda state, batch_idx, _: saved_batches.append(batch_idx))
    monkeypatch.setattr(pipeline, "joblib", SimpleNamespace(dump=lambda *a, **k: None))
    monkeypatch.setattr(pipeline.monitor, "capabilities", SimpleNamespace(cpus=1))
    monkeypatch.setattr(pipeline, "LGBMClassifier", StubClassifier)
    monkeypatch.setattr(pipeline, "load_latest_checkpoint", lambda *_a, **_k: None)

    drift = SimpleNamespace(update=lambda *a, **k: None)
    result = pipeline.train_models(
        dataset,
        minimal_cfg,
        seed=7,
        model_type="lgbm",
        use_multi_task_heads=False,
        drift_monitor=drift,
        donor_booster=None,
        use_focal=False,
        fobj=None,
        feval=None,
        scaler_path=tmp_path / "scaler.pkl",
        root=tmp_path,
        risk_target=None,
        recorder=SimpleNamespace(),
        resume_online=True,
    )

    assert isinstance(result, pipeline.TrainingResult)
    assert result.final_pipe is not None
    clf = result.final_pipe.named_steps["clf"]
    assert isinstance(clf, StubClassifier)
    assert clf.booster_ == "trained"
    assert saved_batches == [0, 1]
    assert result.should_log_artifacts is False


def test_log_artifacts_handles_meta_and_pseudo(monkeypatch, tmp_path: Path, minimal_cfg: DummyConfig) -> None:
    features = ["feature_a"]

    class DummyPipe:
        def __init__(self) -> None:
            self.named_steps = {"clf": object()}

        def predict_proba(self, X):
            return [[0.1, 0.9] for _ in range(len(X))]

    class DummyQueue:
        def __init__(self) -> None:
            self.stats_called = False

        def push_low_confidence(self, ids, probs, threshold):
            assert threshold == pytest.approx(0.6)
            return len(list(ids))

        def stats(self):
            self.stats_called = True
            return {"awaiting_label": 2, "ready_for_merge": 1}

        def __len__(self) -> int:
            return 0

    training = pipeline.TrainingResult(
        final_pipe=DummyPipe(),
        features=features,
        aggregate_report={"weighted avg": {"f1-score": 0.5}},
        boot_metrics={"f1": 0.5, "precision": 0.6, "recall": 0.7},
        base_f1=0.5,
        regime_thresholds={0: 0.5},
        overall_params=None,
        overall_q={},
        overall_cov=0.0,
        all_probs=[0.9, 0.8],
        all_conf=[0.1, 0.2],
        all_true=[1, 0],
        al_queue=DummyQueue(),
        al_threshold=0.6,
        df=pd.DataFrame({"feature_a": [0.1, 0.2]}),
        X=pd.DataFrame({"feature_a": [0.1, 0.2]}),
        X_train_final=None,
        risk_budget=None,
        model_metadata={},
        f1_ci=(0.4, 0.6),
        prec_ci=(0.5, 0.7),
        rec_ci=(0.3, 0.8),
        final_score=0.5,
        should_log_artifacts=True,
    )

    class RecorderStub:
        def __init__(self) -> None:
            self.artifacts: list[tuple[Path, dict[str, Any]]] = []

        def add_artifact(self, path, **kwargs):
            self.artifacts.append((Path(path), kwargs))

    card_md = tmp_path / "card.md"
    card_md.write_text("# card")
    card_json = tmp_path / "card.json"
    card_json.write_text("{}")

    monkeypatch.setattr(pipeline, "train_meta_classifier", lambda *a, **k: "meta-model")
    monkeypatch.setattr(pipeline.model_store, "save_model", lambda *a, **k: "meta-1")
    monkeypatch.setattr(
        pipeline,
        "persist_model",
        lambda final_pipe, cfg, performance, *, features, root: "model-1",
    )
    monkeypatch.setattr(pipeline.mlflow, "log_metric", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.mlflow, "log_artifact", lambda *a, **k: None)
    monkeypatch.setattr(pipeline.model_card, "generate", lambda *a, **k: (card_md, card_json))

    start_time = datetime.now() - timedelta(seconds=1)
    recorder = RecorderStub()

    result = pipeline.log_artifacts(
        training,
        minimal_cfg,
        tmp_path,
        export=False,
        start_time=start_time,
        recorder=recorder,
        df_unlabeled=None,
    )

    assert isinstance(result, pipeline.ArtifactLogResult)
    assert result.meta_model_id == "meta-1"
    assert result.model_version_id == "model-1"
    assert result.pseudo_label_path is None
    assert result.queued_count == 0
    assert training.model_metadata["meta_model_id"] == "meta-1"
    assert recorder.artifacts  # ensure artifacts were registered

