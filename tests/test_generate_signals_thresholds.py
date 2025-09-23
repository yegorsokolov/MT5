import json
import sys
import types
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

def _make_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _identity(func):
    return func


LOG_CALLS: list[tuple[tuple[object, ...], dict[str, object]]] = []


def _setup_logging(*args, **kwargs):
    LOG_CALLS.append((args, kwargs))
    return logging.getLogger("test_generate_signals_thresholds")


log_utils_stub = _make_module(
    "log_utils",
    setup_logging=_setup_logging,
    log_predictions=lambda *args, **kwargs: None,
    log_exceptions=_identity,
)
log_utils_stub.LOG_DIR = Path.cwd() / "logs"
sys.modules["log_utils"] = log_utils_stub
sys.modules.setdefault("mt5.log_utils", log_utils_stub)

state_manager_stub = _make_module(
    "state_manager",
    load_runtime_state=lambda *args, **kwargs: None,
    migrate_runtime_state=lambda *args, **kwargs: None,
    save_runtime_state=lambda *args, **kwargs: None,
    legacy_runtime_state_exists=lambda *args, **kwargs: False,
)
sys.modules.setdefault("state_manager", state_manager_stub)

prediction_cache_stub = _make_module(
    "prediction_cache",
    PredictionCache=type(
        "_PredictionCache",
        (),
        {
            "__init__": lambda self, *a, **k: setattr(self, "maxsize", 0),
            "get": lambda self, *a, **k: None,
            "set": lambda self, *a, **k: None,
        },
    ),
)
sys.modules.setdefault("prediction_cache", prediction_cache_stub)

utils_stub = _make_module("utils", load_config=lambda *a, **k: {})
sys.modules["utils"] = utils_stub
sys.modules["utils.market_hours"] = _make_module(
    "utils.market_hours", is_market_open=lambda: True
)

sys.modules["backtest"] = _make_module(
    "backtest", run_rolling_backtest=lambda *a, **k: None
)
sys.modules["mt5.backtest"] = sys.modules["backtest"]

river_stub = _make_module("river", compose=_make_module("compose"))
sys.modules["river"] = river_stub
sys.modules["river.compose"] = river_stub.compose

data_history_stub = _make_module(
    "data.history",
    load_history_parquet=lambda *a, **k: None,
    load_history_config=lambda *a, **k: None,
)
sys.modules["data.history"] = data_history_stub

data_features_stub = _make_module(
    "data.features",
    make_features=lambda df, *a, **k: df,
    make_sequence_arrays=lambda df, cols, seq_len: (np.zeros((0, 0)), None),
)
sys.modules["data.features"] = data_features_stub

train_rl_stub = _make_module(
    "train_rl",
    TradingEnv=object,
    DiscreteTradingEnv=object,
    RLLibTradingEnv=object,
    HierarchicalTradingEnv=object,
)
sys.modules.setdefault("train_rl", train_rl_stub)

stable_stub = _make_module("stable_baselines3", PPO=object, SAC=object, A2C=object)
sys.modules.setdefault("stable_baselines3", stable_stub)

sb3_contrib_stub = _make_module(
    "sb3_contrib",
    qrdqn=_make_module("sb3_contrib.qrdqn", QRDQN=object),
    TRPO=object,
    RecurrentPPO=object,
)
sys.modules.setdefault("sb3_contrib", sb3_contrib_stub)
sys.modules.setdefault("sb3_contrib.qrdqn", sb3_contrib_stub.qrdqn)

signal_queue_stub = _make_module(
    "signal_queue",
    publish_dataframe_async=lambda *a, **k: None,
    get_signal_backend=lambda *a, **k: None,
)
sys.modules.setdefault("signal_queue", signal_queue_stub)

sys.modules.setdefault("models.ensemble", _make_module("models.ensemble", EnsembleModel=object))

model_store_stub = _make_module(
    "models.model_store",
    load_model=lambda *a, **k: (object(), {"performance": {}}),
    list_versions=lambda *a, **k: [],
)
sys.modules.setdefault("models.model_store", model_store_stub)
sys.modules.setdefault("models", _make_module("models", model_store=model_store_stub))

concept_drift_stub = _make_module("analysis.concept_drift", ConceptDriftMonitor=object)
sys.modules.setdefault("analysis.concept_drift", concept_drift_stub)


class _StubSampler:
    def __init__(self, *args, **kwargs):  # pragma: no cover - simple stub
        pass


class _StubStudy:
    def __init__(self, *args, **kwargs):  # pragma: no cover - simple stub
        self.best_params: dict[str, float] = {}
        self.best_value: float = 0.0

    def optimize(self, objective, n_trials):  # pragma: no cover - simple stub
        return None


optuna_stub = _make_module(
    "optuna",
    samplers=_make_module("samplers", TPESampler=_StubSampler),
    create_study=lambda *a, **k: _StubStudy(),
    Study=_StubStudy,
)
sys.modules.setdefault("optuna", optuna_stub)


class _StubLGBMClassifier:
    def __init__(self, **kwargs):
        from sklearn.linear_model import LogisticRegression  # type: ignore

        self._model = LogisticRegression(max_iter=1000)

    def fit(self, X, y):  # pragma: no cover - simple wrapper
        self._model.fit(X, y)
        return self

    def predict_proba(self, X):  # pragma: no cover - simple wrapper
        return self._model.predict_proba(X)


sys.modules.setdefault(
    "lightgbm",
    _make_module("lightgbm", LGBMClassifier=_StubLGBMClassifier),
)

conformal_stub = _make_module(
    "models.conformal",
    ConformalIntervalParams=object,
    evaluate_coverage=lambda *a, **k: 0.0,
    predict_interval=lambda *a, **k: (np.zeros(0), np.zeros(0)),
)
sys.modules.setdefault("models.conformal", conformal_stub)

from mt5 import generate_signals  # type: ignore  # noqa: E402

assert not LOG_CALLS


def test_resolve_cache_dir_defaults_to_logs(tmp_path, monkeypatch):
    monkeypatch.delenv("MT5_CACHE_DIR", raising=False)
    monkeypatch.setattr(generate_signals.log_utils, "LOG_DIR", tmp_path, raising=False)
    result = generate_signals._resolve_cache_dir({})
    assert result == tmp_path / "cache"


def test_resolve_cache_dir_honours_env_override(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_signals.log_utils, "LOG_DIR", tmp_path, raising=False)
    override = tmp_path / "alt"
    monkeypatch.setenv("MT5_CACHE_DIR", str(override))
    result = generate_signals._resolve_cache_dir({})
    assert result == override


def test_resolve_cache_dir_prefers_config_over_env(tmp_path, monkeypatch):
    monkeypatch.setattr(generate_signals.log_utils, "LOG_DIR", tmp_path, raising=False)
    monkeypatch.setenv("MT5_CACHE_DIR", str(tmp_path / "ignored"))
    cfg_override = tmp_path / "cfg"
    result = generate_signals._resolve_cache_dir({"cache_dir": str(cfg_override)})
    assert result == cfg_override


def test_init_logging_configures_once():
    LOG_CALLS.clear()
    first_logger = generate_signals.init_logging()
    assert len(LOG_CALLS) == 1
    second_logger = generate_signals.init_logging()
    assert len(LOG_CALLS) == 1
    assert first_logger is second_logger
    LOG_CALLS.clear()


class _DummyModel:
    def predict_proba(self, X):  # pragma: no cover - helper stub
        arr = np.asarray(X, dtype=float).reshape(-1, 1)
        return np.hstack([1 - arr, arr])


def test_per_symbol_thresholds_applied(tmp_path):
    model_path = tmp_path / "eurusd_model.joblib"
    dummy = _DummyModel()
    joblib.dump(dummy, model_path)

    metadata = {
        "features": ["f0"],
        "performance": {
            "regime_thresholds": {"0": 0.8},
            "best_threshold": 0.8,
            "validation_f1": 0.72,
        },
    }
    meta_path = model_path.with_name(f"{model_path.stem}_metadata.json")
    meta_path.write_text(json.dumps(metadata))

    models, features = generate_signals.load_models([str(model_path)], [])
    assert features == metadata["features"]
    assert getattr(models[0], "model_metadata") == metadata

    thresholds, default_thr = generate_signals._normalise_thresholds(
        getattr(models[0], "regime_thresholds", {})
    )
    assert thresholds == {0: 0.8}
    assert default_thr is None

    probs = np.array([0.55, 0.79, 0.81])
    regimes = np.array([0, 0, 0])
    preds = generate_signals.apply_regime_thresholds(probs, regimes, thresholds, 0.5)

    baseline = (probs >= 0.5).astype(int)
    assert preds.tolist() == [0, 0, 1]
    assert preds.tolist() != baseline.tolist()


def test_online_threshold_overrides_default_cutoff():
    online_model = types.SimpleNamespace(
        best_threshold_=0.8,
        regime_thresholds={0: 0.8},
    )
    cfg = {"threshold": 0.5}
    thresholds, default_thr = generate_signals._resolve_threshold_metadata(
        [], online_model, cfg
    )
    assert np.isclose(default_thr, 0.8)
    probs = np.array([0.55, 0.79, 0.81])
    regimes = np.array([0, 0, 0])
    if thresholds:
        preds = generate_signals.apply_regime_thresholds(
            probs, regimes, thresholds, default_thr
        )
    else:
        preds = (probs >= default_thr).astype(int)
    assert preds.tolist() == [0, 0, 1]


def test_online_regime_thresholds_applied():
    online_model = types.SimpleNamespace(
        best_threshold_=0.6,
        regime_thresholds={0: 0.55, 1: 0.75},
    )
    cfg = {"threshold": 0.5}
    thresholds, default_thr = generate_signals._resolve_threshold_metadata(
        [], online_model, cfg
    )
    probs = np.array([0.56, 0.74, 0.76])
    regimes = np.array([0, 1, 1])
    preds = generate_signals.apply_regime_thresholds(
        probs, regimes, thresholds, default_thr
    )
    assert preds.tolist() == [1, 0, 1]


def test_regime_thresholds_fallback_preserves_import_state():
    import importlib

    module_name = "analysis.regime_thresholds"
    original_module = sys.modules.pop(module_name, None)
    original_sklearn = sys.modules.get("sklearn")
    original_metrics = sys.modules.get("sklearn.metrics")
    original_scipy = sys.modules.get("scipy")
    original_scipy_stats = sys.modules.get("scipy.stats")

    created_sklearn_stub = False
    if original_sklearn is None:
        sklearn_stub = types.ModuleType("sklearn")
        sklearn_stub.__path__ = []  # type: ignore[attr-defined]
        sys.modules["sklearn"] = sklearn_stub
        created_sklearn_stub = True

    broken_metrics = types.ModuleType("sklearn.metrics")

    def _missing_attr(name):  # pragma: no cover - exercised in tests
        raise ImportError("precision_recall_curve requires SciPy")

    broken_metrics.__getattr__ = _missing_attr  # type: ignore[attr-defined]
    sys.modules["sklearn.metrics"] = broken_metrics
    if created_sklearn_stub:
        sys.modules["sklearn"].metrics = broken_metrics  # type: ignore[attr-defined]

    sentinel_scipy = types.ModuleType("scipy")
    sentinel_scipy_stats = types.ModuleType("scipy.stats")
    sys.modules["scipy"] = sentinel_scipy
    sys.modules["scipy.stats"] = sentinel_scipy_stats

    try:
        regime_thresholds = importlib.import_module(module_name)
        pr_curve = regime_thresholds.find_regime_thresholds.__globals__["precision_recall_curve"]
        assert pr_curve.__module__ == module_name

        y = np.array([0, 1, 1, 0, 0, 1])
        probs = np.array([0.1, 0.8, 0.9, 0.2, 0.6, 0.7])
        regimes = np.array([0, 0, 0, 1, 1, 1])

        thresholds, preds = regime_thresholds.find_regime_thresholds(y, probs, regimes)

        assert np.isclose(thresholds[0], 0.8)
        assert np.isclose(thresholds[1], 0.7)
        np.testing.assert_array_equal(preds, np.array([0, 1, 1, 0, 0, 1]))

        assert sys.modules.get("scipy") is sentinel_scipy
        assert sys.modules.get("scipy.stats") is sentinel_scipy_stats
        assert sys.modules.get("sklearn.metrics") is broken_metrics
    finally:
        sys.modules.pop(module_name, None)
        if original_module is not None:
            sys.modules[module_name] = original_module

        if original_metrics is not None:
            sys.modules["sklearn.metrics"] = original_metrics
        else:
            sys.modules.pop("sklearn.metrics", None)

        if created_sklearn_stub:
            sys.modules.pop("sklearn", None)
        elif original_sklearn is not None:
            sys.modules["sklearn"] = original_sklearn

        if original_scipy is not None:
            sys.modules["scipy"] = original_scipy
        else:
            sys.modules.pop("scipy", None)

        if original_scipy_stats is not None:
            sys.modules["scipy.stats"] = original_scipy_stats
        else:
            sys.modules.pop("scipy.stats", None)


def test_training_regime_thresholds_align_with_application():
    import importlib
    import sys

    for mod in ["scipy", "scipy.sparse", "scipy.stats"]:
        sys.modules.pop(mod, None)
    scipy = importlib.import_module("scipy")
    sys.modules["scipy"] = scipy
    sys.modules["scipy.sparse"] = importlib.import_module("scipy.sparse")
    sys.modules["scipy.stats"] = importlib.import_module("scipy.stats")
    from mt5 import train_parallel
    import warnings

    warnings.filterwarnings(
        "ignore",
        message=".*__sklearn_tags__.*",
        category=DeprecationWarning,
    )
    warnings.filterwarnings(
        "ignore",
        message=".*Pipeline instance is not fitted yet.*",
        category=FutureWarning,
    )

    rng = np.random.default_rng(123)
    reg0_train = rng.normal(loc=-0.2, scale=0.5, size=80)
    reg1_train = rng.normal(loc=0.9, scale=0.5, size=80)
    train_features = np.concatenate([reg0_train, reg1_train])
    train_labels = np.concatenate([
        (reg0_train > 0.0).astype(int),
        (reg1_train > 1.1).astype(int),
    ])

    reg0_val = rng.normal(loc=-0.1, scale=0.45, size=40)
    reg1_val = rng.normal(loc=0.85, scale=0.4, size=40)
    val_features = np.concatenate([reg0_val, reg1_val])
    val_labels = np.concatenate([
        (reg0_val > 0.1).astype(int),
        (reg1_val > 0.95).astype(int),
    ])
    val_regimes = np.concatenate(
        [np.zeros_like(reg0_val, dtype=int), np.ones_like(reg1_val, dtype=int)]
    )

    train_df = pd.DataFrame({"f0": train_features, "tb_label": train_labels})
    val_df = pd.DataFrame(
        {"f0": val_features, "tb_label": val_labels, "market_regime": val_regimes}
    )

    pipeline = train_parallel.build_lightgbm_pipeline(
        train_parallel.DEFAULT_LGBM_PARAMS,
        use_scaler=False,
    )
    pipeline.fit(train_df[["f0"]], train_df["tb_label"])
    val_probs = pipeline.predict_proba(val_df[["f0"]])[:, 1]

    thresholds, preds, val_f1 = train_parallel._derive_regime_thresholds(
        val_df, val_probs
    )

    assert "default" in thresholds
    assert 0 in thresholds and 1 in thresholds
    default_thr = thresholds["default"]

    applied_preds = generate_signals.apply_regime_thresholds(
        val_probs,
        val_regimes,
        thresholds,
        default_thr,
    )
    np.testing.assert_array_equal(applied_preds, preds)

    metadata = {
        "performance": {
            "regime_thresholds": {str(k): float(v) for k, v in thresholds.items()},
            "validation_f1": float(val_f1),
            "best_threshold": float(default_thr),
        }
    }
    norm_map, norm_default = generate_signals._normalise_thresholds(
        metadata["performance"]["regime_thresholds"]
    )
    assert norm_map == {0: thresholds[0], 1: thresholds[1]}
    assert np.isclose(norm_default, default_thr)

    applied_from_meta = generate_signals.apply_regime_thresholds(
        val_probs,
        val_regimes,
        metadata["performance"]["regime_thresholds"],
        default_thr,
    )
    np.testing.assert_array_equal(applied_from_meta, preds)
