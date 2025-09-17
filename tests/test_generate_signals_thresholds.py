import json
import sys
import types
from pathlib import Path

import joblib
import numpy as np

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))

def _make_module(name: str, **attrs):
    mod = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


def _identity(func):
    return func


log_utils_stub = _make_module(
    "log_utils",
    setup_logging=lambda *args, **kwargs: None,
    log_predictions=lambda *args, **kwargs: None,
    log_exceptions=_identity,
)
sys.modules["log_utils"] = log_utils_stub

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

for mod in ["scipy", "scipy.stats", "scipy.sparse"]:
    sys.modules.pop(mod, None)

sys.modules["backtest"] = _make_module(
    "backtest", run_rolling_backtest=lambda *a, **k: None
)

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

conformal_stub = _make_module(
    "models.conformal",
    ConformalIntervalParams=object,
    evaluate_coverage=lambda *a, **k: 0.0,
    predict_interval=lambda *a, **k: (np.zeros(0), np.zeros(0)),
)
sys.modules.setdefault("models.conformal", conformal_stub)

import generate_signals  # type: ignore  # noqa: E402


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

    thresholds = generate_signals._normalise_thresholds(
        getattr(models[0], "regime_thresholds", {})
    )
    assert thresholds == {0: 0.8}

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
