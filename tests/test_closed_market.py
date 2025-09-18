import sys
import types
import builtins
import importlib
from pathlib import Path

import pytest


@pytest.mark.skip(reason="requires heavy optional dependencies")
def test_generate_signals_closed_market(monkeypatch):
    # Ensure repository root on path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    calls = {}

    class DummyAlgo:
        @classmethod
        def load(cls, *args, **kwargs):
            return cls()

    # Stub out heavy optional dependencies before importing module
    sys.modules['stable_baselines3'] = types.SimpleNamespace(PPO=DummyAlgo, SAC=DummyAlgo, A2C=DummyAlgo)
    sb3_contrib = types.ModuleType("sb3_contrib")
    sb3_contrib.TRPO = DummyAlgo
    sb3_contrib.RecurrentPPO = DummyAlgo
    sb3_contrib.qrdqn = types.SimpleNamespace(QRDQN=DummyAlgo)
    sys.modules['sb3_contrib'] = sb3_contrib
    sys.modules['sb3_contrib.qrdqn'] = sb3_contrib.qrdqn
    sys.modules['river'] = types.SimpleNamespace(compose=types.SimpleNamespace())
    sys.modules['train_rl'] = types.SimpleNamespace(
        TradingEnv=object, DiscreteTradingEnv=object, RLLibTradingEnv=object, HierarchicalTradingEnv=object
    )
    sys.modules['signal_queue'] = types.SimpleNamespace(
        get_async_publisher=lambda *a, **k: None,
        publish_dataframe_async=lambda *a, **k: None,
    )

    def fake_backtest(cfg):
        calls['called'] = True

    sys.modules['backtest'] = types.SimpleNamespace(run_rolling_backtest=fake_backtest)

    import generate_signals

    monkeypatch.setattr(generate_signals, 'is_market_open', lambda: False)
    monkeypatch.setattr(generate_signals, 'load_config', lambda: {})

    def stop(*args, **kwargs):
        raise SystemExit

    monkeypatch.setattr(generate_signals, 'load_models', stop)
    monkeypatch.setattr(sys, 'argv', ['generate_signals.py'])

    with pytest.raises(SystemExit):
        generate_signals.main()

    assert calls.get('called')


def test_generate_signals_supervised_without_rl_dependencies(monkeypatch):
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("stable_baselines3") or name.startswith("sb3_contrib"):
            raise ModuleNotFoundError(f"mocked missing optional dependency: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.delitem(sys.modules, "generate_signals", raising=False)

    def _stub_module(name: str, **attrs):
        module = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(module, key, value)
        module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    class _DummySession:
        def post(self, *args, **kwargs):
            return None

    monkeypatch.setitem(
        sys.modules,
        "requests",
        types.SimpleNamespace(Session=lambda *a, **k: _DummySession(), head=lambda *a, **k: None),
    )

    log_utils_stub = types.ModuleType("log_utils")
    log_utils_stub.setup_logging = lambda *a, **k: None

    def _passthrough(func):
        return func

    log_utils_stub.log_exceptions = _passthrough
    log_utils_stub.log_predictions = lambda *a, **k: None
    log_utils_stub.__spec__ = importlib.machinery.ModuleSpec("log_utils", loader=None)
    monkeypatch.setitem(sys.modules, "log_utils", log_utils_stub)

    _stub_module(
        "state_manager",
        load_runtime_state=lambda *a, **k: None,
        migrate_runtime_state=lambda *a, **k: Path("state.json"),
        save_runtime_state=lambda *a, **k: None,
        legacy_runtime_state_exists=lambda: False,
    )

    class _DummyCache:
        def __init__(self, *args, **kwargs):
            pass

    _stub_module("prediction_cache", PredictionCache=_DummyCache)

    utils_stub = _stub_module("utils", load_config=lambda *a, **k: {})
    utils_stub.market_hours = types.SimpleNamespace(is_market_open=lambda *a, **k: True)
    utils_stub.resource_monitor = types.SimpleNamespace(
        monitor=types.SimpleNamespace(capabilities=types.SimpleNamespace(capability_tier=lambda: "lite"))
    )
    monkeypatch.setitem(sys.modules, "utils.market_hours", utils_stub.market_hours)
    monkeypatch.setitem(sys.modules, "utils.resource_monitor", utils_stub.resource_monitor)

    data_pkg = _stub_module("data")
    data_pkg.__path__ = []
    _stub_module("data.history", load_history_parquet=lambda *a, **k: pd.DataFrame(), load_history_config=lambda *a, **k: {})
    _stub_module(
        "data.features",
        make_features=lambda df: df,
        make_sequence_arrays=lambda *_a, **_k: (np.empty((0,)), None),
    )

    river_pkg = _stub_module("river", compose=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "river.compose", river_pkg.compose)

    _stub_module("backtest", run_rolling_backtest=lambda *a, **k: None)

    _stub_module(
        "signal_queue",
        publish_dataframe_async=lambda *a, **k: None,
        get_signal_backend=lambda *a, **k: None,
    )

    _stub_module("models.ensemble", EnsembleModel=object)
    models_pkg = _stub_module("models", model_store=types.SimpleNamespace(load_model=lambda *a, **k: (object(), {})))
    models_pkg.__path__ = []
    _stub_module(
        "models.conformal",
        ConformalIntervalParams=lambda *a, **k: types.SimpleNamespace(
            quantiles=None, coverage=None, alpha=0.1, coverage_by_regime=None
        ),
        evaluate_coverage=lambda *a, **k: {},
        predict_interval=lambda *a, **k: {},
    )

    _stub_module("analysis.concept_drift", ConceptDriftMonitor=lambda *a, **k: types.SimpleNamespace())

    class _DummyEnv:
        def __init__(self, *args, **kwargs):
            self._done = True

        def reset(self):
            return np.zeros(1), {}

        def step(self, action):
            return np.zeros(1), 0.0, True, {}

    _stub_module(
        "train_rl",
        TradingEnv=_DummyEnv,
        DiscreteTradingEnv=_DummyEnv,
        RLLibTradingEnv=_DummyEnv,
        HierarchicalTradingEnv=_DummyEnv,
    )

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    generate_signals = importlib.import_module("generate_signals")

    df = pd.DataFrame({"f1": [0.1, 0.2, 0.3], "f2": [0.4, 0.5, 0.6]})

    class DummyEstimator:
        def predict_regression(self, data, head_name):
            assert head_name == "alpha"
            return np.full(len(data), 0.5)

    class DummyModel:
        regression_heads_ = {"alpha": {}}
        regression_feature_columns_ = ["f1", "f2"]

        def __init__(self):
            self.regression_estimator_ = DummyEstimator()

    predictions = generate_signals.compute_regression_estimates(
        [DummyModel()], df, ["f1", "f2"]
    )

    assert set(predictions) == {"alpha"}
    assert predictions["alpha"].shape == (len(df),)
    assert np.allclose(predictions["alpha"], 0.5)
