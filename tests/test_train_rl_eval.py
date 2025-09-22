import types
import sys
import logging

import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np


def test_eval_rl_compute_metrics_constant_returns(monkeypatch):
    sb3_stub = types.ModuleType("stable_baselines3")
    sb3_stub.__path__ = []
    sb3_stub.PPO = object
    sb3_stub.SAC = object
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3_stub)
    sb3_contrib_stub = types.ModuleType("sb3_contrib")
    sb3_contrib_stub.__path__ = []
    sb3_contrib_qrdqn = types.ModuleType("sb3_contrib.qrdqn")
    sb3_contrib_qrdqn.QRDQN = object
    sb3_contrib_stub.qrdqn = sb3_contrib_qrdqn
    monkeypatch.setitem(sys.modules, "sb3_contrib", sb3_contrib_stub)
    monkeypatch.setitem(sys.modules, "sb3_contrib.qrdqn", sb3_contrib_qrdqn)
    utils_stub = types.ModuleType("utils")
    utils_stub.load_config = lambda: {}
    monkeypatch.setitem(sys.modules, "utils", utils_stub)
    data_pkg = types.ModuleType("data")
    data_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "data", data_pkg)
    sys.modules.pop("features", None)
    features_stub_module = types.ModuleType("features")
    features_stub_module.make_features = lambda df: df
    monkeypatch.setitem(sys.modules, "features", features_stub_module)
    data_history_stub = types.ModuleType("data.history")
    data_history_stub.load_history_parquet = lambda *a, **k: None
    data_history_stub.save_history_parquet = lambda *a, **k: None
    data_history_stub.load_history_config = lambda *a, **k: pd.DataFrame()
    monkeypatch.setitem(sys.modules, "data.history", data_history_stub)
    data_features_stub = types.ModuleType("data.features")
    data_features_stub.make_features = lambda df: df
    monkeypatch.setitem(sys.modules, "data.features", data_features_stub)
    setattr(data_pkg, "features", data_features_stub)
    train_rl_stub = types.ModuleType("train_rl")
    train_rl_stub.TradingEnv = object
    train_rl_stub.DiscreteTradingEnv = object
    monkeypatch.setitem(sys.modules, "train_rl", train_rl_stub)
    rl_stub = types.ModuleType("rl")
    rl_multi = types.ModuleType("rl.multi_objective")
    rl_multi.pareto_frontier = lambda *a, **k: []
    rl_stub.multi_objective = rl_multi
    rl_stub.__path__ = []
    monkeypatch.setitem(sys.modules, "rl", rl_stub)
    monkeypatch.setitem(sys.modules, "rl.multi_objective", rl_multi)
    log_calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
    test_logger = logging.getLogger("test_eval_rl")
    log_utils_stub = types.ModuleType("log_utils")

    def _setup_logging(*args, **kwargs):
        log_calls.append((args, kwargs))
        return test_logger

    log_utils_stub.setup_logging = _setup_logging
    log_utils_stub.log_exceptions = lambda f: f
    monkeypatch.setitem(sys.modules, "log_utils", log_utils_stub)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    spec = importlib.util.spec_from_file_location(
        "eval_rl", Path(__file__).resolve().parents[1] / "eval_rl.py"
    )
    eval_rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_rl)

    assert not log_calls

    returns = pd.Series([0.0] * 10)
    metrics = eval_rl.compute_metrics(returns)
    assert metrics["sharpe"] == 0.0
    assert all(np.isfinite(list(metrics.values())))


def test_rl_evaluation_metrics(monkeypatch, tmp_path):
    metrics_logged = {}
    mlflow_stub = types.SimpleNamespace(
        set_tracking_uri=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        start_run=lambda *a, **k: None,
        log_param=lambda *a, **k: None,
        log_artifact=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
        log_metric=lambda k, v, **kw: metrics_logged.setdefault(k, v),
        __spec__=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)
    prom_stub = types.SimpleNamespace(
        Counter=lambda *a, **k: types.SimpleNamespace(inc=lambda *a, **k: None),
        Gauge=lambda *a, **k: types.SimpleNamespace(set=lambda *a, **k: None),
        __spec__=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "prometheus_client", prom_stub)
    ray_utils_stub = types.SimpleNamespace(
        init=lambda **k: None,
        shutdown=lambda: None,
        cluster_available=lambda: False,
        submit=lambda *a, **k: None,
        ray=types.SimpleNamespace(remote=lambda f: f),
    )
    monkeypatch.setitem(sys.modules, "ray_utils", ray_utils_stub)
    class _TensorDataset:
        def __init__(self, *arrays):
            self.data = list(zip(*arrays))

        def __iter__(self):
            return iter(self.data)

        def __len__(self):
            return len(self.data)

    class _DataLoader:
        def __init__(self, dataset, batch_size=32, shuffle=False):
            self.dataset = dataset

        def __iter__(self):
            return iter(self.dataset)

    torch_utils_data_stub = types.SimpleNamespace(DataLoader=_DataLoader, TensorDataset=_TensorDataset)
    torch_utils_stub = types.SimpleNamespace(data=torch_utils_data_stub)
    torch_nn_stub = types.SimpleNamespace(Module=object, functional=types.SimpleNamespace())
    torch_stub = types.SimpleNamespace(
        manual_seed=lambda *a, **k: None,
        cuda=types.SimpleNamespace(is_available=lambda: False, manual_seed_all=lambda *a, **k: None),
        nn=torch_nn_stub,
        tensor=lambda data, dtype=None: np.array(data),
        float32=np.float32,
        zeros=lambda *a, **k: np.zeros(*a, **k),
    )
    monkeypatch.setitem(sys.modules, "torch", torch_stub)
    monkeypatch.setitem(sys.modules, "torch.nn", torch_nn_stub)
    monkeypatch.setitem(sys.modules, "torch.nn.functional", torch_nn_stub.functional)
    monkeypatch.setitem(sys.modules, "torch.utils", torch_utils_stub)
    monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data_stub)
    class DummyBox:
        def __init__(self, low, high, shape=None, dtype=None):
            self.shape = shape if shape is not None else getattr(low, "shape", (len(low),))

    gym_stub = types.SimpleNamespace(
        Env=object, spaces=types.SimpleNamespace(Box=DummyBox), Wrapper=object
    )
    monkeypatch.setitem(sys.modules, "gym", gym_stub)
    monkeypatch.setitem(sys.modules, "utils", types.SimpleNamespace(load_config=lambda: {}))
    monkeypatch.setitem(
        sys.modules,
        "utils.lr_scheduler",
        types.SimpleNamespace(
            LookaheadAdamW=lambda *a, **k: types.SimpleNamespace(
                state_dict=lambda: {}, get_lr=lambda: 0
            )
        ),
    )
    monitor_stub = types.SimpleNamespace(
        start=lambda: None,
        capability_tier=lambda: "lite",
        capabilities=types.SimpleNamespace(capability_tier=lambda: "lite", ddp=lambda: False),
        create_task=lambda coro: None,
    )
    monkeypatch.setitem(
        sys.modules, "utils.resource_monitor", types.SimpleNamespace(monitor=monitor_stub)
    )
    monkeypatch.setitem(sys.modules, "requests", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "joblib", types.SimpleNamespace())
    state_stub = types.SimpleNamespace(
        save_checkpoint=lambda *a, **k: None,
        load_latest_checkpoint=lambda *a, **k: None,
        load_router_state=lambda *a, **k: None,
        save_router_state=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "state_manager", state_stub)
    history_stub = types.SimpleNamespace(
        load_history_parquet=lambda *a, **k: None,
        save_history_parquet=lambda *a, **k: None,
        load_history_config=lambda *a, **k: pd.DataFrame(),
    )
    data_features_stub = types.ModuleType("data.features")
    data_features_stub.make_features = lambda df: df
    data_pkg = types.ModuleType("data")
    data_pkg.__path__ = []
    monkeypatch.setitem(sys.modules, "data", data_pkg)
    monkeypatch.setitem(sys.modules, "data.history", history_stub)
    monkeypatch.setitem(sys.modules, "data.features", data_features_stub)
    monkeypatch.setitem(
        sys.modules, "data.versioning", types.SimpleNamespace(compute_hash=lambda *a, **k: "0")
    )
    sys.modules.pop("features", None)
    features_module = types.ModuleType("features")
    features_module.make_features = lambda df: df
    monkeypatch.setitem(sys.modules, "features", features_module)
    metrics_stub = types.ModuleType("metrics_store")
    metrics_stub.record_metric = lambda *a, **k: None
    metrics_stub.TS_PATH = ""
    analytics_stub = types.ModuleType("analytics")
    analytics_stub.metrics_store = metrics_stub
    analytics_stub.__path__ = []  # mark as package
    sys.modules.pop("analytics.metrics_store", None)
    monkeypatch.setitem(sys.modules, "analytics", analytics_stub)
    monkeypatch.setitem(sys.modules, "analytics.metrics_store", metrics_stub)
    regime_store_stub = types.ModuleType("analytics.regime_performance_store")
    regime_store_stub.RegimePerformanceStore = object
    monkeypatch.setitem(sys.modules, "analytics.regime_performance_store", regime_store_stub)
    metrics_aggregator_stub = types.ModuleType("analytics.metrics_aggregator")
    metrics_aggregator_stub.record_metric = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "analytics.metrics_aggregator", metrics_aggregator_stub)
    model_store_stub = types.SimpleNamespace(save_model=lambda *a, **k: "0")
    graph_net_stub = types.SimpleNamespace(GraphNet=object)
    models_pkg = types.ModuleType("models")
    models_pkg.__path__ = []
    models_pkg.model_store = model_store_stub
    models_pkg.graph_net = graph_net_stub
    monkeypatch.setitem(sys.modules, "models", models_pkg)
    monkeypatch.setitem(sys.modules, "models.model_store", model_store_stub)
    monkeypatch.setitem(sys.modules, "models.graph_net", graph_net_stub)
    ftrl_stub = types.ModuleType("models.ftrl")
    ftrl_stub.FTRLModel = object
    monkeypatch.setitem(sys.modules, "models.ftrl", ftrl_stub)
    model_registry_stub = types.ModuleType("model_registry")
    model_registry_stub.register_policy = lambda *a, **k: None
    model_registry_stub.save_model = lambda *a, **k: None
    model_registry_stub.get_policy_path = lambda *a, **k: str(tmp_path / "policy.zip")
    monkeypatch.setitem(sys.modules, "model_registry", model_registry_stub)
    event_store_stub = types.ModuleType("event_store")
    event_store_stub.EventStore = object
    monkeypatch.setitem(sys.modules, "event_store", event_store_stub)
    class DummyRiskEnv:
        action_space = types.SimpleNamespace(shape=(1,))
        observation_space = types.SimpleNamespace(shape=(1,))

        def __init__(self, *a, **k):
            pass

        def reset(self):
            return np.zeros(1)

        def step(self, action):
            return np.zeros(1), 0.0, True, {}

    plugins_stub = types.SimpleNamespace(rl_risk=types.SimpleNamespace(RiskEnv=DummyRiskEnv))
    monkeypatch.setitem(sys.modules, "plugins", plugins_stub)
    monkeypatch.setitem(sys.modules, "plugins.rl_risk", plugins_stub.rl_risk)

    class DummyAlgo:
        def __init__(self, policy, env, verbose=0, seed=0, **kwargs):
            self.env = env
            self.policy = types.SimpleNamespace(
                parameters=lambda: [],
                state_dict=lambda: {},
                optimizer=types.SimpleNamespace(state_dict=lambda: {}, get_lr=lambda: 0),
            )

        def learn(self, total_timesteps, **kwargs):
            pass

        def predict(self, obs, deterministic=True):
            shape = getattr(self.env.action_space, "shape", (1,))
            if shape is None:
                shape = (1,)
            return np.zeros(shape), None

        def save(self, path):
            pass

    def eval_policy(model, env, n_eval_episodes=1, deterministic=True):
        obs = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, _, done, _ = env.step(action)

    sb3_common = types.SimpleNamespace(
        vec_env=types.SimpleNamespace(SubprocVecEnv=lambda *a, **k: None),
        evaluation=types.SimpleNamespace(evaluate_policy=eval_policy),
    )
    sb3_stub = types.SimpleNamespace(PPO=DummyAlgo, SAC=DummyAlgo, A2C=DummyAlgo, common=sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3_stub)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", sb3_common.vec_env)
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.evaluation", sb3_common.evaluation
    )
    sb3_contrib_stub = types.SimpleNamespace(
        TRPO=DummyAlgo,
        RecurrentPPO=DummyAlgo,
        qrdqn=types.SimpleNamespace(QRDQN=DummyAlgo),
    )
    monkeypatch.setitem(sys.modules, "sb3_contrib", sb3_contrib_stub)
    monkeypatch.setitem(sys.modules, "sb3_contrib.qrdqn", sb3_contrib_stub.qrdqn)

    market_sim_stub = types.ModuleType("analysis.market_simulator")

    class DummySimulator:
        def __init__(self, *args, **kwargs):
            pass

        def perturb(self, prices, policy, steps=10):
            return np.asarray(list(prices))

    market_sim_stub.AdversarialMarketSimulator = DummySimulator
    market_sim_stub.generate_stress_scenarios = lambda *a, **k: []
    monkeypatch.setitem(sys.modules, "analysis.market_simulator", market_sim_stub)
    regime_detection_stub = types.ModuleType("analysis.regime_detection")
    regime_detection_stub.periodic_reclassification = lambda df, **kwargs: df
    regime_detection_stub.detect_regimes = lambda df, **kwargs: pd.Series(0, index=df.index)
    monkeypatch.setitem(sys.modules, "analysis.regime_detection", regime_detection_stub)
    model_card_stub = types.ModuleType("analysis.model_card")
    model_card_stub.generate = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "analysis.model_card", model_card_stub)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    spec = importlib.util.spec_from_file_location(
        "train_rl", Path(__file__).resolve().parents[1] / "mt5" / "train_rl.py"
    )
    train_rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_rl)
    import rl.trading_env as trading_env
    monkeypatch.setattr(trading_env, "record_metric", lambda *a, **k: None)

    monkeypatch.setattr(train_rl, "__file__", tmp_path / "train_rl.py")
    cfg = {
        "seed": 123,
        "symbols": ["TEST"],
        "rl_algorithm": "PPO",
        "rl_steps": 10,
        "rl_transaction_cost": 0.0,
        "rl_risk_penalty": 0.0,
    }
    monkeypatch.setattr(train_rl, "load_config", lambda: cfg)

    n = 20
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=n, freq="H"),
            "Symbol": ["TEST"] * n,
            "mid": 100.0,
            "return": 0.0,
            "ma_5": 0.0,
            "ma_10": 0.0,
            "ma_30": 0.0,
            "ma_60": 0.0,
            "volatility_30": 0.0,
            "spread": 0.0,
            "rsi_14": 0.0,
            "news_sentiment": 0.0,
            "market_regime": 0.0,
        }
    )
    monkeypatch.setattr(train_rl, "load_history_config", lambda *a, **k: df)
    monkeypatch.setattr(train_rl, "make_features", lambda d: d)
    class DummyRiskEnv:
        action_space = types.SimpleNamespace(shape=(1,))
        observation_space = types.SimpleNamespace(shape=(1,))
        def __init__(self, *a, **k):
            pass
        def reset(self):
            return np.zeros(1)
        def step(self, action):
            return np.zeros(1), 0.0, True, {}
    monkeypatch.setattr(train_rl, "RiskEnv", DummyRiskEnv)

    train_rl.main()

    assert metrics_logged["cumulative_return"] == 0.0
    assert metrics_logged["sharpe_ratio"] == 0.0
    assert metrics_logged["max_drawdown"] == 0.0
