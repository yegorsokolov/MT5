import types
import sys
import importlib.util
from pathlib import Path

repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
from event_store import EventStore

spec_ds = importlib.util.spec_from_file_location(
    "offline_dataset", repo_root / "rl" / "offline_dataset.py"
)
offline_dataset = importlib.util.module_from_spec(spec_ds)
sys.modules["offline_dataset"] = offline_dataset
spec_ds.loader.exec_module(offline_dataset)
OfflineDataset = offline_dataset.OfflineDataset


def _load_train_rl(monkeypatch):
    """Import ``train_rl`` with heavy dependencies stubbed out."""
    mlflow_stub = types.SimpleNamespace(
        set_tracking_uri=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        start_run=lambda *a, **k: None,
        log_param=lambda *a, **k: None,
        log_artifact=lambda *a, **k: None,
        end_run=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
        __spec__=types.SimpleNamespace(),
    )
    monkeypatch.setitem(sys.modules, "mlflow", mlflow_stub)

    log_utils_stub = types.SimpleNamespace(
        setup_logging=lambda *a, **k: None, log_exceptions=lambda f: f, __spec__=types.SimpleNamespace()
    )
    monkeypatch.setitem(sys.modules, "log_utils", log_utils_stub)

    class _NP(types.SimpleNamespace):
        float32 = float

        def __getattr__(self, name):  # pragma: no cover - simple fallback
            def _(*args, **kwargs):
                if name == "linspace" and len(args) >= 3:
                    start, stop, num = args[:3]
                    return [start + (stop - start) * i / (num - 1) for i in range(num)]
                return 0

            return _

    np_stub = _NP(random=types.SimpleNamespace(seed=lambda *a, **k: None, randn=lambda *a, **k: 0, normal=lambda *a, **k: 0))
    monkeypatch.setitem(sys.modules, "numpy", np_stub)

    class _PD(types.SimpleNamespace):
        def __getattr__(self, name):  # pragma: no cover - simple fallback
            return lambda *a, **k: None

    pd_stub = _PD(DataFrame=lambda *a, **k: None)
    monkeypatch.setitem(sys.modules, "pandas", pd_stub)

    torch_stub = types.SimpleNamespace(__spec__=types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "torch", torch_stub)

    gym_stub = types.SimpleNamespace(
        Env=object, Wrapper=object, spaces=types.SimpleNamespace(Box=object)
    )
    monkeypatch.setitem(sys.modules, "gym", gym_stub)

    utils_stub = types.SimpleNamespace(load_config=lambda: {})
    monkeypatch.setitem(sys.modules, "utils", utils_stub)
    state_stub = types.SimpleNamespace(
        save_checkpoint=lambda *a, **k: None, load_latest_checkpoint=lambda *a, **k: None
    )
    monkeypatch.setitem(sys.modules, "state_manager", state_stub)
    history_stub = types.SimpleNamespace(
        load_history_parquet=lambda *a, **k: None,
        save_history_parquet=lambda *a, **k: None,
        load_history_config=lambda *a, **k: pd_stub.DataFrame(),
    )
    features_stub = types.SimpleNamespace(make_features=lambda df, **k: df)
    monkeypatch.setitem(sys.modules, "data", types.SimpleNamespace())
    monkeypatch.setitem(sys.modules, "data.history", history_stub)
    monkeypatch.setitem(sys.modules, "data.features", features_stub)
    models_stub = types.SimpleNamespace(
        model_store=types.SimpleNamespace(save_model=lambda *a, **k: "0"),
        graph_net=types.SimpleNamespace(GraphNet=object),
    )
    monkeypatch.setitem(sys.modules, "models", models_stub)
    monkeypatch.setitem(sys.modules, "models.model_store", models_stub.model_store)
    monkeypatch.setitem(sys.modules, "models.graph_net", models_stub.graph_net)
    analysis_stub = types.SimpleNamespace(
        regime_detection=types.SimpleNamespace(periodic_reclassification=lambda df, **k: df)
    )
    monkeypatch.setitem(sys.modules, "analysis", analysis_stub)
    monkeypatch.setitem(
        sys.modules, "analysis.regime_detection", analysis_stub.regime_detection
    )

    # Minimal stubs for optional heavy libraries
    sb3_common = types.SimpleNamespace(
        vec_env=types.SimpleNamespace(SubprocVecEnv=object, DummyVecEnv=object),
        evaluation=types.SimpleNamespace(evaluate_policy=lambda *a, **k: None),
    )
    sb3_stub = types.SimpleNamespace(PPO=object, SAC=object, A2C=object, common=sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3", sb3_stub)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common", sb3_common)
    monkeypatch.setitem(sys.modules, "stable_baselines3.common.vec_env", sb3_common.vec_env)
    monkeypatch.setitem(
        sys.modules, "stable_baselines3.common.evaluation", sb3_common.evaluation
    )
    sb3_contrib_stub = types.SimpleNamespace(
        TRPO=object,
        RecurrentPPO=object,
        qrdqn=types.SimpleNamespace(QRDQN=object),
    )
    monkeypatch.setitem(sys.modules, "sb3_contrib", sb3_contrib_stub)
    monkeypatch.setitem(sys.modules, "sb3_contrib.qrdqn", sb3_contrib_stub.qrdqn)

    # Stub plugin imports used by train_rl
    plugins_stub = types.SimpleNamespace()
    rl_risk_stub = types.SimpleNamespace(RiskEnv=object)
    monkeypatch.setitem(sys.modules, "plugins", plugins_stub)
    monkeypatch.setitem(sys.modules, "plugins.rl_risk", rl_risk_stub)

    # Stub ray utils
    ray_utils_stub = types.SimpleNamespace(
        init=lambda *a, **k: None,
        shutdown=lambda *a, **k: None,
        cluster_available=lambda: False,
        submit=lambda *a, **k: None,
    )
    monkeypatch.setitem(sys.modules, "ray_utils", ray_utils_stub)

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    spec = importlib.util.spec_from_file_location(
        "train_rl", Path(__file__).resolve().parents[1] / "train_rl.py"
    )
    train_rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_rl)
    return train_rl


def _make_event_store(tmp_path):
    path = tmp_path / "events.db"
    store = EventStore(path)
    # y = 2x mapping
    xs = [(-1.0 + 2.0 * i / 99.0) for i in range(100)]
    for x in xs:
        store.record(
            "experience",
            {
                "obs": [float(x)],
                "action": [float(2 * x)],
                "reward": float(2 * x),
                "next_obs": [float(x)],
                "done": False,
            },
        )
    store.close()
    return path


def _policy_loss(policy, dataset):
    err = 0.0
    for s in dataset.samples:
        pred = policy([s.obs])[0][0]
        err += (pred - s.action[0]) ** 2
    return err / max(1, len(dataset.samples))


def test_offline_pretraining_improves_policy(monkeypatch, tmp_path):
    train_rl = _load_train_rl(monkeypatch)
    db_path = _make_event_store(tmp_path)
    dataset = OfflineDataset(db_path)

    class LinearPolicy:
        def __init__(self):
            self.weight = 0.0
            self.bias = 0.0

        def __call__(self, obs):
            return [[self.weight * o[0] + self.bias] for o in obs]

    policy = LinearPolicy()
    model = types.SimpleNamespace(policy=policy)

    before = _policy_loss(policy, dataset)
    train_rl.offline_pretrain(model, db_path, epochs=200, batch_size=32, lr=0.1)
    after = _policy_loss(policy, dataset)

    assert after < before
