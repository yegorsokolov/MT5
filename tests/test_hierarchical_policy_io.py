import sys
import types
import importlib.machinery
from pathlib import Path

import pandas as pd
import numpy as np
import contextlib

# Ensure repository root on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# ---------------------------------------------------------------------------
# Stub heavy optional dependencies
# ---------------------------------------------------------------------------

sb3 = types.ModuleType("stable_baselines3")
sb3.PPO = sb3.SAC = sb3.A2C = object
sb3.__spec__ = importlib.machinery.ModuleSpec("stable_baselines3", loader=None)
sb3.common = types.ModuleType("stable_baselines3.common")
sb3.common.vec_env = types.ModuleType("stable_baselines3.common.vec_env")
sb3.common.vec_env.SubprocVecEnv = object
sb3.common.vec_env.DummyVecEnv = object
sb3.common.evaluation = types.ModuleType("stable_baselines3.common.evaluation")
sb3.common.evaluation.evaluate_policy = lambda *a, **k: None
sb3.common.__spec__ = importlib.machinery.ModuleSpec("stable_baselines3.common", loader=None)
sb3.common.vec_env.__spec__ = importlib.machinery.ModuleSpec("stable_baselines3.common.vec_env", loader=None)
sb3.common.evaluation.__spec__ = importlib.machinery.ModuleSpec("stable_baselines3.common.evaluation", loader=None)
sys.modules.setdefault("stable_baselines3", sb3)
sys.modules.setdefault("stable_baselines3.common", sb3.common)
sys.modules.setdefault("stable_baselines3.common.vec_env", sb3.common.vec_env)
sys.modules.setdefault("stable_baselines3.common.evaluation", sb3.common.evaluation)

contrib = types.ModuleType("sb3_contrib")
contrib.TRPO = contrib.RecurrentPPO = contrib.HierarchicalPPO = object
contrib.__spec__ = importlib.machinery.ModuleSpec("sb3_contrib", loader=None)
sys.modules.setdefault("sb3_contrib", contrib)
qrdqn_mod = types.ModuleType("sb3_contrib.qrdqn")
qrdqn_mod.QRDQN = object
qrdqn_mod.__spec__ = importlib.machinery.ModuleSpec("sb3_contrib.qrdqn", loader=None)
sys.modules.setdefault("sb3_contrib.qrdqn", qrdqn_mod)

# Torch stub
torch_stub = types.ModuleType("torch")
torch_stub.manual_seed = lambda *a, **k: None
torch_stub.cuda = types.SimpleNamespace(
    is_available=lambda: False,
    manual_seed_all=lambda *a, **k: None,
    device_count=lambda: 0,
)
torch_stub.nn = types.SimpleNamespace(Module=object, Linear=lambda *a, **k: None)
torch_stub.__spec__ = importlib.machinery.ModuleSpec("torch", loader=None)
sys.modules.setdefault("torch", torch_stub)

# Misc stubs
mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.set_tracking_uri = lambda *a, **k: None
mlflow_stub.set_experiment = lambda *a, **k: None
mlflow_stub.start_run = lambda *a, **k: None
mlflow_stub.end_run = lambda *a, **k: None
mlflow_stub.log_params = lambda *a, **k: None
mlflow_stub.log_metrics = lambda *a, **k: None
mlflow_stub.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
sys.modules.setdefault("mlflow", mlflow_stub)

metrics_stub = types.ModuleType("analytics.metrics_store")
metrics_stub.__spec__ = importlib.machinery.ModuleSpec(
    "analytics.metrics_store", loader=None
)
metrics_stub.record_metric = lambda *a, **k: None
metrics_stub.TS_PATH = ""
metrics_stub.model_cache_hit = lambda *a, **k: None
metrics_stub.model_unload = lambda *a, **k: None
sys.modules["analytics.metrics_store"] = metrics_stub
analytics_pkg = types.ModuleType("analytics")
analytics_pkg.__spec__ = importlib.machinery.ModuleSpec("analytics", loader=None)
analytics_pkg.metrics_store = metrics_stub
sys.modules["analytics"] = analytics_pkg

data_pkg = types.ModuleType("data")
history_stub = types.ModuleType("data.history")
history_stub.load_history_config = lambda sym, cfg, root: pd.DataFrame(
    {
        "Timestamp": pd.date_range("2020-01-01", periods=10, freq="min"),
        "mid": np.linspace(1.0, 1.1, 10),
        "Symbol": [sym] * 10,
    }
)
history_stub.load_history_parquet = lambda *a, **k: pd.DataFrame()
history_stub.save_history_parquet = lambda *a, **k: None
data_features_stub = types.ModuleType("data.features")
data_features_stub.make_features = lambda df, validate=False: df.assign(
    **{"return": df["mid"].pct_change().fillna(0.0)}
)
data_pkg.history = history_stub
data_pkg.features = data_features_stub
sys.modules["data"] = data_pkg
sys.modules["data.history"] = history_stub
sys.modules["data.features"] = data_features_stub

telemetry_stub = types.SimpleNamespace(
    get_tracer=lambda *a, **k: types.SimpleNamespace(
        start_as_current_span=lambda *a, **k: contextlib.nullcontext()
    ),
    get_meter=lambda *a, **k: types.SimpleNamespace(
        create_counter=lambda *a, **k: types.SimpleNamespace(add=lambda *a, **k: None)
    ),
)
sys.modules.setdefault("telemetry", telemetry_stub)

sys.modules.setdefault(
    "prometheus_client",
    types.SimpleNamespace(
        Counter=lambda *a, **k: types.SimpleNamespace(inc=lambda *a, **k: None),
        Gauge=lambda *a, **k: types.SimpleNamespace(set=lambda *a, **k: None),
    ),
)

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda *a, **k: {}
utils_stub.__path__ = []
sys.modules.setdefault("utils", utils_stub)

requests_stub = types.ModuleType("requests")
requests_stub.get = lambda *a, **k: None
requests_stub.__spec__ = importlib.machinery.ModuleSpec("requests", loader=None)
sys.modules.setdefault("requests", requests_stub)

date_parser_stub = types.SimpleNamespace(parse=lambda *a, **k: None)
sys.modules.setdefault("dateutil", types.SimpleNamespace(parser=date_parser_stub))
sys.modules.setdefault("dateutil.parser", date_parser_stub)

data_backend_stub = types.ModuleType("utils.data_backend")
data_backend_stub.get_dataframe_module = lambda: pd
sys.modules.setdefault("utils.data_backend", data_backend_stub)

monitor_stub = types.SimpleNamespace(
    start=lambda: None,
    capability_tier="lite",
    capabilities=types.SimpleNamespace(capability_tier=lambda: "lite", ddp=lambda: False),
)
sys.modules.setdefault("utils.resource_monitor", types.SimpleNamespace(monitor=monitor_stub))

ge_stub = types.ModuleType("great_expectations")
ge_stub.__spec__ = importlib.machinery.ModuleSpec("great_expectations", loader=None)
ge_core_stub = types.ModuleType("great_expectations.core")
ge_core_stub.__spec__ = importlib.machinery.ModuleSpec("great_expectations.core", loader=None)
ge_core_stub.expectation_suite = types.ModuleType("great_expectations.core.expectation_suite")
ge_core_stub.expectation_suite.ExpectationSuite = object
sys.modules.setdefault("great_expectations", ge_stub)
sys.modules.setdefault("great_expectations.core", ge_core_stub)
sys.modules.setdefault(
    "great_expectations.core.expectation_suite", ge_core_stub.expectation_suite
)

features_stub = types.ModuleType("features")
features_stub.make_features = data_features_stub.make_features
features_stub.get_feature_pipeline = lambda *a, **k: []
sys.modules["features"] = features_stub

import features

class _Space:
    def __init__(self, *a, **k):
        self.n = 2

    def sample(self):
        return 0

gym_spaces = types.SimpleNamespace(Box=_Space, Dict=_Space, Discrete=_Space, MultiBinary=_Space)
sys.modules.setdefault(
    "gym", types.SimpleNamespace(Env=object, spaces=gym_spaces, Wrapper=object)
)

sys.modules.setdefault(
    "psutil",
    types.SimpleNamespace(
        cpu_count=lambda logical=True: 1,
        virtual_memory=lambda: types.SimpleNamespace(total=0),
        Process=lambda pid=None: types.SimpleNamespace(
            memory_info=lambda: types.SimpleNamespace(rss=0)
        ),
        disk_io_counters=lambda: types.SimpleNamespace(read_bytes=0, write_bytes=0),
    ),
)

filelock_stub = types.ModuleType("filelock")
filelock_stub.FileLock = object
sys.modules.setdefault("filelock", filelock_stub)

crypto_module = types.ModuleType(
    "cryptography.hazmat.primitives.ciphers.aead"
)

class _AESGCM:
    def __init__(self, *a, **k):
        pass

    def encrypt(self, *a, **k):
        return b""

    def decrypt(self, *a, **k):
        return b""

crypto_module.AESGCM = _AESGCM
sys.modules.setdefault("cryptography", types.ModuleType("cryptography"))
sys.modules.setdefault("cryptography.hazmat", types.ModuleType("cryptography.hazmat"))
sys.modules.setdefault(
    "cryptography.hazmat.primitives",
    types.ModuleType("cryptography.hazmat.primitives"),
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.ciphers",
    types.ModuleType("cryptography.hazmat.primitives.ciphers"),
)
sys.modules.setdefault(
    "cryptography.hazmat.primitives.ciphers.aead",
    crypto_module,
)
from mt5.train_rl import train_hierarchical, eval_hierarchical, HierarchicalTradingEnv
from rl.hierarchical_agent import HierarchicalAgent
from mt5.model_registry import get_policy_path
import joblib


def test_hierarchical_policy_io(tmp_path: Path):
    df = pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=10, freq="min"),
            "mid": np.linspace(1.0, 1.1, 10),
            "Symbol": ["A"] * 10,
        }
    )
    csv = tmp_path / "A_history.csv"
    df.to_csv(csv, index=False)
    cfg = {
        "history_path": csv,
        "symbol": "A",
        "hierarchical_steps": 5,
        "checkpoint_dir": tmp_path,
    }
    train_hierarchical(cfg)

    # Load the saved policies and ensure agent acts
    manager_path = get_policy_path("hier_manager")
    assert manager_path and manager_path.exists()
    manager = joblib.load(manager_path)
    workers = {}
    for name in ["mean_reversion", "news", "trend"]:
        path = get_policy_path(f"hier_worker_{name}")
        assert path and path.exists()
        workers[name] = joblib.load(path)

    feats = features.make_features(df)
    env = HierarchicalTradingEnv(feats, ["return"], max_position=1.0)
    agent = HierarchicalAgent(manager, workers)
    obs = env.reset()
    action = agent.act(obs)
    assert "manager" in action and "worker" in action
