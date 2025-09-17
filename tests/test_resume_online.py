import numpy as np
import pandas as pd

import sys
from pathlib import Path
import numpy as np
import pandas as pd

import os
import base64

sys.path.append(str(Path(__file__).resolve().parents[1]))
import types
import contextlib

class _DummyFileLock:
    def __init__(self, *args, **kwargs):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

sys.modules.setdefault("filelock", types.SimpleNamespace(FileLock=_DummyFileLock))

env_mod = types.ModuleType("utils.environment")
env_mod.ensure_environment = lambda: None
sys.modules.setdefault("utils.environment", env_mod)
hist_mod = types.ModuleType("data.history")
hist_mod.load_history_parquet = lambda *a, **k: None
hist_mod.save_history_parquet = lambda *a, **k: None
hist_mod.load_history_config = lambda *a, **k: None
hist_mod.load_history_iter = lambda *a, **k: []
hist_mod.load_history_from_urls = lambda *a, **k: None
hist_mod.load_history_mt5 = lambda *a, **k: None
hist_mod.load_history = lambda *a, **k: None
hist_mod.load_multiple_histories = lambda *a, **k: []
hist_mod.load_history_memmap = lambda *a, **k: None
sys.modules.setdefault("data.history", hist_mod)
feat_mod = types.ModuleType("data.features")


def _feat_noop(*args, **kwargs):
    return args[0] if args else None


feat_mod.make_features = lambda df, validate=False: df
feat_mod.train_test_split = lambda df, rows: (df, df.iloc[0:0])
feat_mod.make_sequence_arrays = lambda df, features, seq_len: (
    np.zeros((0, seq_len, len(features))),
    np.zeros((0,)),
)
feat_mod.add_index_features = _feat_noop
feat_mod.add_economic_calendar_features = _feat_noop
feat_mod.__file__ = __file__
feat_mod.__getattr__ = lambda name: _feat_noop
sys.modules.setdefault("data.features", feat_mod)
reg_mod = types.ModuleType("analysis.regime_detection")
reg_mod.periodic_reclassification = lambda df, step=500: df
sys.modules.setdefault("analysis.regime_detection", reg_mod)
mon_mod = types.ModuleType("utils.resource_monitor")


class _DummyMonitor:
    def start(self):
        pass

    capability_tier = "lite"
    capabilities = types.SimpleNamespace(
        cpus=1, ddp=lambda: False, capability_tier=lambda: "lite"
    )


mon_mod.monitor = _DummyMonitor()
mon_mod.ResourceMonitor = _DummyMonitor
sys.modules.setdefault("utils.resource_monitor", mon_mod)
mlflow_mod = types.SimpleNamespace(
    set_tracking_uri=lambda *a, **k: None,
    set_experiment=lambda *a, **k: None,
    start_run=lambda *a, **k: contextlib.nullcontext(),
    log_dict=lambda *a, **k: None,
    log_metric=lambda *a, **k: None,
    log_param=lambda *a, **k: None,
    log_artifact=lambda *a, **k: None,
)
sys.modules.setdefault("mlflow", mlflow_mod)
sys.modules.setdefault(
    "analytics.mlflow_client",
    types.SimpleNamespace(
        start_run=lambda *a, **k: contextlib.nullcontext(),
        end_run=lambda *a, **k: None,
        log_param=lambda *a, **k: None,
        log_metric=lambda *a, **k: None,
        log_dict=lambda *a, **k: None,
        log_artifact=lambda *a, **k: None,
    ),
)
for mod in ["scipy", "scipy.stats"]:
    sys.modules.pop(mod, None)
prob_mod = types.SimpleNamespace(
    ProbabilityCalibrator=object,
    CalibratedModel=object,
    log_reliability=lambda *a, **k: None,
)
sys.modules.setdefault("analysis.prob_calibration", prob_mod)
version_mod = types.ModuleType("data.versioning")
version_mod.compute_hash = lambda path: "0"
sys.modules.setdefault("data.versioning", version_mod)

orch_mod = types.SimpleNamespace(Orchestrator=types.SimpleNamespace(start=lambda: None))
sys.modules.setdefault("core.orchestrator", orch_mod)
model_registry_mod = types.ModuleType("model_registry")
model_registry_mod.ModelRegistry = lambda *a, **k: None
sys.modules.setdefault("model_registry", model_registry_mod)
risk_manager_mod = types.ModuleType("risk_manager")
risk_manager_mod.risk_manager = lambda *a, **k: None
risk_manager_mod.subscribe_to_broker_alerts = lambda *a, **k: None
sys.modules.setdefault("risk_manager", risk_manager_mod)


class _DummyMeter:
    def create_counter(self, *a, **k):
        return types.SimpleNamespace(add=lambda *a, **k: None)


class _DummyTracer:
    def start_as_current_span(self, *a, **k):
        return contextlib.nullcontext()


telemetry_mod = types.SimpleNamespace(
    get_tracer=lambda *a, **k: _DummyTracer(),
    get_meter=lambda *a, **k: _DummyMeter(),
)
sys.modules.setdefault("telemetry", telemetry_mod)
from state_manager import load_latest_checkpoint

os.environ.setdefault("CHECKPOINT_AES_KEY", base64.b64encode(b"0" * 32).decode())
import train_nn


FEATURES = [
    "return",
    "ma_5",
    "ma_10",
    "ma_30",
    "ma_60",
    "volatility_30",
    "spread",
    "rsi_14",
    "news_sentiment",
    "market_regime",
]


def _make_df(n: int) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    data = {f: rng.standard_normal(n) for f in FEATURES[:-1]}
    data["market_regime"] = np.zeros(n, dtype=int)
    return pd.DataFrame(data)


def test_lightgbm_resume_online(tmp_path):
    import train

    cfg = {"checkpoint_dir": str(tmp_path), "online_batch_size": 5}
    df_full = _make_df(10)
    train.main(cfg, resume_online=True, df_override=df_full.iloc[:5])
    step, _ = load_latest_checkpoint(str(tmp_path))
    assert step == 0
    train.main(cfg, resume_online=True, df_override=df_full)
    step, _ = load_latest_checkpoint(str(tmp_path))
    assert step == 1


def test_pytorch_resume_online(tmp_path, monkeypatch):
    df_holder = {"df": _make_df(4)}

    def load_history_config(sym, cfg, root, validate=False):
        return df_holder["df"]

    monkeypatch.setattr(train_nn, "load_history_config", load_history_config)
    monkeypatch.setattr(train_nn, "make_features", lambda df, validate=False: df)
    class _DummySplit:
        def __init__(self, *a, **k):
            pass

        def split(self, X, y=None, groups=None):
            n = len(X)
            yield list(range(n - 1)), [n - 1]

    monkeypatch.setattr(train_nn, "PurgedTimeSeriesSplit", _DummySplit)
    monkeypatch.setattr(
        train_nn,
        "generate_time_series_folds",
        lambda n_samples, **_: [
            (
                np.arange(max(n_samples - 1, 0), dtype=int),
                np.arange(max(n_samples - 1, 0), n_samples, dtype=int),
            )
        ]
        if n_samples
        else [],
    )
    monkeypatch.setattr(train_nn, "resolve_group_labels", lambda df: None)
    monkeypatch.setattr(
        train_nn,
        "make_sequence_arrays",
        lambda df, features, seq_len: (
            (
                np.stack(
                    [
                        df[features].values[i - seq_len : i]
                        for i in range(seq_len, len(df))
                    ]
                )
                if len(df) > seq_len
                else np.empty((0, seq_len, len(features)))
            ),
            np.ones(max(len(df) - seq_len, 0)),
        ),
    )

    cfg = {
        "symbols": ["SYM"],
        "checkpoint_dir": str(tmp_path),
        "sequence_length": 2,
        "epochs": 1,
        "batch_size": 1,
        "train_rows": len(df_holder["df"]),
    }
    train_nn.main(0, 1, cfg, resume_online=True)
    step, _ = load_latest_checkpoint(str(tmp_path))
    assert step == 1

    df_holder["df"] = _make_df(7)
    cfg["train_rows"] = len(df_holder["df"])
    train_nn.main(0, 1, cfg, resume_online=True)
    step, _ = load_latest_checkpoint(str(tmp_path))
    assert step == 4
