import json
from pathlib import Path
import sys
import types
import importlib
import importlib.util

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

# Load live_recorder without importing data package
spec = importlib.util.spec_from_file_location(
    "data.live_recorder", root / "data" / "live_recorder.py"
)
live_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
pyarrow_mod = types.ModuleType("pyarrow")


class _FakeTable:
    @staticmethod
    def from_pandas(frame, preserve_index=False):
        return frame


pyarrow_mod.Table = _FakeTable
pyarrow_parquet_mod = types.SimpleNamespace(write_to_dataset=lambda *a, **k: None)
pyarrow_mod.parquet = pyarrow_parquet_mod
sys.modules.setdefault("pyarrow", pyarrow_mod)
sys.modules.setdefault("pyarrow.parquet", pyarrow_parquet_mod)
spec.loader.exec_module(live_mod)
LiveRecorder = live_mod.LiveRecorder
if not hasattr(LiveRecorder, "record"):
    LiveRecorder.record = lambda self, frame: self._write_batch(frame)
sys.modules.setdefault("data", types.ModuleType("data"))
sys.modules["data"].live_recorder = live_mod
sys.modules["data.live_recorder"] = live_mod

features_mod = types.ModuleType("data.features")


def _make_features(frame, validate=False):
    frame = frame.copy()
    if "mid" not in frame.columns and {"Bid", "Ask"}.issubset(frame.columns):
        frame["mid"] = (frame["Bid"] + frame["Ask"]) / 2
    frame["feat_return"] = frame["mid"].pct_change().fillna(0.0)
    return frame


features_mod.make_features = _make_features
sys.modules.setdefault("data.features", features_mod)
sys.modules["data"].features = features_mod

labels_mod = types.ModuleType("data.labels")


def _triple_barrier(prices, pt_mult, sl_mult, horizon):
    changes = prices.diff().fillna(0.0)
    signs = np.sign(changes).astype(int)
    return pd.Series(signs, index=prices.index)


labels_mod.triple_barrier = _triple_barrier
sys.modules.setdefault("data.labels", labels_mod)
sys.modules["data"].labels = labels_mod

train_utils_mod = types.ModuleType("train_utils")


def _resolve_features(df, target, cfg, **kwargs):
    numeric_cols = df.select_dtypes(include=[np.number, "bool"]).columns
    drop_cols = {"Timestamp", "Symbol", "tb_label"}
    return [col for col in df.columns if col in numeric_cols and col not in drop_cols]


train_utils_mod.resolve_training_features = _resolve_features
sys.modules.setdefault("train_utils", train_utils_mod)


class DummyModel:
    def learn_one(self, x, y):
        return self

    def predict_proba_one(self, x):  # pragma: no cover - deterministic stub
        return {0: 0.5, 1: 0.5}


river = types.ModuleType("river")
river.compose = types.SimpleNamespace(Pipeline=lambda *a, **k: DummyModel())
river.preprocessing = types.SimpleNamespace(StandardScaler=lambda: None)
river.linear_model = types.SimpleNamespace(LogisticRegression=lambda: DummyModel())
sys.modules.setdefault("river", river)

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {"seed": 1, "drawdown_limit": 0.1}
spec_utils = importlib.machinery.ModuleSpec("utils", loader=None, is_package=True)
spec_utils.submodule_search_locations = []
utils_stub.__spec__ = spec_utils
utils_stub.__path__ = []
sys.modules["utils"] = utils_stub

log_utils_stub = types.ModuleType("log_utils")
log_utils_stub.setup_logging = lambda: None
log_utils_stub.log_exceptions = lambda f: f
log_utils_stub.log_predictions = lambda *a, **k: None
sys.modules["log_utils"] = log_utils_stub

state_manager_stub = types.ModuleType("state_manager")
state_manager_stub.watch_config = lambda cfg: types.SimpleNamespace(stop=lambda: None)
state_manager_stub.load_runtime_state = lambda *a, **k: None
state_manager_stub.save_runtime_state = lambda *a, **k: None
state_manager_stub.migrate_runtime_state = lambda *a, **k: None
state_manager_stub.legacy_runtime_state_exists = lambda *a, **k: False
sys.modules["state_manager"] = state_manager_stub

for mod in ["scipy", "scipy.stats", "scipy.sparse"]:
    sys.modules.pop(mod, None)

model_registry_stub = sys.modules["model_registry"]
saved_paths = model_registry_stub.saved_paths

import pandas as pd
import numpy as np
import joblib
import time

import train_online


def _make_ticks(ts, n=3):
    return pd.DataFrame(
        {
            "Timestamp": [ts + pd.Timedelta(seconds=i) for i in range(n)],
            "Bid": [1.0 + i * 0.1 for i in range(n)],
            "Ask": [1.0 + i * 0.1 for i in range(n)],
            "Symbol": ["EURUSD"] * n,
        }
    )


def test_live_recording_triggers_retraining(tmp_path, monkeypatch):
    data_path = tmp_path / "live"
    models = tmp_path / "models"
    batches: list[pd.DataFrame] = []

    def _record(self, frame: pd.DataFrame) -> None:
        batches.append(frame.copy())

    monkeypatch.setattr(LiveRecorder, "record", _record, raising=False)

    def fake_load_ticks(path, since):
        if not batches:
            return pd.DataFrame(columns=["Timestamp", "Bid", "Ask", "Symbol"])
        combined = pd.concat(batches, ignore_index=True)
        if since is not None:
            combined = combined[combined["Timestamp"] > since]
        return combined.sort_values("Timestamp").reset_index(drop=True)

    monkeypatch.setattr(train_online, "load_ticks", fake_load_ticks)

    def fake_make_features(frame, validate=False):
        frame = frame.copy()
        if "mid" not in frame.columns:
            frame["mid"] = (frame["Bid"] + frame["Ask"]) / 2
        frame["feat_a"] = frame["mid"].pct_change().fillna(0.0)
        frame["feat_b"] = frame["feat_a"].rolling(2).mean().fillna(0.0)
        return frame

    monkeypatch.setattr(train_online, "make_features", fake_make_features)
    monkeypatch.setattr(
        train_online,
        "resolve_training_features",
        lambda df, target, cfg, **kwargs: ["feat_a", "feat_b"],
    )

    class _CfgTraining:
        def __init__(self) -> None:
            self.pt_mult = 0.01
            self.sl_mult = 0.01
            self.max_horizon = 2

    class _Cfg:
        def __init__(self) -> None:
            self.training = _CfgTraining()

        def get(self, key, default=None):
            mapping = {
                "seed": 0,
                "pt_mult": self.training.pt_mult,
                "sl_mult": self.training.sl_mult,
                "max_horizon": self.training.max_horizon,
                "drawdown_limit": 0.2,
                "online_feature_window": 5,
            }
            return mapping.get(key, default)

    monkeypatch.setattr(train_online, "load_config", lambda: _Cfg())
    monkeypatch.setattr(
        train_online, "watch_config", lambda cfg: types.SimpleNamespace(stop=lambda: None)
    )

    rec = LiveRecorder(data_path)
    ts = pd.Timestamp("2023-01-01T00:00:00Z")
    rec.record(_make_ticks(ts))

    train_online.train_online(
        data_path=data_path, model_dir=models, run_once=True, min_ticks=1, interval=0
    )
    latest = models / "online_latest.joblib"
    assert latest.exists()
    state = joblib.load(latest)
    last1 = state["last_train_ts"]

    rec.record(_make_ticks(ts + pd.Timedelta(seconds=3), n=2))
    time.sleep(1)
    train_online.train_online(
        data_path=data_path, model_dir=models, run_once=True, min_ticks=1, interval=0
    )
    state = joblib.load(latest)
    last2 = state["last_train_ts"]
    assert last2 > last1
    assert len(saved_paths) == 2
    # Rollback to the previous model
    train_online.rollback_model(model_dir=models)
    state = joblib.load(latest)
    assert state["last_train_ts"] == last1
