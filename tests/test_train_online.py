import sys
import importlib.machinery
import types
import json
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

state_manager_stub = types.ModuleType("state_manager")
state_manager_stub.watch_config = lambda cfg: types.SimpleNamespace(
    stop=lambda: None, join=lambda: None
)
state_manager_stub.load_runtime_state = lambda *a, **k: None
state_manager_stub.save_runtime_state = lambda *a, **k: None
state_manager_stub.migrate_runtime_state = lambda *a, **k: None
state_manager_stub.legacy_runtime_state_exists = lambda *a, **k: False
state_manager_stub.save_checkpoint = lambda *a, **k: None
state_manager_stub.load_latest_checkpoint = lambda *a, **k: None
sys.modules["state_manager"] = state_manager_stub

for mod in ["scipy", "scipy.stats", "scipy.sparse"]:
    sys.modules.pop(mod, None)

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda *a, **k: {}
utils_spec = importlib.machinery.ModuleSpec("utils", loader=None, is_package=True)
utils_spec.submodule_search_locations = []
utils_stub.__spec__ = utils_spec
utils_stub.__path__ = []
sys.modules["utils"] = utils_stub

resource_monitor_stub = types.ModuleType("utils.resource_monitor")
resource_monitor_stub.monitor = types.SimpleNamespace()
resource_monitor_stub.ResourceCapabilities = lambda *a, **k: types.SimpleNamespace()
resource_monitor_stub.ResourceMonitor = lambda *a, **k: types.SimpleNamespace()
sys.modules["utils.resource_monitor"] = resource_monitor_stub

ge_mod = types.ModuleType("great_expectations")
core_mod = types.ModuleType("great_expectations.core")
suite_mod = types.ModuleType("great_expectations.core.expectation_suite")
suite_mod.ExpectationSuite = type("ExpectationSuite", (), {})
ge_mod.core = core_mod
sys.modules["great_expectations"] = ge_mod
sys.modules["great_expectations.core"] = core_mod
sys.modules["great_expectations.core.expectation_suite"] = suite_mod

sys.modules["psutil"] = types.ModuleType("psutil")

log_utils_stub = types.ModuleType("log_utils")
log_utils_stub.setup_logging = lambda: None
log_utils_stub.log_exceptions = lambda f: f
log_utils_stub.log_predictions = lambda *a, **k: None
log_utils_stub.LOG_DIR = Path("/tmp")
log_utils_stub.LOG_FILE = Path("/tmp/app.log")
sys.modules["log_utils"] = log_utils_stub

sklearn_stub = types.ModuleType("sklearn")
metrics_stub = types.ModuleType("sklearn.metrics")


def _f1_score(y_true, y_pred, zero_division=0):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    if tp == 0:
        return float(zero_division)
    precision = tp / (tp + fp) if (tp + fp) else zero_division
    recall = tp / (tp + fn) if (tp + fn) else zero_division
    if precision == 0 and recall == 0:
        return float(zero_division)
    return float(2 * precision * recall / (precision + recall))


metrics_stub.f1_score = _f1_score
sklearn_stub.metrics = metrics_stub
sys.modules["sklearn"] = sklearn_stub
sys.modules["sklearn.metrics"] = metrics_stub


class _StubModel:
    def learn_one(self, x, y):
        return self


river_stub = types.ModuleType("river")
river_stub.compose = types.SimpleNamespace(Pipeline=lambda *a, **k: _StubModel())
river_stub.preprocessing = types.SimpleNamespace(StandardScaler=lambda: None)
river_stub.linear_model = types.SimpleNamespace(LogisticRegression=lambda: _StubModel())
sys.modules.setdefault("river", river_stub)

import numpy as np
import pandas as pd
import joblib

data_features_stub = types.ModuleType("data.features")
data_features_stub.make_features = lambda df, *a, **k: df
sys.modules["data.features"] = data_features_stub

labels_stub = types.ModuleType("data.labels")
labels_stub.triple_barrier = lambda prices, *a, **k: pd.Series(np.ones(len(prices)))


def _multi_horizon_stub(prices, horizons):
    data = {f"direction_{h}": pd.Series(0, index=prices.index, dtype=int) for h in horizons}
    return pd.DataFrame(data)


labels_stub.multi_horizon_labels = _multi_horizon_stub
sys.modules["data.labels"] = labels_stub

live_recorder_stub = types.ModuleType("data.live_recorder")
live_recorder_stub.load_ticks = lambda *a, **k: pd.DataFrame()
sys.modules["data.live_recorder"] = live_recorder_stub

train_utils_stub = types.ModuleType("train_utils")
train_utils_stub.resolve_training_features = (
    lambda df, target, cfg, **kwargs: [
        col
        for col in df.columns
        if col not in {"Timestamp", "Symbol", "tb_label"}
    ]
)
train_utils_stub.prepare_modal_arrays = lambda *a, **k: None
sys.modules["train_utils"] = train_utils_stub
sys.modules["mt5.train_utils"] = train_utils_stub

sys.modules.pop("model_registry", None)
model_registry_stub = types.ModuleType("model_registry")
model_registry_stub.saved_paths = []

def _stub_save_model(name, model, metadata, path=None):
    target = Path(path or name)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, target)
    meta_path = target.with_suffix(".json")
    try:
        meta_path.write_text(json.dumps(metadata))
    except Exception:
        pass
    model_registry_stub.saved_paths.append(target)
    return target


model_registry_stub.save_model = _stub_save_model
sys.modules["model_registry"] = model_registry_stub
sys.modules["mt5.model_registry"] = model_registry_stub

from mt5 import train_online

train_online.init_logging = lambda: None


class DummyModel:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], int]] = []

    def learn_one(self, x, y):
        self.calls.append((x, y))
        return self

    def predict_proba_one(self, x):  # pragma: no cover - deterministic stub
        return {0: 0.5, 1: 0.5}


def test_train_online_enriches_features(tmp_path, monkeypatch):
    dummy_model = DummyModel()
    monkeypatch.setattr(train_online.compose, "Pipeline", lambda *a, **k: dummy_model)
    monkeypatch.setattr(train_online.preprocessing, "StandardScaler", lambda: None)
    monkeypatch.setattr(train_online.linear_model, "LogisticRegression", lambda: None)

    class DummyTraining:
        def __init__(self) -> None:
            self.pt_mult = 0.01
            self.sl_mult = 0.01
            self.max_horizon = 2

    class DummyCfg:
        def __init__(self) -> None:
            self.training = DummyTraining()

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

    monkeypatch.setattr(train_online, "load_config", lambda: DummyCfg())
    class ObserverStub:
        def __init__(self) -> None:
            self.stop_calls = 0
            self.join_calls = 0

        def stop(self) -> None:
            self.stop_calls += 1

        def join(self) -> None:
            self.join_calls += 1

    observer = ObserverStub()
    monkeypatch.setattr(train_online, "watch_config", lambda cfg: observer)

    timestamps = pd.date_range("2024-01-01", periods=6, freq="min", tz="UTC")
    ticks = pd.DataFrame(
        {
            "Timestamp": timestamps,
            "Bid": np.linspace(1.0, 1.5, len(timestamps)),
            "Ask": np.linspace(1.001, 1.501, len(timestamps)),
            "Symbol": ["TEST"] * len(timestamps),
        }
    )

    call_state = {"count": 0}

    dump_calls: list[tuple[object, Path]] = []
    original_dump = train_online.joblib.dump

    def _tracking_dump(obj, path):
        dump_calls.append((obj, Path(path)))
        return original_dump(obj, path)

    monkeypatch.setattr(train_online.joblib, "dump", _tracking_dump)

    def fake_load_ticks(path, since):
        if call_state["count"] == 0:
            call_state["count"] += 1
            return ticks.copy()
        call_state["count"] += 1
        return pd.DataFrame(columns=ticks.columns)

    monkeypatch.setattr(train_online, "load_ticks", fake_load_ticks)

    def fake_make_features(frame, validate=False):
        frame = frame.copy()
        if "mid" not in frame.columns:
            frame["mid"] = (frame["Bid"] + frame["Ask"]) / 2
        frame["feat_a"] = np.linspace(0.0, 1.0, len(frame), dtype=float)
        frame["feat_b"] = np.linspace(1.0, 2.0, len(frame), dtype=float)
        return frame

    monkeypatch.setattr(train_online, "make_features", fake_make_features)
    monkeypatch.setattr(
        train_online,
        "resolve_training_features",
        lambda df, target, cfg, **kwargs: ["feat_a", "feat_b"],
    )

    saved_metadata: dict[str, object] = {}

    def fake_save_model(name, model, metadata, path):
        saved_metadata["metadata"] = metadata
        saved_metadata["path"] = Path(path)
        joblib.dump(model, path)
        return Path(path)

    monkeypatch.setattr(train_online, "save_model", fake_save_model)

    model_dir = tmp_path / "models"
    data_dir = tmp_path / "data"
    model_dir.mkdir()
    data_dir.mkdir()

    train_online.train_online(
        data_path=data_dir,
        model_dir=model_dir,
        run_once=True,
        min_ticks=1,
        interval=0,
    )

    assert dump_calls, "Expected joblib.dump to be invoked"
    latest_path = model_dir / "online_latest.joblib"
    latest_entries = [
        (obj, path) for obj, path in dump_calls if Path(path) == latest_path
    ]
    assert latest_entries, "State dump to online_latest.joblib was not recorded"
    state_obj, state_path = latest_entries[-1]
    assert isinstance(state_obj, dict)
    assert state_obj["feature_columns"] == ["feat_a", "feat_b"]
    assert state_obj["label_column"] == "tb_label"
    assert isinstance(state_obj["last_train_ts"], pd.Timestamp)
    assert dummy_model.calls
    assert saved_metadata["metadata"]["preprocessing"]["feature_columns"] == ["feat_a", "feat_b"]
    assert saved_metadata["metadata"]["preprocessing"]["label_column"] == "tb_label"
    assert observer.stop_calls == 1
    assert observer.join_calls == 1
