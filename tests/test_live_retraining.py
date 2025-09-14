import json
from pathlib import Path
import sys
import types
import importlib.util

root = Path(__file__).resolve().parents[1]
sys.path.append(str(root))

# Load live_recorder without importing data package
spec = importlib.util.spec_from_file_location(
    "data.live_recorder", root / "data" / "live_recorder.py"
)
live_mod = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(live_mod)
LiveRecorder = live_mod.LiveRecorder
sys.modules.setdefault("data", types.ModuleType("data"))
sys.modules["data"].live_recorder = live_mod
sys.modules["data.live_recorder"] = live_mod


class DummyModel:
    def learn_one(self, x, y):
        return self


river = types.ModuleType("river")
river.compose = types.SimpleNamespace(Pipeline=lambda *a, **k: DummyModel())
river.preprocessing = types.SimpleNamespace(StandardScaler=lambda: None)
river.linear_model = types.SimpleNamespace(LogisticRegression=lambda: DummyModel())
sys.modules.setdefault("river", river)

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {"seed": 1, "drawdown_limit": 0.1}
sys.modules.setdefault("utils", utils_stub)

log_utils_stub = types.ModuleType("log_utils")
log_utils_stub.setup_logging = lambda: None
log_utils_stub.log_exceptions = lambda f: f
sys.modules.setdefault("log_utils", log_utils_stub)

saved_paths: list[Path] = []

model_registry_stub = types.ModuleType("model_registry")


def save_model(name, model, metadata, path=None):
    p = Path(path or name)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p)
    p.with_suffix(".json").write_text(json.dumps(metadata))
    saved_paths.append(p)
    return p


model_registry_stub.save_model = save_model
sys.modules.setdefault("model_registry", model_registry_stub)

import pandas as pd
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


def test_live_recording_triggers_retraining(tmp_path):
    data_path = tmp_path / "live"
    models = tmp_path / "models"
    rec = LiveRecorder(data_path)
    ts = pd.Timestamp("2023-01-01T00:00:00Z")
    rec.record(_make_ticks(ts))

    train_online.train_online(
        data_path=data_path, model_dir=models, run_once=True, min_ticks=1, interval=0
    )
    latest = models / "online_latest.joblib"
    assert latest.exists()
    _, last1 = joblib.load(latest)

    rec.record(_make_ticks(ts + pd.Timedelta(seconds=3), n=2))
    time.sleep(1)
    train_online.train_online(
        data_path=data_path, model_dir=models, run_once=True, min_ticks=1, interval=0
    )
    _, last2 = joblib.load(latest)
    assert last2 > last1
    assert len(saved_paths) == 2
    # Rollback to the previous model
    train_online.rollback_model(model_dir=models)
    _, last3 = joblib.load(latest)
    assert last3 == last1
