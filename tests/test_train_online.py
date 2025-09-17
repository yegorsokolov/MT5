import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

state_manager_stub = types.ModuleType("state_manager")
state_manager_stub.watch_config = lambda cfg: types.SimpleNamespace(stop=lambda: None)
sys.modules.setdefault("state_manager", state_manager_stub)

utils_stub = types.ModuleType("utils")
utils_stub.load_config = lambda: {"seed": 0, "drawdown_limit": 0.2}
sys.modules.setdefault("utils", utils_stub)

log_utils_stub = types.ModuleType("log_utils")
log_utils_stub.setup_logging = lambda: None
log_utils_stub.log_exceptions = lambda f: f
sys.modules.setdefault("log_utils", log_utils_stub)


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

import train_online


class DummyModel:
    def __init__(self) -> None:
        self.calls: list[tuple[dict[str, float], int]] = []

    def learn_one(self, x, y):
        self.calls.append((x, y))
        return self


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
    monkeypatch.setattr(train_online, "watch_config", lambda cfg: types.SimpleNamespace(stop=lambda: None))

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

    latest_path = model_dir / "online_latest.joblib"
    state = joblib.load(latest_path)
    assert state["feature_columns"] == ["feat_a", "feat_b"]
    assert state["label_column"] == "tb_label"
    assert isinstance(state["last_train_ts"], pd.Timestamp)
    assert dummy_model.calls
    assert saved_metadata["metadata"]["preprocessing"]["feature_columns"] == ["feat_a", "feat_b"]
    assert saved_metadata["metadata"]["preprocessing"]["label_column"] == "tb_label"
