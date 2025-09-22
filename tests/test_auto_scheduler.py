import asyncio
import json
from pathlib import Path

import pandas as pd
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import types
sklearn_cluster = types.SimpleNamespace(KMeans=object)
sys.modules.setdefault("sklearn", types.SimpleNamespace(cluster=sklearn_cluster))
sys.modules.setdefault("sklearn.cluster", sklearn_cluster)
sys.modules["telemetry"] = types.SimpleNamespace(
    get_tracer=lambda name: types.SimpleNamespace(
        start_as_current_span=lambda *a, **k: contextlib.nullcontext()
    ),
    get_meter=lambda name: types.SimpleNamespace(
        create_counter=lambda *a, **k: types.SimpleNamespace(add=lambda *a, **k: None)
    ),
)
import contextlib

from tuning.auto_scheduler import AutoScheduler
import tuning.auto_scheduler as auto
from mt5.model_registry import ModelRegistry, ResourceCapabilities
from models import model_store
import models.hot_reload as hot


class DummyMonitor:
    def __init__(self, caps):
        self.capabilities = caps
        self.capability_tier = caps.capability_tier()

    def start(self):  # pragma: no cover - no background task
        pass


def _run_scheduler(tmp_path, monkeypatch):
    monkeypatch.setattr(model_store, "STORE_DIR", tmp_path)
    # regime detection always returns regime 0
    monkeypatch.setattr(auto.regime_detection, "detect_regimes", lambda df: pd.Series([0], index=df.index))

    # patch ray tune run
    results = {"score": 1.0}

    class DummyAnalysis:
        best_config = {"lr": 0.1}
        best_result = results
        best_score = 1.0

    def fake_run(func, config=None, num_samples=1, search_alg=None):
        func({"lr": 0.1})
        return DummyAnalysis()

    monkeypatch.setattr(auto.tune, "run", fake_run)
    monkeypatch.setattr(auto.tune, "with_parameters", lambda f, **k: lambda cfg: f(cfg))
    monkeypatch.setattr(auto.tune, "report", lambda **k: None)

    hot_calls = []
    monkeypatch.setattr(auto, "hot_reload", lambda params: hot_calls.append(params))

    sched = AutoScheduler(lambda p, d: 1.0, {}, lambda: pd.DataFrame({"a": [1]}), n_samples=1, margin=0.05)
    asyncio.get_event_loop().run_until_complete(sched.run_once())
    return hot_calls


def test_scheduler_stores_and_registry_reloads(tmp_path, monkeypatch):
    hot_calls = _run_scheduler(tmp_path, monkeypatch)
    assert hot_calls == [{"lr": 0.1}]

    files = list(tmp_path.glob("tuned_*.json"))
    assert files
    meta = json.loads(files[0].read_text())
    assert meta["regime"] == 0
    assert meta["params"] == {"lr": 0.1}
    assert meta["score"] == 1.0

    monitor = DummyMonitor(ResourceCapabilities(1, 1, False, gpu_count=0))
    hot_calls.clear()
    monkeypatch.setattr(hot, "hot_reload", lambda params, model_id=None: hot_calls.append(params))
    reg = ModelRegistry(monitor, auto_refresh=False, regime_getter=lambda: 0)
    reg.refresh()
    assert hot_calls == [{"lr": 0.1}]

    # Second run with insufficient improvement should not deploy
    class DummyAnalysis2:
        best_config = {"lr": 0.2}
        best_result = {"score": 1.02}
        best_score = 1.02

    def fake_run2(func, config=None, num_samples=1, search_alg=None):
        func({"lr": 0.2})
        return DummyAnalysis2()

    monkeypatch.setattr(auto.tune, "run", fake_run2)
    hot_calls.clear()
    sched = AutoScheduler(lambda p, d: 1.02, {}, lambda: pd.DataFrame({"a": [1]}), n_samples=1, margin=0.05)
    asyncio.get_event_loop().run_until_complete(sched.run_once())
    assert hot_calls == []
