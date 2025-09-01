import types
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import optuna
import pytest

from tuning import distributed_search as ds

# ResourceCapabilities type for constructing fake capability objects
ResourceCapabilities = type(ds.monitor.capabilities)


def dummy_train(cfg):
    return 1.0 if cfg["x"] == 0.5 else 0.0


def _run(cap, tmp_path, monkeypatch, latest_usage=None, **kwargs):
    ds.monitor.capabilities = cap
    ds.monitor.latest_usage = latest_usage or {}

    saved = {}

    import importlib
    ms = importlib.import_module("models.model_store")

    def fake_save(params, store_dir=None):
        saved["params"] = params
        return "0"

    monkeypatch.setattr(ms, "save_tuned_params", fake_save)

    grid = {"x": [0.1, 0.5, 0.9]}
    space = {"x": lambda t: t.suggest_categorical("x", grid["x"])}
    sampler = optuna.samplers.GridSampler(grid)
    best, conc = ds.run_search(
        dummy_train,
        {},
        space,
        n_trials=3,
        sampler=sampler,
        storage=tmp_path / "study.db",
        **kwargs,
    )
    assert saved["params"] == best
    return best, conc


@pytest.mark.parametrize(
    "cap,expected",
    [
        (ResourceCapabilities(cpus=1, memory_gb=4, has_gpu=False, gpu_count=0), 1),
        (ResourceCapabilities(cpus=4, memory_gb=16, has_gpu=False, gpu_count=0), 4),
    ],
)
def test_cpu_scaling(cap, expected, tmp_path, monkeypatch):
    best, conc = _run(cap, tmp_path, monkeypatch)
    assert conc == expected
    assert best["x"] == 0.5


def test_gpu_scaling(tmp_path, monkeypatch):
    cap = ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=2)
    best, conc = _run(cap, tmp_path, monkeypatch, gpus_per_trial=1)
    assert conc == 2
    assert best["x"] == 0.5


def test_cpu_usage_throttling(tmp_path, monkeypatch):
    cap = ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=False, gpu_count=0)
    best, conc = _run(cap, tmp_path, monkeypatch, latest_usage={"cpu": 75})
    assert conc == 2  # 8 CPUs with 75% busy -> 2 slots
    assert best["x"] == 0.5
