import importlib
import importlib.util
import sys
import types
from types import SimpleNamespace

import pytest
from mt5.config_models import AppConfig


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch dependency unavailable",
)
def test_train_nn_module_exposes_main():
    module = importlib.import_module("train_nn")
    assert hasattr(module, "main")


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch dependency unavailable",
)
def test_batch_size_backoff_respects_typed_training_config(monkeypatch):
    module = importlib.import_module("train_nn")

    cfg = AppConfig.model_validate(
        {
            "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.05},
            "training": {
                "batch_size": 128,
                "eval_batch_size": 64,
                "min_batch_size": 16,
            },
        }
    )
    cfg_dict = cfg.training.model_dump()

    attempts = {"count": 0}

    monkeypatch.setattr(
        module.monitor,
        "capabilities",
        SimpleNamespace(memory_gb=4),
        raising=False,
    )

    monkeypatch.setattr(
        module.psutil,
        "Process",
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=0)),
    )
    monkeypatch.setattr(module.torch.cuda, "empty_cache", lambda: None)

    def _train(batch_size, eval_batch_size):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("CUDA out of memory")
        assert batch_size == 64
        assert eval_batch_size == 32
        return batch_size, eval_batch_size

    result = module.batch_size_backoff(cfg_dict, _train)

    assert result == (64, 32)
    assert cfg_dict["batch_size"] == 64
    assert cfg_dict["eval_batch_size"] == 32
    assert attempts["count"] == 2


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch dependency unavailable",
)
def test_train_nn_main_lazily_starts_orchestrator(monkeypatch):
    pd = pytest.importorskip("pandas")

    class _Stop(Exception):
        pass

    orchestrator_calls = {"count": 0}
    orch_module = types.ModuleType("core.orchestrator")

    class _StubOrchestrator:
        @staticmethod
        def start():
            orchestrator_calls["count"] += 1
            return object()

    orch_module.Orchestrator = _StubOrchestrator
    orch_module.__spec__ = importlib.machinery.ModuleSpec(
        "core.orchestrator", loader=None
    )
    monkeypatch.setitem(sys.modules, "core.orchestrator", orch_module)
    monkeypatch.delitem(sys.modules, "train_nn", raising=False)
    monkeypatch.delitem(sys.modules, "mt5.train_nn", raising=False)

    module = importlib.import_module("train_nn")

    assert orchestrator_calls["count"] == 0

    monitor_stub = SimpleNamespace(
        capabilities=SimpleNamespace(capability_tier=lambda: "lite", cpus=1)
    )
    monkeypatch.setattr(module, "monitor", monitor_stub, raising=False)
    fake_df = pd.DataFrame({"mid": [1.0], "Symbol": ["SYM"]})
    monkeypatch.setattr(
        module,
        "load_history_config",
        lambda *a, **k: fake_df.copy(),
        raising=False,
    )

    def _raise_make_features(*_args, **_kwargs):
        raise _Stop()

    monkeypatch.setattr(module, "make_features", _raise_make_features, raising=False)

    cfg = {"symbols": ["SYM"], "stream_history": False, "seed": 1}

    with pytest.raises(_Stop):
        module.main(0, 1, cfg)

    assert orchestrator_calls["count"] == 1

    with pytest.raises(_Stop):
        module.main(0, 1, cfg)

    assert orchestrator_calls["count"] == 1
    module._ORCHESTRATOR_STARTED = False
    module._ORCHESTRATOR_INSTANCE = None
    monkeypatch.delitem(sys.modules, "train_nn", raising=False)
    monkeypatch.delitem(sys.modules, "mt5.train_nn", raising=False)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch dependency unavailable",
)
def test_launch_aggregates_multi_seed_results(monkeypatch, caplog):
    module = importlib.import_module("train_nn")

    seeds = [101, 202, 303]
    metrics = {seed: 0.4 + idx * 0.1 for idx, seed in enumerate(seeds)}
    submitted: list[int] = []

    monkeypatch.setattr(module, "ensure_orchestrator_started", lambda: None)
    monkeypatch.setattr(module, "cluster_available", lambda: True)

    def _fake_submit(fn, *args, **kwargs):
        cfg_seed = args[2]["seed"]
        submitted.append(cfg_seed)
        return metrics[cfg_seed]

    monkeypatch.setattr(module, "submit", _fake_submit)
    monkeypatch.setattr(module, "main", lambda *a, **k: 0.0)

    cfg = {"seed": 0, "seeds": seeds}

    with caplog.at_level("INFO", logger=module.logger.name):
        aggregated = module.launch(cfg)

    expected = sum(metrics.values()) / len(metrics)
    assert aggregated == pytest.approx(expected)
    assert submitted == seeds

    for seed in seeds:
        assert any(f"Seed {seed} score" in message for message in caplog.messages)
    assert any(
        "Aggregated mean score" in message for message in caplog.messages
    )


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch dependency unavailable",
)
def test_launch_initializes_ray_before_cluster_check(monkeypatch):
    module = importlib.import_module("train_nn")
    ray_utils = importlib.import_module("mt5.ray_utils")

    calls: list[str] = []
    submit_seeds: list[int] = []

    def _fake_init(*_args, **_kwargs):
        calls.append("init")
        return True

    def _fake_cluster_available():
        calls.append("cluster")
        return True

    def _fake_submit(fn, *args, **kwargs):
        del fn  # unused in test
        calls.append("submit")
        cfg_seed = args[2]["seed"]
        submit_seeds.append(cfg_seed)
        return float(cfg_seed)

    def _fake_shutdown():
        calls.append("shutdown")

    monkeypatch.setattr(ray_utils, "init", _fake_init)
    monkeypatch.setattr(ray_utils, "cluster_available", _fake_cluster_available)
    monkeypatch.setattr(ray_utils, "submit", _fake_submit)
    monkeypatch.setattr(ray_utils, "shutdown", _fake_shutdown)

    monkeypatch.setattr(module, "ray_init", ray_utils.init)
    monkeypatch.setattr(module, "cluster_available", ray_utils.cluster_available)
    monkeypatch.setattr(module, "submit", ray_utils.submit)
    monkeypatch.setattr(module, "ray_shutdown", ray_utils.shutdown)
    monkeypatch.setattr(module, "ensure_orchestrator_started", lambda: None)
    monkeypatch.setattr(module.mlflow, "log_metric", lambda *a, **k: None, raising=False)

    cfg = {"seed": 0, "seeds": [7, 11]}
    aggregated = module.launch(cfg)

    expected = sum(cfg["seeds"]) / len(cfg["seeds"])
    assert aggregated == pytest.approx(expected)
    assert submit_seeds == cfg["seeds"]

    assert calls[0] == "init"
    cluster_index = calls.index("cluster")
    assert cluster_index > calls.index("init")
    assert calls.count("submit") == len(cfg["seeds"])
    assert calls.count("shutdown") == 1
    assert calls.index("shutdown") > max(
        idx for idx, name in enumerate(calls) if name == "submit"
    )

    module._ORCHESTRATOR_STARTED = False
    module._ORCHESTRATOR_INSTANCE = None
