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
