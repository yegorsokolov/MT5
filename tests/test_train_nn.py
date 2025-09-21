import importlib
import importlib.util
from types import SimpleNamespace

import pytest

from config_models import AppConfig


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
