import pytest
import sys
from pathlib import Path
import threading
import yaml
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.set_tracking_uri = lambda *a, **k: None
mlflow_stub.set_experiment = lambda *a, **k: None
mlflow_stub.start_run = lambda *a, **k: types.SimpleNamespace(
    __enter__=lambda self: None,
    __exit__=lambda self, exc_type, exc, tb: False,
)
mlflow_stub.log_dict = lambda *a, **k: None
sys.modules.setdefault("mlflow", mlflow_stub)

log_utils_stub = types.ModuleType("log_utils")
log_utils_stub.LOG_DIR = Path(".")
sys.modules.setdefault("log_utils", log_utils_stub)

env_stub = types.ModuleType("utils.environment")
env_stub.ensure_environment = lambda: None
sys.modules.setdefault("utils.environment", env_stub)

import utils
from utils import update_config

@pytest.mark.parametrize('key', [
    'max_daily_loss',
    'max_drawdown',
    'max_var',
    'max_stress_loss',
    'max_cvar',
    'risk_per_trade',
    'rl_max_position',
])
def test_update_protected_key_raises(key):
    with pytest.raises(ValueError):
        update_config(key, 0, 'test')


def test_concurrent_updates(tmp_path, monkeypatch):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(
        yaml.safe_dump(
            {
                "seed": 0,
                "strategy": {
                    "risk_per_trade": 0.1,
                    "symbols": ["EURUSD"],
                },
                "threshold": 0,
            }
        )
    )
    monkeypatch.setenv("CONFIG_FILE", str(cfg_file))

    # bypass pydantic validation for this concurrency test
    class _Cfg(dict):
        def model_dump(self):
            return dict(self)

    monkeypatch.setattr(
        utils,
        "load_config",
        lambda path=None: _Cfg(yaml.safe_load(cfg_file.read_text()) or {}),
    )
    monkeypatch.setattr(
        utils,
        "save_config",
        lambda cfg: cfg_file.write_text(yaml.safe_dump(cfg)),
    )

    log_file = tmp_path / "config_changes.csv"
    log_file.parent.mkdir(exist_ok=True)
    monkeypatch.setattr(utils, "_LOG_PATH", log_file)

    def worker(i: int) -> None:
        update_config("threshold", i + 1, f"reason{i}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    lines = log_file.read_text().strip().splitlines()
    assert len(lines) == 5
    values = [int(line.split(",")[3]) for line in lines]
    assert set(values) == {1, 2, 3, 4, 5}

