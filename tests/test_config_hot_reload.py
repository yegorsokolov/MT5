import os
import sys
import time
import importlib
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
sys.modules.pop("yaml", None)
yaml = importlib.import_module("yaml")

from utils import load_config
from mt5.state_manager import watch_config


def test_config_hot_reload(tmp_path):
    cfg_data = {
        "training": {},
        "features": {"features": []},
        "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.1, "use_kalman_smoothing": False},
        "services": {},
    }
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(cfg_data))

    cfg = load_config(cfg_file)
    cfg_id = id(cfg)
    observer = watch_config(cfg, cfg_file)
    try:
        data = yaml.safe_load(cfg_file.read_text())
        data["strategy"]["risk_per_trade"] = 0.2
        data["strategy"]["use_kalman_smoothing"] = True
        cfg_file.write_text(yaml.safe_dump(data))

        deadline = time.time() + 5
        while time.time() < deadline:
            if cfg.strategy.risk_per_trade == 0.2 and cfg.strategy.use_kalman_smoothing:
                break
            time.sleep(0.1)
        assert cfg.strategy.risk_per_trade == 0.2
        assert cfg.strategy.use_kalman_smoothing is True
        assert cfg.strategy.symbols == ["EURUSD"]
        assert id(cfg) == cfg_id
    finally:
        observer.stop()
        observer.join()
