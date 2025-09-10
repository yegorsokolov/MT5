import yaml
import pytest

import sys
from pathlib import Path
import importlib

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.pop("yaml", None)
yaml = importlib.import_module("yaml")

import utils


def write_cfg(tmp_path, data):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(data))
    return cfg_file


def test_defaults(tmp_path):
    cfg_file = write_cfg(tmp_path, {
        "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.1}
    })
    cfg = utils.load_config(cfg_file)
    assert cfg.training.seed == 42
    assert cfg.features.latency_threshold == 0.0


def test_invalid_risk_per_trade(tmp_path):
    cfg_file = write_cfg(tmp_path, {
        "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 1.5}
    })
    with pytest.raises(ValueError) as exc:
        utils.load_config(cfg_file)
    assert "risk_per_trade" in str(exc.value)


def test_unknown_field(tmp_path):
    cfg_file = write_cfg(tmp_path, {
        "training": {"unknown": 1},
        "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.1}
    })
    with pytest.raises(ValueError) as exc:
        utils.load_config(cfg_file)
    assert "Extra inputs" in str(exc.value)
