import yaml
import pytest
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

import utils


def write_config(tmp_path, data):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(data))
    return cfg_file


def test_invalid_risk_per_trade(monkeypatch, tmp_path):
    cfg_file = write_config(tmp_path, {"risk_per_trade": -0.1, "symbols": ["XAUUSD"]})
    monkeypatch.setenv("CONFIG_FILE", str(cfg_file))
    with pytest.raises(ValueError) as exc:
        utils.load_config()
    assert "risk_per_trade" in str(exc.value)


def test_symbols_must_be_list(monkeypatch, tmp_path):
    cfg_file = write_config(tmp_path, {"risk_per_trade": 0.1, "symbols": "XAUUSD"})
    monkeypatch.setenv("CONFIG_FILE", str(cfg_file))
    with pytest.raises(ValueError) as exc:
        utils.load_config()
    assert "symbols" in str(exc.value)
