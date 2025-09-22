import contextlib
import importlib.machinery
import sys
from copy import deepcopy
from pathlib import Path
import types

import pytest
import yaml

sys.path.append(str(Path(__file__).resolve().parents[1]))

sys.modules.pop("config_models", None)
sys.modules.pop("pydantic", None)
sys.modules.pop("utils", None)

mlflow_stub = types.ModuleType("mlflow")
mlflow_stub.set_tracking_uri = lambda *a, **k: None
mlflow_stub.set_experiment = lambda *a, **k: None
mlflow_stub.start_run = lambda *a, **k: contextlib.nullcontext()
mlflow_stub.end_run = lambda *a, **k: None
mlflow_stub.log_dict = lambda *a, **k: None
mlflow_stub.log_param = lambda *a, **k: None
mlflow_stub.log_params = lambda *a, **k: None
mlflow_stub.log_metric = lambda *a, **k: None
mlflow_stub.log_metrics = lambda *a, **k: None
mlflow_stub.log_artifact = lambda *a, **k: None
mlflow_stub.log_artifacts = lambda *a, **k: None
mlflow_stub.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
sys.modules.setdefault("mlflow", mlflow_stub)

import utils
from mt5.config_models import ConfigError

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "config"


def load_config_fixture(name: str) -> dict:
    path = FIXTURE_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        loaded = yaml.safe_load(fh) or {}
    assert isinstance(loaded, dict)
    return deepcopy(loaded)


def write_cfg(tmp_path: Path, data: dict) -> Path:
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.safe_dump(data))
    return cfg_file


def test_defaults(tmp_path: Path) -> None:
    cfg_data = load_config_fixture("minimal.yaml")
    cfg_file = write_cfg(tmp_path, cfg_data)
    assert yaml.safe_load(cfg_file.read_text()) == cfg_data

    cfg = utils.load_config(cfg_file)
    assert cfg.training.seed == 42
    assert cfg.features.latency_threshold == 0.0
    assert cfg.services.service_cmds == {}


def test_invalid_risk_per_trade(tmp_path: Path) -> None:
    cfg_data = load_config_fixture("minimal.yaml")
    cfg_data["strategy"]["risk_per_trade"] = 1.5
    cfg_file = write_cfg(tmp_path, cfg_data)

    with pytest.raises(ConfigError) as exc:
        utils.load_config(cfg_file)
    assert "risk_per_trade" in str(exc.value)


def test_unknown_field(tmp_path: Path) -> None:
    cfg_data = load_config_fixture("minimal.yaml")
    cfg_data["training"] = {"unknown": 1}
    cfg_file = write_cfg(tmp_path, cfg_data)

    with pytest.raises(ConfigError) as exc:
        utils.load_config(cfg_file)
    assert "Extra inputs" in str(exc.value)


def test_invalid_service_cmd(tmp_path: Path) -> None:
    cfg_data = load_config_fixture("minimal.yaml")
    cfg_data["services"] = {"service_cmds": {"queue": "python"}}
    cfg_file = write_cfg(tmp_path, cfg_data)

    with pytest.raises(ConfigError) as exc:
        utils.load_config(cfg_file)
    assert "service_cmds" in str(exc.value)


def test_top_level_strategy_keys(tmp_path: Path) -> None:
    cfg_data = load_config_fixture("top_level_strategy.yaml")
    cfg_file = write_cfg(tmp_path, cfg_data)

    cfg = utils.load_config(cfg_file)
    assert cfg.strategy.symbols == ["EURUSD"]
    assert cfg.strategy.risk_per_trade == 0.05


def test_feature_deduplication(tmp_path: Path) -> None:
    cfg_data = load_config_fixture("minimal.yaml")
    cfg_data["features"] = {"features": ["price", "price", "news"]}
    cfg_file = write_cfg(tmp_path, cfg_data)

    raw = yaml.safe_load(cfg_file.read_text())
    assert raw["features"]["features"] == ["price", "price", "news"]

    cfg = utils.load_config(cfg_file)
    assert cfg.features.features == ["price", "news"]


def test_alerting_recipient_string(tmp_path: Path) -> None:
    cfg_data = load_config_fixture("minimal.yaml")
    cfg_data["alerting"] = {
        "smtp": {
            "host": "smtp.test",
            "to": "ops@example.com,alerts@example.com",
        }
    }
    cfg_file = write_cfg(tmp_path, cfg_data)

    raw = yaml.safe_load(cfg_file.read_text())
    assert raw["alerting"]["smtp"]["to"] == "ops@example.com,alerts@example.com"

    cfg = utils.load_config(cfg_file)
    assert cfg.alerting.smtp.recipients == [
        "ops@example.com",
        "alerts@example.com",
    ]


def test_app_config_get_returns_optional_section(tmp_path: Path) -> None:
    cfg_data = load_config_fixture("minimal.yaml")
    cfg_data["alerting"] = {
        "telegram_bot_token": "12345:ABC",
        "telegram_chat_id": 98765,
    }
    cfg_file = write_cfg(tmp_path, cfg_data)

    cfg = utils.load_config(cfg_file)
    alerting_section = cfg.get("alerting")
    assert alerting_section is cfg.alerting
    assert alerting_section.telegram_bot_token == "12345:ABC"
    assert alerting_section.telegram_chat_id == "98765"
