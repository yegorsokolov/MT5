import yaml
import pytest

import sys
from pathlib import Path
import importlib
import importlib.machinery
import types
import contextlib

import pytest

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

sys.modules.pop("yaml", None)
yaml = importlib.import_module("yaml")

import utils
from config_models import ConfigError


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
    assert cfg.services.service_cmds == {}


def test_invalid_risk_per_trade(tmp_path):
    cfg_file = write_cfg(tmp_path, {
        "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 1.5}
    })
    with pytest.raises(ConfigError) as exc:
        utils.load_config(cfg_file)
    assert "risk_per_trade" in str(exc.value)


def test_unknown_field(tmp_path):
    cfg_file = write_cfg(tmp_path, {
        "training": {"unknown": 1},
        "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.1}
    })
    with pytest.raises(ConfigError) as exc:
        utils.load_config(cfg_file)
    assert "Extra inputs" in str(exc.value)


def test_invalid_service_cmd(tmp_path):
    cfg_file = write_cfg(
        tmp_path,
        {
            "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.1},
            "services": {"service_cmds": {"queue": "python"}},
        },
    )
    with pytest.raises(ConfigError) as exc:
        utils.load_config(cfg_file)
    assert "service_cmds" in str(exc.value)


def test_top_level_strategy_keys(tmp_path):
    cfg_file = write_cfg(
        tmp_path,
        {
            "symbols": ["EURUSD"],
            "risk_per_trade": 0.05,
        },
    )
    cfg = utils.load_config(cfg_file)
    assert cfg.strategy.symbols == ["EURUSD"]
    assert cfg.strategy.risk_per_trade == 0.05


def test_feature_deduplication(tmp_path):
    cfg_file = write_cfg(
        tmp_path,
        {
            "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.1},
            "features": {"features": ["price", "price", "news"]},
        },
    )
    cfg = utils.load_config(cfg_file)
    assert cfg.features.features == ["price", "news"]


def test_alerting_recipient_string(tmp_path):
    cfg_file = write_cfg(
        tmp_path,
        {
            "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.1},
            "alerting": {
                "smtp": {
                    "host": "smtp.test",
                    "to": "ops@example.com,alerts@example.com",
                }
            },
        },
    )
    cfg = utils.load_config(cfg_file)
    assert cfg.alerting.smtp.recipients == [
        "ops@example.com",
        "alerts@example.com",
    ]


def test_app_config_get_returns_optional_section(tmp_path):
    cfg_file = write_cfg(
        tmp_path,
        {
            "strategy": {"symbols": ["EURUSD"], "risk_per_trade": 0.1},
            "alerting": {"slack_webhook": "https://hooks.slack.test/alert"},
        },
    )
    cfg = utils.load_config(cfg_file)
    alerting_section = cfg.get("alerting")
    assert alerting_section is cfg.alerting
    assert alerting_section.slack_webhook == "https://hooks.slack.test/alert"
