import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import environment


def _run_low_spec_adjustment(tmp_path, monkeypatch, config_data=None):
    config_path = tmp_path / "config.yaml"
    if config_data is None:
        config_data = {
            "training": {
                "batch_size": 128,
                "n_jobs": 8,
                "other": "keep",
            },
            "strategy": {
                "symbols": ["EURUSD"],
                "risk_per_trade": 0.05,
            },
            "alerting": {
                "slack_webhook": "secret://alert-webhook",
            },
        }
    config_path.write_text(json.dumps(config_data, indent=2))

    monkeypatch.setenv("CONFIG_FILE", str(config_path))
    monkeypatch.setattr(environment, "CONFIG_FILE", config_path)
    monkeypatch.setattr(environment, "_check_dependencies", lambda: [])

    def _resolve_secrets(value):
        if isinstance(value, dict):
            return {k: _resolve_secrets(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_resolve_secrets(v) for v in value]
        if isinstance(value, str) and value.startswith("secret://"):
            return ""
        return value

    def _load_config_data(*, path=None, resolve_secrets=True):
        target = Path(path or config_path)
        with target.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return _resolve_secrets(data) if resolve_secrets else data

    def _save_config(cfg, path=None):
        target = Path(path or config_path)
        with target.open("w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)

    monkeypatch.setattr(environment, "load_config_data", _load_config_data)
    monkeypatch.setattr(environment, "save_config", _save_config)

    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=4 * 1_000_000_000),
        cpu_count=lambda: 2,
    )
    monkeypatch.setattr(environment, "psutil", fake_psutil)

    environment.ensure_environment()

    saved_text = config_path.read_text()
    saved_data = json.loads(saved_text)
    return config_data, saved_text, saved_data


def test_low_spec_adjustment_preserves_secret_placeholders(tmp_path, monkeypatch):
    _, saved_text, saved_data = _run_low_spec_adjustment(tmp_path, monkeypatch)

    assert "secret://alert-webhook" in saved_text
    assert saved_data["alerting"]["slack_webhook"] == "secret://alert-webhook"


def test_low_spec_adjustment_updates_training_values(tmp_path, monkeypatch):
    config_data, _, saved_data = _run_low_spec_adjustment(tmp_path, monkeypatch)

    assert saved_data["training"]["batch_size"] == 32
    assert saved_data["training"]["n_jobs"] == 1
    assert saved_data["training"]["other"] == "keep"

    assert config_data["training"]["batch_size"] == 128
    assert config_data["training"]["n_jobs"] == 8

    assert saved_data["strategy"] == config_data["strategy"]


def test_low_spec_adjustment_updates_flat_keys(tmp_path, monkeypatch):
    custom_cfg = {
        "batch_size": 256,
        "n_jobs": 4,
        "strategy": {
            "symbols": ["EURUSD"],
            "risk_per_trade": 0.05,
        },
    }

    original, _, saved = _run_low_spec_adjustment(tmp_path, monkeypatch, custom_cfg)

    assert original["batch_size"] == 256
    assert original["n_jobs"] == 4

    assert saved["batch_size"] == 32
    assert saved["n_jobs"] == 1
    assert saved["training"]["batch_size"] == 32
    assert saved["training"]["n_jobs"] == 1


def test_low_spec_adjustment_validates_with_app_config(tmp_path, monkeypatch):
    calls: list[dict[str, object]] = []

    class _SentinelConfig:
        def __init__(self, **values):
            calls.append(values)

    monkeypatch.setattr(environment, "AppConfig", _SentinelConfig)

    _run_low_spec_adjustment(tmp_path, monkeypatch)

    assert calls, "AppConfig validation should be invoked"
    resolved = calls[-1]
    training = resolved.get("training")
    assert isinstance(training, dict)
    assert training.get("batch_size") == 32
    assert training.get("n_jobs") == 1
