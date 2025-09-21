import logging
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

from utils import environment

FIXTURE_DIR = Path(__file__).resolve().parent / "data" / "config"


def load_config_fixture(name: str) -> dict:
    path = FIXTURE_DIR / name
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    assert isinstance(data, dict)
    return deepcopy(data)


def _run_low_spec_adjustment(tmp_path, monkeypatch, config_name="environment_low_spec.yaml"):
    config_path = tmp_path / "config.yaml"
    config_data = load_config_fixture(config_name)
    original = deepcopy(config_data)
    config_path.write_text(yaml.safe_dump(config_data))

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
            data = yaml.safe_load(fh) or {}
        return _resolve_secrets(data) if resolve_secrets else data

    def _save_config(cfg, path=None):
        target = Path(path or config_path)
        with target.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh)

    monkeypatch.setattr(environment, "load_config_data", _load_config_data)
    monkeypatch.setattr(environment, "save_config", _save_config)

    fake_psutil = SimpleNamespace(
        virtual_memory=lambda: SimpleNamespace(total=4 * 1_000_000_000),
        cpu_count=lambda: 2,
    )
    monkeypatch.setattr(environment, "psutil", fake_psutil)

    environment.ensure_environment()

    saved_text = config_path.read_text()
    saved_data = yaml.safe_load(saved_text) or {}
    return original, saved_text, saved_data


def test_low_spec_adjustment_preserves_secret_placeholders(tmp_path, monkeypatch):
    _, saved_text, saved_data = _run_low_spec_adjustment(tmp_path, monkeypatch)

    assert "secret://alert-webhook" in saved_text
    assert saved_data["alerting"]["slack_webhook"] == "secret://alert-webhook"


def test_low_spec_adjustment_updates_training_values(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.WARNING, environment.__name__)
    config_data, _, saved_data = _run_low_spec_adjustment(tmp_path, monkeypatch)

    assert saved_data["training"]["batch_size"] == 32
    assert saved_data["training"]["n_jobs"] == 1
    assert saved_data["training"]["use_pseudo_labels"] is True

    assert config_data["training"]["batch_size"] == 128
    assert config_data["training"]["n_jobs"] == 8
    assert config_data["training"]["use_pseudo_labels"] is True

    assert saved_data["strategy"] == config_data["strategy"]

    assert any("low-spec hardware" in record.getMessage() for record in caplog.records)


def test_low_spec_adjustment_updates_flat_keys(tmp_path, monkeypatch):
    original, _, saved = _run_low_spec_adjustment(
        tmp_path, monkeypatch, "environment_flat.yaml"
    )

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


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        ("requests>=2.31", ("requests", "requests")),
        ("uvicorn[standard]==0.20.0", ("uvicorn", "uvicorn")),
        (
            "some-package[extra1,extra2]; python_version<'3.10'",
            ("some-package", "some_package"),
        ),
        (
            "rich @ git+https://github.com/Textualize/rich.git",
            ("rich", "rich"),
        ),
        ("numpy>=1.23  # inline comment", ("numpy", "numpy")),
        ("   # comment line", None),
        ("", None),
    ],
)
def test_parse_requirement_line_normalises_inputs(line, expected):
    assert environment._parse_requirement_line(line) == expected


def test_check_dependencies_skips_comments_and_normalises(tmp_path, monkeypatch):
    requirements = [
        "# global comment",
        "requests>=2.31",
        "uvicorn[standard]==0.20.0 ; python_version >= '3.8'",
        "some-package[extra1]; python_version<'3.10'  # trailing comment",
        "rich @ git+https://github.com/Textualize/rich.git",
        "   ",
    ]

    req_file = tmp_path / "requirements-core.txt"
    req_file.write_text("\n".join(requirements))

    monkeypatch.setattr(environment, "REQ_FILE", req_file)

    requested: list[str] = []

    def _fake_find_loader(name: str):
        requested.append(name)
        return object()

    monkeypatch.setattr(environment.pkgutil, "find_loader", _fake_find_loader)

    assert environment._check_dependencies() == []
    assert requested == ["requests", "uvicorn", "some_package", "rich"]
