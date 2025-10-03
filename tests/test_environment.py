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
    monkeypatch.setattr(environment, "_python_major_minor", lambda: (3, 12))
    monkeypatch.setattr(environment, "_distribution_installed", lambda *a, **k: True)

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

    assert "secret://alert-bot-token" in saved_text
    assert saved_data["alerting"]["telegram_bot_token"] == "secret://alert-bot-token"
    assert saved_data["alerting"]["telegram_chat_id"] == "secret://alert-chat"


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
    result = environment._parse_requirement_line(line)
    if "some-package" in line and environment.Marker is not None:
        assert result in {expected, None}
    else:
        assert result == expected


def test_check_dependencies_skips_comments_and_normalises(tmp_path, monkeypatch):
    requirements = [
        "# global comment",
        "requests>=2.31",
        "uvicorn[standard]==0.20.0 ; python_version >= '3.8'",
        "some-package[extra1]; python_version<'3.10'  # trailing comment",
        "rich @ git+https://github.com/Textualize/rich.git",
        "   ",
    ]

    req_file = tmp_path / "requirements.txt"
    req_file.write_text("\n".join(requirements))

    monkeypatch.setattr(environment, "REQ_FILE", req_file)

    requested: list[str] = []
    fallback: list[str] = []

    def _fake_distribution(name: str):
        requested.append(name)
        if name == "rich":
            raise environment.metadata.PackageNotFoundError  # pragma: no cover
        return object()

    monkeypatch.setattr(environment.metadata, "distribution", _fake_distribution)
    monkeypatch.setattr(
        environment,
        "find_spec",
        lambda name: (fallback.append(name), None)[1],
    )

    assert environment._check_dependencies() == ["rich"]
    assert requested == ["requests", "uvicorn", "rich"]
    assert fallback == ["rich"]


def test_check_dependencies_falls_back_to_module_search(tmp_path, monkeypatch):
    req_file = tmp_path / "requirements.txt"
    req_file.write_text("example-pkg\n")

    monkeypatch.setattr(environment, "REQ_FILE", req_file)

    monkeypatch.setattr(
        environment.metadata,
        "distribution",
        lambda name: (_ for _ in ()).throw(environment.metadata.PackageNotFoundError()),
    )

    calls: list[str] = []

    def _fake_find_spec(name: str):
        calls.append(name)
        return object()

    monkeypatch.setattr(environment, "find_spec", _fake_find_spec)

    assert environment._check_dependencies() == []
    assert calls == ["example_pkg"]


def test_collect_missing_dependencies_skips_mt5_with_bridge_env(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(environment, "_check_dependencies", lambda: ["MetaTrader5"])
    monkeypatch.setattr(environment.sys, "platform", "linux")
    monkeypatch.setattr(environment, "find_spec", lambda name: object())
    monkeypatch.setenv("WINE_PYTHON", "/fake/windows/python.exe")

    assert environment._collect_missing_dependencies() == []


def test_collect_missing_dependencies_skips_mt5_with_login_instructions(tmp_path, monkeypatch):
    instructions = tmp_path / "LOGIN_INSTRUCTIONS_WINE.txt"
    instructions.write_text("bridge configured")

    monkeypatch.setattr(environment, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(environment, "_check_dependencies", lambda: ["MetaTrader5"])
    monkeypatch.setattr(environment.sys, "platform", "linux")
    monkeypatch.setattr(environment, "find_spec", lambda name: object())

    assert environment._collect_missing_dependencies() == []


def test_collect_missing_dependencies_requires_mt5_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(environment, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(environment, "_check_dependencies", lambda: ["MetaTrader5"])
    monkeypatch.setattr(environment.sys, "platform", "win32")
    monkeypatch.setattr(environment, "find_spec", lambda name: object())
    monkeypatch.setenv("WINE_PYTHON", "/fake/windows/python.exe")

    assert environment._collect_missing_dependencies() == ["MetaTrader5"]


def test_read_env_pairs_normalises_quotes_and_escapes(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                'KEY="value"',
                'ESCAPED="line\\nnext"',
                "SINGLE=' spaced '",
                "PLAIN=bare",
            ]
        )
    )

    monkeypatch.setattr(environment, "PROJECT_ROOT", tmp_path)

    pairs = environment._read_env_pairs(env_path)

    assert pairs["KEY"] == "value"
    assert pairs["ESCAPED"] == "line\nnext"
    assert pairs["SINGLE"] == " spaced "
    assert pairs["PLAIN"] == "bare"


def test_check_env_loaded_accepts_quoted_values(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text('KEY="value"\nESCAPED=line\\nnext\n')

    monkeypatch.setattr(environment, "PROJECT_ROOT", tmp_path)
    monkeypatch.setenv("KEY", "value")
    monkeypatch.setenv("ESCAPED", "line\nnext")

    result = environment._check_env_loaded()

    assert result["status"] == "passed"
    assert "Loaded 2 environment variables" in result["detail"]


def test_check_distributed_dependencies_skips_on_py313(monkeypatch):
    monkeypatch.setattr(environment, "_python_major_minor", lambda: (3, 13))
    monkeypatch.setattr(environment.sys, "platform", "linux")
    monkeypatch.setattr(environment, "_distribution_installed", lambda *a, **k: True)
    result = environment._check_distributed_dependencies()

    assert result["status"] == "skipped"
    detail = result["detail"].lower()
    assert "fall back" in detail or "fallback" in detail
    assert "uvloop" in detail


def test_check_distributed_dependencies_reports_missing(monkeypatch):
    monkeypatch.setattr(environment, "_python_major_minor", lambda: (3, 12))
    monkeypatch.setattr(environment.sys, "platform", "linux")

    def _fake_distribution(pkg: str, module: str) -> bool:
        if pkg == "ray":
            return False
        return True

    monkeypatch.setattr(environment, "_distribution_installed", _fake_distribution)

    result = environment._check_distributed_dependencies()

    assert result["status"] == "failed"
    assert "ray" in result["detail"].lower()
    assert "pip install" in (result["followup"] or "")


def test_check_distributed_dependencies_passes(monkeypatch):
    monkeypatch.setattr(environment, "_python_major_minor", lambda: (3, 12))
    monkeypatch.setattr(environment.sys, "platform", "linux")
    monkeypatch.setattr(environment, "_distribution_installed", lambda *a, **k: True)

    result = environment._check_distributed_dependencies()

    assert result["status"] == "passed"


def test_check_distributed_dependencies_requires_uvloop(monkeypatch):
    monkeypatch.setattr(environment, "_python_major_minor", lambda: (3, 12))
    monkeypatch.setattr(environment.sys, "platform", "linux")

    def _fake_distribution(pkg: str, module: str) -> bool:
        return module != "uvloop"

    monkeypatch.setattr(environment, "_distribution_installed", _fake_distribution)

    result = environment._check_distributed_dependencies()

    assert result["status"] == "failed"
    assert "uvloop" in result["detail"].lower()
