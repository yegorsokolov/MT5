from __future__ import annotations

import base64
from pathlib import Path
import sys

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment import runtime_secrets


def _read_env(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    if not path.exists():
        return data
    for line in path.read_text().splitlines():
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def test_generates_all_sections(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    result = runtime_secrets.ensure_runtime_secrets(env_file)
    values = _read_env(env_file)

    expected_keys = {
        "CONFIG_ADMIN_KEY",
        "CONFIG_READER_KEY",
        "API_KEY",
        "AUDIT_LOG_SECRET",
        "CHECKPOINT_AES_KEY",
        "DECISION_AES_KEY",
    }
    assert expected_keys.issubset(values.keys())
    assert set(result.created) == expected_keys
    assert not result.rotated

    # AES keys must decode to 32 bytes.
    for key in ("CHECKPOINT_AES_KEY", "DECISION_AES_KEY"):
        decoded = base64.b64decode(values[key])
        assert len(decoded) == 32


def test_reuse_existing_values(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    first = runtime_secrets.ensure_runtime_secrets(env_file)
    second = runtime_secrets.ensure_runtime_secrets(env_file)

    assert first.values == second.values
    assert not second.created
    assert not second.rotated
    assert not second.written


def test_rotate_single_secret(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    first = runtime_secrets.ensure_runtime_secrets(env_file)
    rotated = runtime_secrets.ensure_runtime_secrets(env_file, rotate=["API_KEY"])

    assert set(rotated.rotated) == {"API_KEY"}
    assert rotated.values["API_KEY"] != first.values["API_KEY"]
    assert rotated.values["CONFIG_ADMIN_KEY"] == first.values["CONFIG_ADMIN_KEY"]


def test_skip_encryption(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    runtime_secrets.ensure_runtime_secrets(env_file, include_encryption=False)
    values = _read_env(env_file)

    assert "CHECKPOINT_AES_KEY" not in values
    assert "DECISION_AES_KEY" not in values
    assert "CONFIG_ADMIN_KEY" in values
    assert "API_KEY" in values


def test_unknown_rotation_raises(tmp_path: Path) -> None:
    env_file = tmp_path / "runtime.env"
    with pytest.raises(RuntimeError):
        runtime_secrets.ensure_runtime_secrets(env_file, rotate=["NOT_A_SECRET"])
