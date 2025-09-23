from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deployment import mlflow_credentials


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


def test_generates_credentials(tmp_path: Path) -> None:
    env_file = tmp_path / "mlflow.env"
    result = mlflow_credentials.ensure_mlflow_credentials(env_file)

    assert set(result.created) == {"MLFLOW_USER", "MLFLOW_PASS"}
    assert not result.rotated
    assert result.written

    values = _read_env(env_file)
    assert values["MLFLOW_USER"].startswith("mlflow-")
    assert len(values["MLFLOW_PASS"]) >= 32


def test_reuses_existing_credentials(tmp_path: Path) -> None:
    env_file = tmp_path / "mlflow.env"
    first = mlflow_credentials.ensure_mlflow_credentials(env_file)
    second = mlflow_credentials.ensure_mlflow_credentials(env_file)

    assert first.values == second.values
    assert not second.created
    assert not second.rotated
    assert not second.written


def test_rotate_password(tmp_path: Path) -> None:
    env_file = tmp_path / "mlflow.env"
    first = mlflow_credentials.ensure_mlflow_credentials(env_file)
    rotated = mlflow_credentials.ensure_mlflow_credentials(env_file, rotate_password=True)

    assert rotated.values["MLFLOW_PASS"] != first.values["MLFLOW_PASS"]
    assert rotated.values["MLFLOW_USER"] == first.values["MLFLOW_USER"]
    assert "MLFLOW_PASS" in rotated.rotated


def test_username_override_requires_flag(tmp_path: Path) -> None:
    env_file = tmp_path / "mlflow.env"
    mlflow_credentials.ensure_mlflow_credentials(env_file)

    with pytest.raises(RuntimeError):
        mlflow_credentials.ensure_mlflow_credentials(env_file, username="custom-user")

    updated = mlflow_credentials.ensure_mlflow_credentials(
        env_file,
        username="custom-user",
        rotate_username=True,
    )

    assert updated.values["MLFLOW_USER"] == "custom-user"
