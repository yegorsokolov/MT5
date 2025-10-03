from __future__ import annotations

import os
from pathlib import Path

import pytest

from utils import mt5_bridge


@pytest.fixture(autouse=True)
def _reset_dotenv_cache():
    mt5_bridge._load_dotenv.cache_clear()  # type: ignore[attr-defined]
    yield
    mt5_bridge._load_dotenv.cache_clear()  # type: ignore[attr-defined]


def test_seed_bridge_environment_reads_dotenv(monkeypatch, tmp_path: Path) -> None:
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "PYMT5LINUX_PYTHON=\"C:/Program Files/Python/python.exe\"\n"
        "PYMT5LINUX_WINEPREFIX='/home/test/.wine'\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(mt5_bridge, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(mt5_bridge, "_DOTENV_PATH", dotenv)
    monkeypatch.setattr(mt5_bridge, "_LOGIN_INSTRUCTIONS", tmp_path / "LOGIN.txt")

    for key in (
        "PYMT5LINUX_PYTHON",
        "PYMT5LINUX_WINDOWS_PYTHON",
        "WINE_PYTHON",
        "PYMT5LINUX_WINEPREFIX",
        "WIN_PY_WINE_PREFIX",
        "WINEPREFIX",
    ):
        monkeypatch.delenv(key, raising=False)

    updates = mt5_bridge._seed_bridge_environment()

    expected_python = "C:/Program Files/Python/python.exe"
    expected_prefix = "/home/test/.wine"

    assert updates["PYMT5LINUX_PYTHON"] == expected_python
    assert updates["PYMT5LINUX_WINDOWS_PYTHON"] == expected_python
    assert updates["WINE_PYTHON"] == expected_python
    assert updates["PYMT5LINUX_WINEPREFIX"] == expected_prefix
    assert updates["WIN_PY_WINE_PREFIX"] == expected_prefix
    assert updates["WINEPREFIX"] == expected_prefix

    for key, expected in (
        ("PYMT5LINUX_PYTHON", expected_python),
        ("PYMT5LINUX_WINDOWS_PYTHON", expected_python),
        ("WINE_PYTHON", expected_python),
        ("PYMT5LINUX_WINEPREFIX", expected_prefix),
        ("WIN_PY_WINE_PREFIX", expected_prefix),
        ("WINEPREFIX", expected_prefix),
    ):
        assert os.environ[key] == expected

    for key in (
        "PYMT5LINUX_PYTHON",
        "PYMT5LINUX_WINDOWS_PYTHON",
        "WINE_PYTHON",
        "PYMT5LINUX_WINEPREFIX",
        "WIN_PY_WINE_PREFIX",
        "WINEPREFIX",
    ):
        monkeypatch.delenv(key, raising=False)
