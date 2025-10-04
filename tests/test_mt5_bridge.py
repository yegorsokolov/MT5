from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType

import pytest

from utils import mt5_bridge


@pytest.fixture(autouse=True)
def _reset_dotenv_cache():
    mt5_bridge._load_dotenv.cache_clear()  # type: ignore[attr-defined]
    yield
    mt5_bridge._load_dotenv.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def _reset_bridge(monkeypatch):
    monkeypatch.setattr(mt5_bridge, "_MT5_MODULE", None)
    monkeypatch.setattr(mt5_bridge, "_BRIDGE_INFO", {})
    for key in ("MT5_BRIDGE_BACKEND", "MT5_BRIDGE_MODULE"):
        monkeypatch.delenv(key, raising=False)
    yield
    monkeypatch.setattr(mt5_bridge, "_MT5_MODULE", None)
    monkeypatch.setattr(mt5_bridge, "_BRIDGE_INFO", {})
    for name in (
        "MetaTrader5",
        "pymt5linux",
        "custom_backend",
        "utils.bridge_clients.mt5linux_client",
    ):
        sys.modules.pop(name, None)


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


def test_load_mt5_module_uses_wine_backend(monkeypatch, _reset_bridge):
    monkeypatch.setenv("MT5_BRIDGE_BACKEND", "wine")
    sys.modules.pop("MetaTrader5", None)

    bridge_module = ModuleType("utils.bridge_clients.mt5linux_client")
    mt5_module = ModuleType("MetaTrader5")
    calls: list[str] = []

    def initializer() -> None:
        calls.append("initialize")

    def bridge_info() -> dict[str, str]:
        return {"host": "127.0.0.1", "port": 2000}

    bridge_module.initialize = initializer  # type: ignore[attr-defined]
    bridge_module.MetaTrader5 = mt5_module  # type: ignore[attr-defined]
    bridge_module.bridge_info = bridge_info  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "utils.bridge_clients.mt5linux_client", bridge_module)

    module = mt5_bridge.load_mt5_module(force=True)

    assert module is mt5_module
    assert calls == ["initialize"]
    backend_info = mt5_bridge.describe_backend()
    assert backend_info["backend"] == "wine"
    assert backend_info["bridge_package"] == "utils.bridge_clients.mt5linux_client"
    assert backend_info["requested_backend"] == "wine"
    assert backend_info["mt5linux"]["host"] == "127.0.0.1"


def test_load_mt5_module_supports_custom_backend(monkeypatch, _reset_bridge):
    sys.modules.pop("MetaTrader5", None)

    backend_module = ModuleType("custom_backend")
    mt5_module = ModuleType("MetaTrader5")
    calls: list[str] = []

    def initializer() -> None:
        calls.append("initialize")

    def loader() -> ModuleType:
        calls.append("load")
        return mt5_module

    backend_module.initialize = initializer  # type: ignore[attr-defined]
    backend_module.load_mt5 = loader  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "custom_backend", backend_module)
    monkeypatch.setenv("MT5_BRIDGE_BACKEND", "custom_backend")

    module = mt5_bridge.load_mt5_module(force=True)

    assert module is mt5_module
    assert calls == ["initialize", "load"]
    backend_info = mt5_bridge.describe_backend()
    assert backend_info["backend"] == "custom_backend"
    assert backend_info["bridge_package"] == "custom_backend"
    assert backend_info["requested_backend"] == "custom_backend"


def test_get_mt5linux_connection_settings(monkeypatch):
    monkeypatch.setenv("MT5LINUX_HOST", "10.0.0.1")
    monkeypatch.setenv("MT5LINUX_PORT", "19999")
    monkeypatch.setenv("MT5LINUX_TIMEOUT", "12.5")

    settings = mt5_bridge.get_mt5linux_connection_settings()
    assert settings["host"] == "10.0.0.1"
    assert settings["port"] == 19999
    assert settings["timeout"] == pytest.approx(12.5)
    assert settings["host_source"].startswith("env:")
    assert settings["port_source"].startswith("env:")
