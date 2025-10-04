from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from utils import environment


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    monkeypatch.delenv("MT5_BRIDGE_BACKEND", raising=False)
    sys.modules.pop("pymt5linux", None)
    yield
    monkeypatch.delenv("MT5_BRIDGE_BACKEND", raising=False)
    sys.modules.pop("pymt5linux", None)


def test_check_wine_bridge_skipped_for_custom_backend(monkeypatch):
    monkeypatch.setenv("MT5_BRIDGE_BACKEND", "custom.backend:load")
    result = environment._check_wine_bridge()
    assert result["status"] == "skipped"
    assert "custom.backend" in result["detail"]


def test_check_wine_bridge_runs_for_wine_backend(monkeypatch):
    monkeypatch.setenv("MT5_BRIDGE_BACKEND", "wine")
    monkeypatch.setattr(environment, "sys", SimpleNamespace(platform="linux"))
    monkeypatch.setattr(environment.shutil, "which", lambda executable: "/usr/bin/wine")

    stub_bridge = ModuleType("pymt5linux")
    monkeypatch.setitem(sys.modules, "pymt5linux", stub_bridge)

    monkeypatch.setattr(
        environment,
        "_discover_wine_python_path",
        lambda: ("C:/Python/python.exe", "env"),
    )
    monkeypatch.setattr(
        environment,
        "_discover_wine_prefix",
        lambda: ("/home/test/.wine", "env"),
    )

    class _Completed:
        returncode = 0
        stdout = "MetaTrader5 5.0"
        stderr = ""

    monkeypatch.setattr(environment.subprocess, "run", lambda *args, **kwargs: _Completed())

    result = environment._check_wine_bridge()
    assert result["status"] == "passed"
    assert "MetaTrader5" in result["detail"]
