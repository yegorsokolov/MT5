from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from utils import environment


@pytest.fixture(autouse=True)
def _clear_backend_env(monkeypatch):
    monkeypatch.delenv("MT5_BRIDGE_BACKEND", raising=False)
    for name in ("pymt5linux", "utils.bridge_clients.mt5linux_client"):
        sys.modules.pop(name, None)
    yield
    monkeypatch.delenv("MT5_BRIDGE_BACKEND", raising=False)
    for name in ("pymt5linux", "utils.bridge_clients.mt5linux_client"):
        sys.modules.pop(name, None)


def test_check_wine_bridge_skipped_for_custom_backend(monkeypatch):
    monkeypatch.setenv("MT5_BRIDGE_BACKEND", "custom.backend:load")
    result = environment._check_wine_bridge()
    assert result["status"] == "skipped"
    assert "custom.backend" in result["detail"]


def test_check_wine_bridge_runs_for_wine_backend(monkeypatch):
    monkeypatch.setenv("MT5_BRIDGE_BACKEND", "wine")
    monkeypatch.setattr(environment, "sys", SimpleNamespace(platform="linux"))

    settings = {
        "host": "127.0.0.1",
        "port": 18812,
        "host_source": "env:MT5LINUX_HOST",
        "port_source": "env:MT5LINUX_PORT",
    }
    monkeypatch.setattr(environment.mt5_bridge, "get_mt5linux_connection_settings", lambda: settings)

    stub_mt5linux = ModuleType("mt5linux")
    monkeypatch.setitem(sys.modules, "mt5linux", stub_mt5linux)

    connect_calls: list[tuple] = []

    class _Connection:
        def eval(self, command: str) -> str:
            connect_calls.append(("eval", command))
            assert "MetaTrader5" in command
            return "5.0"

        def close(self) -> None:
            connect_calls.append(("close", None))

    class _RPyC(ModuleType):
        def __init__(self) -> None:
            super().__init__("rpyc")
            self.classic = SimpleNamespace(connect=lambda host, port, config: _Connection())

    monkeypatch.setitem(sys.modules, "rpyc", _RPyC())

    result = environment._check_wine_bridge()
    assert result["status"] == "passed"
    assert "127.0.0.1:18812" in result["detail"]
    assert any(call[0] == "eval" for call in connect_calls)
