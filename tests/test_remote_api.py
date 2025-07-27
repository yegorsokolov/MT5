import sys
from pathlib import Path

import types
import importlib
import pytest
from fastapi.testclient import TestClient
import asyncio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def load_api(tmp_log):
    sys.modules['log_utils'] = types.SimpleNamespace(
        LOG_FILE=tmp_log,
        setup_logging=lambda: None,
        log_exceptions=lambda f: f,
    )
    return importlib.reload(importlib.import_module('remote_api'))

class DummyProc:
    def __init__(self):
        self.terminated = False
        self.pid = 123
        self.returncode = None
    def poll(self):
        return None if not self.terminated else 0
    def terminate(self):
        self.terminated = True

def setup_client(tmp_path):
    api = load_api(tmp_path / "app.log")
    api.API_KEY = "token"
    api.bots.clear()
    Path(api.LOG_FILE).write_text("line1\nline2\n")
    return api, TestClient(api.app)

def test_health_auth(tmp_path):
    api, client = setup_client(tmp_path)
    resp = client.get("/health")
    assert resp.status_code == 401
    resp = client.get("/health", headers={"x-api-key": "token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is True
    assert "line2" in data["logs"]
    assert isinstance(data["bots"], dict)

def test_bot_status(tmp_path):
    api, client = setup_client(tmp_path)
    api.bots["bot1"] = DummyProc()
    resp = client.get("/bots/bot1/status", headers={"x-api-key": "token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is True
    assert "pid" in data and "returncode" in data
    resp = client.get("/bots/none/status", headers={"x-api-key": "token"})
    assert resp.status_code == 404


def test_metrics_websocket(tmp_path):
    api, client = setup_client(tmp_path)

    with pytest.raises(Exception):
        with client.websocket_connect("/ws/metrics") as ws:
            ws.receive_text()

    with client.websocket_connect("/ws/metrics?api_key=token") as ws:
        asyncio.get_event_loop().run_until_complete(
            api.broadcast_update({"equity": [1, 2, 3], "metrics": {"sharpe": 1.0}})
        )
        data = ws.receive_json()
        assert data["equity"] == [1, 2, 3]
        assert data["metrics"]["sharpe"] == 1.0
