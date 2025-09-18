import sys
from pathlib import Path

import types
import importlib
import pytest
from fastapi.testclient import TestClient
import asyncio
import contextlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def load_api(tmp_log, monkeypatch):
    sm_mod = types.ModuleType('utils.secret_manager')
    class SM:
        def get_secret(self, *a, **k):
            return 'token'
    sm_mod.SecretManager = SM
    sys.modules['utils.secret_manager'] = sm_mod
    monkeypatch.setenv("API_KEY", "token")
    monkeypatch.setenv("AUDIT_LOG_SECRET", "audit")
    logs = []
    logger = types.SimpleNamespace(warning=lambda msg, *a: logs.append(msg % a if a else msg))
    sys.modules['log_utils'] = types.SimpleNamespace(
        LOG_FILE=tmp_log,
        setup_logging=lambda: logger,
        log_exceptions=lambda f: f,
        TRADE_COUNT=types.SimpleNamespace(inc=lambda: None),
        ERROR_COUNT=types.SimpleNamespace(inc=lambda: None),
        log_decision=lambda *a, **k: None,
    )
    risk_mod = types.ModuleType('risk_manager')
    risk_mod.risk_manager = types.SimpleNamespace(status=lambda: {})
    sys.modules['risk_manager'] = risk_mod
    sched_mod = types.ModuleType('scheduler')
    sched_mod.start_scheduler = lambda: None
    sched_mod.stop_scheduler = lambda: None
    sys.modules['scheduler'] = sched_mod
    sys.modules['prometheus_client'] = types.SimpleNamespace(
        Counter=lambda *a, **k: None,
        Gauge=lambda *a, **k: None,
        generate_latest=lambda: b"",
        CONTENT_TYPE_LATEST="text/plain",
        REGISTRY=types.SimpleNamespace(_names_to_collectors={}),
    )
    sys.modules['yaml'] = types.SimpleNamespace(
        safe_load=lambda *a, **k: {},
        safe_dump=lambda *a, **k: "",
    )
    utils_mod = types.ModuleType('utils')
    utils_mod.update_config = lambda *a, **k: None
    sys.modules['utils'] = utils_mod
    sys.modules['utils.graceful_exit'] = types.SimpleNamespace(graceful_exit=lambda *a, **k: None)
    sys.modules['metrics'] = importlib.import_module('metrics')
    mlflow_mod = types.ModuleType('mlflow')
    mlflow_mod.set_tracking_uri = lambda *a, **k: None
    mlflow_mod.set_experiment = lambda *a, **k: None
    mlflow_mod.start_run = contextlib.nullcontext
    mlflow_mod.log_dict = lambda *a, **k: None
    mlflow_mod.__spec__ = importlib.machinery.ModuleSpec('mlflow', loader=None)
    sys.modules['mlflow'] = mlflow_mod
    mod = importlib.reload(importlib.import_module('remote_api'))
    mod.init_remote_api()
    mod._logs = logs
    return mod

class DummyProc:
    def __init__(self):
        self.terminated = False
        self.pid = 123
        self.returncode = None

    def poll(self):
        return None if not self.terminated else self.returncode or 0

    def terminate(self):
        self.terminated = True

def setup_client(tmp_path, monkeypatch):
    api = load_api(tmp_path / "app.log", monkeypatch)
    api.bots.clear()
    Path(api.LOG_FILE).write_text("line1\nline2\n")
    return api, TestClient(api.app)

def test_health_auth(tmp_path, monkeypatch):
    api, client = setup_client(tmp_path, monkeypatch)
    resp = client.get("/health")
    assert resp.status_code == 401
    resp = client.get("/health", headers={"x-api-key": "token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is True
    assert "line2" in data["logs"]
    assert isinstance(data["bots"], dict)

def test_bot_status(tmp_path, monkeypatch):
    api, client = setup_client(tmp_path, monkeypatch)
    api.bots["bot1"] = api.BotInfo(proc=DummyProc())
    resp = client.get("/bots/bot1/status", headers={"x-api-key": "token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["running"] is True
    assert "pid" in data and "returncode" in data
    resp = client.get("/bots/none/status", headers={"x-api-key": "token"})
    assert resp.status_code == 404


def test_bot_logs(tmp_path, monkeypatch):
    api, client = setup_client(tmp_path, monkeypatch)
    api.bots["bot1"] = api.BotInfo(proc=DummyProc())
    resp = client.get("/bots/bot1/logs", headers={"x-api-key": "token"})
    assert resp.status_code == 200
    assert "line2" in resp.json()["logs"]
    resp = client.get("/bots/none/logs", headers={"x-api-key": "token"})
    assert resp.status_code == 404


def test_metrics_websocket(tmp_path, monkeypatch):
    api, client = setup_client(tmp_path, monkeypatch)

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


def test_metrics_endpoint(tmp_path, monkeypatch):
    api, client = setup_client(tmp_path, monkeypatch)
    resp = client.get("/metrics")
    assert resp.status_code == 200


def test_bot_restart_and_health(tmp_path, monkeypatch):
    api, client = setup_client(tmp_path, monkeypatch)

    class CrashProc(DummyProc):
        def __init__(self, code=1):
            super().__init__()
            self.terminated = True
            self.returncode = code

    restarted = DummyProc()

    def fake_popen(cmd):
        return restarted

    api.Popen = fake_popen
    api.bots["bot1"] = api.BotInfo(proc=CrashProc())

    asyncio.get_event_loop().run_until_complete(api._check_bots_once())

    resp = client.get("/bots", headers={"x-api-key": "token"})
    data = resp.json()["bot1"]
    assert data["restart_count"] == 1
    assert data["exit_code"] == 1
    assert data["running"] is True
    assert "Bot bot1 exited with code 1" in api._logs[0]


def test_bot_removed_on_restart_failure(tmp_path, monkeypatch):
    api, client = setup_client(tmp_path, monkeypatch)

    class CrashProc(DummyProc):
        def __init__(self):
            super().__init__()
            self.terminated = True
            self.returncode = 2

    def fail_popen(cmd):
        raise RuntimeError("fail")

    api.Popen = fail_popen
    api.bots["bot1"] = api.BotInfo(proc=CrashProc())

    asyncio.get_event_loop().run_until_complete(api._check_bots_once())

    assert "bot1" not in api.bots
    assert any("bot1" in msg for msg in api._logs)
