import sys
from pathlib import Path

import types
import importlib
import pytest
from fastapi.testclient import TestClient
import asyncio
import contextlib
from typing import List

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

    def _log(msg, *a):
        logs.append(msg % a if a else msg)

    logger = types.SimpleNamespace(
        warning=_log,
        info=_log,
        error=_log,
        exception=_log,
    )
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
    risk_mod.ensure_scheduler_started = lambda: None
    sys.modules['risk_manager'] = risk_mod
    sched_mod = types.ModuleType('scheduler')
    sched_mod.start_scheduler = lambda: None
    sched_mod.stop_scheduler = lambda: None
    sys.modules['scheduler'] = sched_mod
    rm_mod = types.ModuleType('utils.resource_monitor')

    class DummyMonitor:
        def __init__(self, *a, **k):
            self.max_rss_mb = k.get('max_rss_mb')
            self.max_cpu_pct = k.get('max_cpu_pct')
            self.alert_callback = None
            self.start_calls = 0
            self.stop_calls = 0
            self.started = False
            self.capabilities = types.SimpleNamespace(cpus=1)

        def start(self):
            self.start_calls += 1
            self.started = True

        def stop(self):
            self.stop_calls += 1
            self.started = False

    rm_mod.ResourceMonitor = DummyMonitor
    sys.modules['utils.resource_monitor'] = rm_mod
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
    mod.init_logging()
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

def setup_api(tmp_path, monkeypatch):
    api = load_api(tmp_path / "app.log", monkeypatch)
    api.bots.clear()
    Path(api.LOG_FILE).write_text("line1\nline2\n")
    return api

def test_health_auth(tmp_path, monkeypatch):
    api = setup_api(tmp_path, monkeypatch)
    with TestClient(api.app) as client:
        resp = client.get("/health")
        assert resp.status_code == 401
        resp = client.get("/health", headers={"x-api-key": "token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is True
        assert "line2" in data["logs"]
        assert isinstance(data["bots"], dict)

def test_bot_status(tmp_path, monkeypatch):
    api = setup_api(tmp_path, monkeypatch)
    api.bots["bot1"] = api.BotInfo(proc=DummyProc())
    with TestClient(api.app) as client:
        resp = client.get("/bots/bot1/status", headers={"x-api-key": "token"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["running"] is True
        assert "pid" in data and "returncode" in data
        resp = client.get("/bots/none/status", headers={"x-api-key": "token"})
        assert resp.status_code == 404


def test_bot_logs(tmp_path, monkeypatch):
    api = setup_api(tmp_path, monkeypatch)
    api.bots["bot1"] = api.BotInfo(proc=DummyProc())
    with TestClient(api.app) as client:
        resp = client.get("/bots/bot1/logs", headers={"x-api-key": "token"})
        assert resp.status_code == 200
        assert "line2" in resp.json()["logs"]
        resp = client.get("/bots/none/logs", headers={"x-api-key": "token"})
        assert resp.status_code == 404


def test_metrics_websocket(tmp_path, monkeypatch):
    api = setup_api(tmp_path, monkeypatch)

    with TestClient(api.app) as client:
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
    api = setup_api(tmp_path, monkeypatch)
    with TestClient(api.app) as client:
        resp = client.get("/metrics")
        assert resp.status_code == 200


def test_shutdown_cancels_background_tasks(tmp_path, monkeypatch):
    api = setup_api(tmp_path, monkeypatch)
    assert not api._background_tasks

    api.resource_watchdog.max_rss_mb = 100
    loop = asyncio.get_event_loop()
    loop.run_until_complete(api._start_watcher())
    loop.run_until_complete(api._start_watcher())
    assert api.resource_watchdog.start_calls == 1
    tasks = list(api._background_tasks)

    loop.run_until_complete(api._stop_scheduler_event())

    assert api.resource_watchdog.stop_calls == 1
    assert api.resource_watchdog.started is False
    assert all(task.cancelled() or task.done() for task in tasks)
    assert not api._background_tasks


def test_bot_restart_and_health(tmp_path, monkeypatch):
    api = setup_api(tmp_path, monkeypatch)

    class CrashProc(DummyProc):
        def __init__(self, code=1):
            super().__init__()
            self.terminated = True
            self.returncode = code

    restarted = DummyProc()

    def fake_popen(cmd):
        return restarted

    api.Popen = fake_popen
    monotonic = [0.0]

    def fake_monotonic():
        return monotonic[0]

    monkeypatch.setattr(api.time, "monotonic", fake_monotonic)

    api.bots["bot1"] = api.BotInfo(proc=CrashProc())

    loop = asyncio.get_event_loop()
    loop.run_until_complete(api._check_bots_once())

    # No restart yet due to backoff
    assert api.bots["bot1"].restart_count == 0

    monotonic[0] += api.BOT_BACKOFF_BASE_SECONDS + 0.1
    loop.run_until_complete(api._check_bots_once())

    with TestClient(api.app) as client:
        resp = client.get("/bots", headers={"x-api-key": "token"})
        data = resp.json()["bot1"]
        assert data["restart_count"] == 1
        assert data["exit_code"] == 1
        assert data["running"] is True

    assert any("Bot bot1 exited with code 1" in msg for msg in api._logs)


def test_bot_removed_on_restart_failure(tmp_path, monkeypatch):
    api = setup_api(tmp_path, monkeypatch)

    class CrashProc(DummyProc):
        def __init__(self):
            super().__init__()
            self.terminated = True
            self.returncode = 2

    def fail_popen(cmd):
        raise RuntimeError("fail")

    api.Popen = fail_popen
    monotonic = [0.0]

    def fake_monotonic():
        return monotonic[0]

    monkeypatch.setattr(api.time, "monotonic", fake_monotonic)

    api.bots["bot1"] = api.BotInfo(proc=CrashProc())

    loop = asyncio.get_event_loop()
    loop.run_until_complete(api._check_bots_once())
    assert "bot1" in api.bots

    monotonic[0] += api.BOT_BACKOFF_BASE_SECONDS + 0.1
    loop.run_until_complete(api._check_bots_once())

    assert "bot1" not in api.bots
    assert any("bot1" in msg for msg in api._logs)


def test_bot_crash_limit_triggers_alert(tmp_path, monkeypatch):
    api = setup_api(tmp_path, monkeypatch)
    alerts: List[str] = []

    def capture_alert(message: str) -> None:
        alerts.append(message)

    monkeypatch.setattr(api, "send_alert", capture_alert)
    monkeypatch.setattr(api, "BOT_MAX_CRASHES", 2)
    monkeypatch.setattr(api, "BOT_CRASH_WINDOW", 100.0)
    monkeypatch.setattr(api, "BOT_BACKOFF_BASE_SECONDS", 1.0)
    monkeypatch.setattr(api, "BOT_BACKOFF_MAX_SECONDS", 5.0)
    monkeypatch.setattr(api, "RESTART_HISTORY_LIMIT", 5)

    class CrashProc(DummyProc):
        def __init__(self):
            super().__init__()
            self.terminated = True
            self.returncode = 3

    api.Popen = lambda cmd: CrashProc()
    monotonic = [0.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: monotonic[0])

    api.bots["bot-loop"] = api.BotInfo(proc=CrashProc())

    loop = asyncio.get_event_loop()
    loop.run_until_complete(api._check_bots_once())
    for _ in range(4):
        monotonic[0] += 10
        loop.run_until_complete(api._check_bots_once())

    assert "bot-loop" not in api.bots
    assert alerts
    assert any("bot-loop" in msg for msg in alerts)
