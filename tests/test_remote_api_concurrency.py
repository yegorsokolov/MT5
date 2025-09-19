import sys
from pathlib import Path
import types
import importlib
import contextlib
import asyncio

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def load_api(tmp_log, monkeypatch):
    sm_mod = types.ModuleType('utils.secret_manager')
    class SM:
        def get_secret(self, *a, **k):
            return 'token'
    sm_mod.SecretManager = SM
    sys.modules['utils.secret_manager'] = sm_mod
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
    risk_mod.ensure_scheduler_started = lambda: None
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
    )
    sys.modules['yaml'] = types.SimpleNamespace(
        safe_load=lambda *a, **k: {},
        safe_dump=lambda *a, **k: "",
    )
    env_mod = types.ModuleType("environment")
    env_mod.ensure_environment = lambda: None
    sys.modules['utils.environment'] = env_mod
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

def test_concurrent_start(tmp_path, monkeypatch):
    api = load_api(tmp_path / "app.log", monkeypatch)
    api.bots.clear()

    count = 0
    def fake_popen(cmd):
        nonlocal count
        count += 1
        return DummyProc()
    api.Popen = fake_popen

    async def start_call():
        try:
            await api.start_bot("bot1", None)
            return True
        except Exception:
            return False

    async def run_all():
        return await asyncio.gather(*[start_call() for _ in range(5)])

    results = asyncio.run(run_all())
    assert sum(results) == 1
    assert count == 1
    assert list(api.bots.keys()) == ["bot1"]

def test_concurrent_stop(tmp_path, monkeypatch):
    api = load_api(tmp_path / "app.log", monkeypatch)
    api.bots.clear()
    api.bots["bot1"] = api.BotInfo(proc=DummyProc())

    async def stop_call():
        try:
            return await api.stop_bot("bot1", None)
        except HTTPException as e:
            return e.status_code

    async def run_all():
        return await asyncio.gather(stop_call(), stop_call())

    res = asyncio.run(run_all())
    assert any(isinstance(r, dict) and r["status"] == "stopped" for r in res)
    assert any(r == 404 for r in res)
    assert "bot1" not in api.bots

def test_status_during_stop(tmp_path, monkeypatch):
    api = load_api(tmp_path / "app.log", monkeypatch)
    api.bots.clear()
    api.bots["bot1"] = api.BotInfo(proc=DummyProc())

    async def status_call():
        try:
            return await api.bot_status("bot1", 20, None)
        except HTTPException as e:
            return e.status_code

    async def stop_call():
        try:
            return await api.stop_bot("bot1", None)
        except HTTPException as e:
            return e.status_code

    async def run_all():
        return await asyncio.gather(status_call(), stop_call())

    status_res, stop_res = asyncio.run(run_all())
    assert stop_res == {"bot": "bot1", "status": "stopped"}
    if isinstance(status_res, dict):
        assert status_res["bot"] == "bot1"
    else:
        assert status_res == 404
    assert "bot1" not in api.bots
