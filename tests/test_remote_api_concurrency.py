import sys
from pathlib import Path
import os
import types
import importlib
import contextlib
import asyncio

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def load_api(tmp_log):
    os.environ['API_KEY'] = 'token'
    sys.modules['log_utils'] = types.SimpleNamespace(
        LOG_FILE=tmp_log,
        setup_logging=lambda: None,
        log_exceptions=lambda f: f,
        TRADE_COUNT=types.SimpleNamespace(inc=lambda: None),
        ERROR_COUNT=types.SimpleNamespace(inc=lambda: None),
    )
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
    sys.modules['metrics'] = importlib.import_module('metrics')
    sys.modules['mlflow'] = types.SimpleNamespace(
        set_tracking_uri=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        start_run=contextlib.nullcontext,
        log_dict=lambda *a, **k: None,
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

def test_concurrent_start(tmp_path):
    api = load_api(tmp_path / "app.log")
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

def test_concurrent_stop(tmp_path):
    api = load_api(tmp_path / "app.log")
    api.bots.clear()
    api.bots["bot1"] = DummyProc()

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

def test_status_during_stop(tmp_path):
    api = load_api(tmp_path / "app.log")
    api.bots.clear()
    api.bots["bot1"] = DummyProc()

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
