import sys
from pathlib import Path
import types
import importlib
import contextlib
import time

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_api(tmp_path, monkeypatch, rate=2):
    monkeypatch.setenv("RATE_LIMIT", str(rate))
    sm_mod = types.ModuleType("utils.secret_manager")

    class SM:
        def get_secret(self, *a, **k):
            return "token"

    sm_mod.SecretManager = SM
    sys.modules["utils.secret_manager"] = sm_mod
    logger = types.SimpleNamespace(warning=lambda *a, **k: None)
    sys.modules["log_utils"] = types.SimpleNamespace(
        LOG_FILE=tmp_path / "app.log",
        setup_logging=lambda: logger,
        log_exceptions=lambda f: f,
        TRADE_COUNT=types.SimpleNamespace(inc=lambda: None),
        ERROR_COUNT=types.SimpleNamespace(inc=lambda: None),
        log_decision=lambda *a, **k: None,
    )
    risk_mod = types.ModuleType("risk_manager")
    risk_mod.risk_manager = types.SimpleNamespace(status=lambda: {})
    sys.modules["risk_manager"] = risk_mod
    sched_mod = types.ModuleType("scheduler")
    sched_mod.start_scheduler = lambda: None
    sched_mod.stop_scheduler = lambda: None
    sys.modules["scheduler"] = sched_mod
    sys.modules["prometheus_client"] = types.SimpleNamespace(
        Counter=lambda *a, **k: None,
        Gauge=lambda *a, **k: None,
        generate_latest=lambda: b"",
        CONTENT_TYPE_LATEST="text/plain",
    )
    sys.modules["yaml"] = types.SimpleNamespace(
        safe_load=lambda *a, **k: {},
        safe_dump=lambda *a, **k: "",
    )
    env_mod = types.ModuleType("environment")
    env_mod.ensure_environment = lambda: None
    sys.modules["utils.environment"] = env_mod
    utils_mod = types.ModuleType("utils")
    utils_mod.update_config = lambda *a, **k: None
    sys.modules["utils"] = utils_mod
    sys.modules["utils.graceful_exit"] = types.SimpleNamespace(graceful_exit=lambda *a, **k: None)
    sys.modules["metrics"] = importlib.import_module("metrics")
    mlflow_mod = types.ModuleType("mlflow")
    mlflow_mod.set_tracking_uri = lambda *a, **k: None
    mlflow_mod.set_experiment = lambda *a, **k: None
    mlflow_mod.start_run = contextlib.nullcontext
    mlflow_mod.log_dict = lambda *a, **k: None
    mlflow_mod.__spec__ = importlib.machinery.ModuleSpec("mlflow", loader=None)
    sys.modules["mlflow"] = mlflow_mod
    mod = importlib.reload(importlib.import_module("remote_api"))
    mod.AUDIT_LOG = tmp_path / "audit.csv"
    return mod


def setup_client(tmp_path, monkeypatch):
    api = load_api(tmp_path, monkeypatch)
    return api, TestClient(api.app)


def test_rate_limiting_and_logging(tmp_path, monkeypatch):
    api, client = setup_client(tmp_path, monkeypatch)

    # Unauthorized request
    assert client.get("/health").status_code == 401

    # Within rate limit
    assert client.get("/health", headers={"x-api-key": "token"}).status_code == 200
    assert client.get("/health", headers={"x-api-key": "token"}).status_code == 200

    # Exceed rate limit
    resp = client.get("/health", headers={"x-api-key": "token"})
    assert resp.status_code == 429

    lines = (tmp_path / "audit.csv").read_text().strip().splitlines()
    assert len(lines) == 4
    statuses = [line.split(",")[-1] for line in lines]
    assert statuses == ["401", "200", "200", "429"]


def test_bucket_expiry(tmp_path, monkeypatch):
    monkeypatch.setenv("BUCKET_TTL", "1")
    api = load_api(tmp_path, monkeypatch)
    current = [time.time()]

    def fake_time():
        return current[0]

    monkeypatch.setattr(api.time, "time", fake_time)

    assert api._allow_request("a")
    assert "a" in api._buckets

    current[0] += 2
    assert api._allow_request("b")
    assert "a" not in api._buckets
    assert "b" in api._buckets
