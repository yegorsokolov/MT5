import pytest

pytest.skip("Remote management API removed; see archive/bot_apis.", allow_module_level=True)

import sys
import types
import importlib
import hmac
import hashlib
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

import pytest


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def load_api(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "token")
    monkeypatch.setenv("AUDIT_LOG_SECRET", "secret")
    sm_mod = types.ModuleType("utils.secret_manager")

    class SM:
        def get_secret(self, name, *a, **k):
            if name == "AUDIT_LOG_SECRET":
                return "secret"
            return "token"

    sm_mod.SecretManager = SM
    sys.modules["utils.secret_manager"] = sm_mod
    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    sys.modules["log_utils"] = types.SimpleNamespace(
        LOG_FILE=tmp_path / "app.log", setup_logging=lambda: logger
    )
    risk_mod = types.ModuleType("risk_manager")
    risk_mod.risk_manager = types.SimpleNamespace(status=lambda: {})
    risk_mod.ensure_scheduler_started = lambda: None
    sys.modules["risk_manager"] = risk_mod
    sched_mod = types.ModuleType("scheduler")
    sched_mod.start_scheduler = lambda: None
    sched_mod.stop_scheduler = lambda: None
    async def async_stub(*a, **k):
        return None

    sched_mod.schedule_retrain = lambda *a, **k: None
    sched_mod.resource_reprobe = async_stub
    sched_mod.run_drift_detection = lambda: None
    sched_mod.run_feature_importance_drift = lambda: None
    sched_mod.run_change_point_detection = lambda: None
    sched_mod.run_trade_analysis = lambda: None
    sched_mod.run_decision_review = lambda: None
    sched_mod.run_diagnostics = lambda: None
    sched_mod.rebuild_news_vectors = lambda: None
    sched_mod.update_regime_performance = lambda: None
    sched_mod.run_backups = lambda: None
    sched_mod.cleanup_checkpoints = lambda: None
    sys.modules["scheduler"] = sched_mod
    rm_mod = types.ModuleType("utils.resource_monitor")

    class DummyMonitor:
        def __init__(self, *a, **k):
            self.max_rss_mb = k.get("max_rss_mb")
            self.max_cpu_pct = k.get("max_cpu_pct")
            self.alert_callback = None
            self.started = False

        def start(self):
            self.started = True

        def stop(self):
            self.started = False

    rm_mod.ResourceMonitor = DummyMonitor
    sys.modules["utils.resource_monitor"] = rm_mod
    sys.modules["utils"] = types.SimpleNamespace(update_config=lambda *a, **k: None)
    sys.modules["utils.graceful_exit"] = types.SimpleNamespace(
        graceful_exit=lambda *a, **k: None
    )
    if "prometheus_client" in sys.modules:
        del sys.modules["prometheus_client"]
    import prometheus_client
    sys.modules["prometheus_client"] = prometheus_client
    mod = importlib.reload(importlib.import_module("remote_api"))
    mod.init_logging()
    mod.init_remote_api()
    mod.AUDIT_LOG = tmp_path / "audit.csv"
    for h in list(mod.audit_logger.handlers):
        mod.audit_logger.removeHandler(h)
        h.close()
    handler = RotatingFileHandler(mod.AUDIT_LOG, maxBytes=1024, backupCount=1)
    handler.setFormatter(logging.Formatter("%(message)s"))
    mod.audit_logger.addHandler(handler)
    mod._buckets.clear()
    return mod


def test_audit_log_hmac(tmp_path, monkeypatch):
    api = load_api(tmp_path, monkeypatch)
    api._audit_log("key1", "/test", 200)
    line = api.AUDIT_LOG.read_text().strip().splitlines()[-1]
    parts = line.split(",")
    data = ",".join(parts[:-1])
    expected = hmac.new(b"secret", data.encode(), hashlib.sha256).hexdigest()
    assert parts[-1] == expected


def test_rate_limit_metric(tmp_path, monkeypatch):
    api = load_api(tmp_path, monkeypatch)
    key = "client"
    assert api._allow_request(key)
    child = api.API_RATE_LIMIT_REMAINING.labels(key=key)
    assert child._value.get() == pytest.approx(api.RATE_LIMIT - 1, rel=1e-3)
    assert api._allow_request(key)
    assert child._value.get() == pytest.approx(api.RATE_LIMIT - 2, abs=1e-2)
