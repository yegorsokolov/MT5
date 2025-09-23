import asyncio
import importlib
import importlib.machinery as machinery
from pathlib import Path
import sys
import types

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DummyProc:
    def __init__(self) -> None:
        self.terminated = False
        self.pid = 321
        self.returncode = 0

    def poll(self) -> int | None:
        return None if not self.terminated else self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.terminated = True
        return self.returncode


@pytest.fixture
def remote_api(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "token")
    monkeypatch.setenv("AUDIT_LOG_SECRET", "audit")
    monkeypatch.setenv("MAX_RSS_MB", "0")
    monkeypatch.setenv("MAX_CPU_PCT", "0")

    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    log_utils_stub = types.SimpleNamespace(
        LOG_FILE=tmp_path / "app.log",
        setup_logging=lambda: logger,
    )
    original_log_utils = sys.modules.get("mt5.log_utils")
    sys.modules["mt5.log_utils"] = log_utils_stub

    utils_stub = types.ModuleType("utils")
    utils_stub.__path__ = []
    utils_stub.update_config = lambda *a, **k: None
    secret_manager_stub = types.ModuleType("utils.secret_manager")

    class _SecretManager:
        def get_secret(self, name: str, default: str | None = None) -> str:
            return default or "token"

    secret_manager_stub.SecretManager = _SecretManager

    async def _async_noop() -> None:
        return None

    graceful_exit_stub = types.ModuleType("utils.graceful_exit")
    graceful_exit_stub.__spec__ = machinery.ModuleSpec("utils.graceful_exit", loader=None)
    graceful_exit_stub.graceful_exit = lambda: _async_noop()

    alerting_stub = types.ModuleType("utils.alerting")
    alerting_stub.__spec__ = machinery.ModuleSpec("utils.alerting", loader=None)
    alerting_stub.send_alert = lambda *a, **k: None

    resource_monitor_stub = types.ModuleType("utils.resource_monitor")
    resource_monitor_stub.__spec__ = machinery.ModuleSpec("utils.resource_monitor", loader=None)

    class _ResourceMonitor:
        def __init__(self, *a, **k) -> None:
            self.max_rss_mb = k.get("max_rss_mb")
            self.max_cpu_pct = k.get("max_cpu_pct")
            self.alert_callback = None
            self.capabilities = types.SimpleNamespace(cpus=1)

        def start(self) -> None:
            return None

        def stop(self) -> None:
            return None

    resource_monitor_stub.ResourceMonitor = _ResourceMonitor

    prometheus_stub = types.ModuleType("prometheus_client")
    prometheus_stub.__spec__ = machinery.ModuleSpec("prometheus_client", loader=None)
    prometheus_stub.CONTENT_TYPE_LATEST = "text/plain"
    prometheus_stub.generate_latest = lambda: b""

    original_utils = sys.modules.get("utils")
    original_secret_manager = sys.modules.get("utils.secret_manager")
    original_graceful_exit = sys.modules.get("utils.graceful_exit")
    original_alerting = sys.modules.get("utils.alerting")
    original_resource_monitor = sys.modules.get("utils.resource_monitor")
    original_prometheus = sys.modules.get("prometheus_client")
    sys.modules["utils"] = utils_stub
    sys.modules["utils.secret_manager"] = secret_manager_stub
    sys.modules["utils.graceful_exit"] = graceful_exit_stub
    sys.modules["utils.alerting"] = alerting_stub
    sys.modules["utils.resource_monitor"] = resource_monitor_stub
    sys.modules["prometheus_client"] = prometheus_stub

    mt5_pkg = importlib.import_module("mt5")

    risk_manager_stub = types.ModuleType("mt5.risk_manager")
    risk_manager_stub.risk_manager = types.SimpleNamespace(
        status=lambda: {
            "exposure": 0.0,
            "daily_loss": 0.0,
            "var": 0.0,
            "trading_halted": False,
        },
        reset=lambda: None,
        update=lambda *a, **k: None,
    )
    risk_manager_stub.ensure_scheduler_started = lambda: None

    scheduler_stub = types.ModuleType("mt5.scheduler")
    scheduler_stub.start_scheduler = lambda: None
    scheduler_stub.stop_scheduler = lambda: None
    scheduler_stub.schedule_retrain = lambda **kw: None
    scheduler_stub.resource_reprobe = lambda: _async_noop()
    scheduler_stub.run_drift_detection = lambda: None
    scheduler_stub.run_feature_importance_drift = lambda: None
    scheduler_stub.run_change_point_detection = lambda: None
    scheduler_stub.run_trade_analysis = lambda: None
    scheduler_stub.run_decision_review = lambda: None
    scheduler_stub.run_diagnostics = lambda: None
    scheduler_stub.rebuild_news_vectors = lambda: None
    scheduler_stub.update_regime_performance = lambda: None
    scheduler_stub.run_backups = lambda: None
    scheduler_stub.cleanup_checkpoints = lambda: None

    class _Counter:
        def labels(self, **kwargs):
            return self

        def inc(self) -> None:
            return None

    metrics_stub = types.ModuleType("mt5.metrics")
    metrics_stub.BOT_BACKOFFS = _Counter()
    metrics_stub.BOT_RESTARTS = _Counter()
    metrics_stub.BOT_RESTART_FAILURES = _Counter()
    metrics_stub.RESOURCE_RESTARTS = types.SimpleNamespace(inc=lambda: None)

    original_risk = sys.modules.get("mt5.risk_manager")
    original_scheduler = sys.modules.get("mt5.scheduler")
    original_metrics = sys.modules.get("mt5.metrics")

    sys.modules["mt5.risk_manager"] = risk_manager_stub
    sys.modules["mt5.scheduler"] = scheduler_stub
    sys.modules["mt5.metrics"] = metrics_stub
    setattr(mt5_pkg, "risk_manager", risk_manager_stub)

    try:
        ra = importlib.reload(importlib.import_module("mt5.remote_api"))
    finally:
        if original_log_utils is not None:
            sys.modules["mt5.log_utils"] = original_log_utils
        else:
            sys.modules.pop("mt5.log_utils", None)

        if original_utils is not None:
            sys.modules["utils"] = original_utils
        else:
            sys.modules.pop("utils", None)

        if original_secret_manager is not None:
            sys.modules["utils.secret_manager"] = original_secret_manager
        else:
            sys.modules.pop("utils.secret_manager", None)

        if original_graceful_exit is not None:
            sys.modules["utils.graceful_exit"] = original_graceful_exit
        else:
            sys.modules.pop("utils.graceful_exit", None)

        if original_alerting is not None:
            sys.modules["utils.alerting"] = original_alerting
        else:
            sys.modules.pop("utils.alerting", None)

        if original_resource_monitor is not None:
            sys.modules["utils.resource_monitor"] = original_resource_monitor
        else:
            sys.modules.pop("utils.resource_monitor", None)

        if original_prometheus is not None:
            sys.modules["prometheus_client"] = original_prometheus
        else:
            sys.modules.pop("prometheus_client", None)

        if original_risk is not None:
            sys.modules["mt5.risk_manager"] = original_risk
            setattr(mt5_pkg, "risk_manager", original_risk)
        else:
            sys.modules.pop("mt5.risk_manager", None)
            if hasattr(mt5_pkg, "risk_manager"):
                delattr(mt5_pkg, "risk_manager")

        if original_scheduler is not None:
            sys.modules["mt5.scheduler"] = original_scheduler
        else:
            sys.modules.pop("mt5.scheduler", None)

        if original_metrics is not None:
            sys.modules["mt5.metrics"] = original_metrics
        else:
            sys.modules.pop("mt5.metrics", None)
    ra.bots.clear()
    Path(ra.LOG_FILE).parent.mkdir(parents=True, exist_ok=True)
    Path(ra.LOG_FILE).write_text("first\nsecond\n", encoding="utf-8")

    updates: dict[str, tuple[str, str]] = {}
    retrains: list[dict[str, object]] = []
    control_calls: list[str] = []

    monkeypatch.setattr(ra, "update_config", lambda k, v, r: updates.update({k: (v, r)}), raising=False)
    monkeypatch.setattr(ra, "schedule_retrain", lambda **kw: retrains.append(kw), raising=False)
    monkeypatch.setattr(ra, "send_alert", lambda *a, **k: None, raising=False)
    monkeypatch.setattr(ra, "Popen", lambda *a, **k: DummyProc(), raising=False)

    async def dummy_control() -> None:
        control_calls.append("ran")

    monkeypatch.setattr(ra, "CONTROL_TASKS", {"dummy": lambda: dummy_control()}, raising=False)

    ra.init_remote_api(api_key="token", audit_secret="audit")

    yield ra, updates, retrains, control_calls

    asyncio.get_event_loop().run_until_complete(ra.shutdown())
    ra.bots.clear()


def test_start_stop_and_status(remote_api):
    ra, updates, retrains, control_calls = remote_api

    loop = asyncio.get_event_loop()

    result = loop.run_until_complete(ra.start_bot("bot1"))
    assert result["status"] == "started"
    assert "bot1" in ra.bots

    bots = loop.run_until_complete(ra.list_bots())
    assert bots["bot1"]["running"] is True

    status = loop.run_until_complete(ra.bot_status("bot1", lines=1))
    assert status["running"] is True
    assert "second" in status["logs"]

    logs = loop.run_until_complete(ra.bot_logs("bot1", lines=1))
    assert "second" in logs["logs"]

    health = loop.run_until_complete(ra.health(lines=1))
    assert health["running"] is True

    stopped = loop.run_until_complete(ra.stop_bot("bot1"))
    assert stopped["status"] == "stopped"
    assert "bot1" not in ra.bots


def test_update_configuration_and_controls(remote_api):
    ra, updates, retrains, control_calls = remote_api

    loop = asyncio.get_event_loop()

    change = ra.ConfigUpdate(key="threshold", value="0.9", reason="test")
    resp = loop.run_until_complete(ra.update_configuration(change))
    assert resp["status"] == "updated"
    assert updates["threshold"] == ("0.9", "test")

    control = loop.run_until_complete(ra.run_control("dummy"))
    assert control["status"] == "ok"
    assert control_calls == ["ran"]

    scheduled = loop.run_until_complete(
        ra.schedule_manual_retrain("classic", update_hyperparams=True)
    )
    assert scheduled["model"] == "classic"
    assert retrains == [{"model": "classic", "update_hyperparams": True}]

    with pytest.raises(HTTPException):
        loop.run_until_complete(ra.schedule_manual_retrain("unknown"))


def test_metrics_and_risk(remote_api):
    ra, *_ = remote_api
    loop = asyncio.get_event_loop()
    received: list[dict[str, int]] = []
    remove = ra.register_metrics_consumer(lambda payload: received.append(payload))

    loop.run_until_complete(ra.push_metrics({"value": 1}))
    assert received == [{"value": 1}]
    remove()

    risk = loop.run_until_complete(ra.risk_status())
    assert set(risk) >= {"exposure", "daily_loss", "var", "trading_halted"}


def test_get_logs_requires_file(remote_api, monkeypatch):
    ra, *_ = remote_api
    missing = Path(ra.LOG_FILE).with_name("missing.log")
    monkeypatch.setattr(ra, "LOG_FILE", missing, raising=False)
    with pytest.raises(HTTPException):
        asyncio.get_event_loop().run_until_complete(ra.get_logs())
