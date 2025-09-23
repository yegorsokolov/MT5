import asyncio
import importlib
import importlib.machinery as machinery
from dataclasses import asdict
from pathlib import Path
import sys
import types

import pytest

grpc = pytest.importorskip("grpc")

repo_root = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "proto"))


class DummyProc:
    def __init__(self) -> None:
        self.terminated = False
        self.pid = 123
        self.returncode = 0

    def poll(self) -> int | None:
        return None if not self.terminated else self.returncode

    def terminate(self) -> None:
        self.terminated = True

    def wait(self, timeout: float | None = None) -> int:
        self.terminated = True
        return self.returncode


@pytest.fixture
def remote_api_env(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "token")
    monkeypatch.setenv("AUDIT_LOG_SECRET", "audit")

    logger = types.SimpleNamespace(
        info=lambda *a, **k: None,
        warning=lambda *a, **k: None,
        error=lambda *a, **k: None,
        exception=lambda *a, **k: None,
    )
    log_utils_stub = types.SimpleNamespace(
        LOG_FILE=tmp_path / "grpc.log",
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
    from mt5.controller_settings import get_controller_settings, update_controller_settings

    original_settings = get_controller_settings()
    update_controller_settings(max_rss_mb=None, max_cpu_pct=None, watchdog_usec=0)

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

    async def _async_noop() -> None:
        return None

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
    Path(ra.LOG_FILE).write_text("line1\nline2\n", encoding="utf-8")

    updates: dict[str, tuple[str, str]] = {}
    monkeypatch.setattr(ra, "update_config", lambda k, v, r: updates.update({k: (v, r)}), raising=False)
    monkeypatch.setattr(ra, "Popen", lambda *a, **k: DummyProc(), raising=False)
    monkeypatch.setattr(ra, "send_alert", lambda *a, **k: None, raising=False)

    ra.init_remote_api(api_key="token", audit_secret="audit")
    grpc_mod = importlib.reload(importlib.import_module("mt5.grpc_api"))

    try:
        yield ra, grpc_mod, updates
    finally:
        asyncio.get_event_loop().run_until_complete(ra.shutdown())
        ra.bots.clear()
        update_controller_settings(**asdict(original_settings))


async def start_server(grpc_mod):
    server = grpc.aio.server()
    grpc_mod.management_pb2_grpc.add_ManagementServiceServicer_to_server(
        grpc_mod.ManagementServicer(), server
    )
    cert_dir = repo_root / "certs"
    private_key = (cert_dir / "server.key").read_bytes()
    certificate_chain = (cert_dir / "server.crt").read_bytes()
    server_credentials = grpc.ssl_server_credentials(((private_key, certificate_chain),))
    port = server.add_secure_port("localhost:0", server_credentials)
    await server.start()
    return server, port


def test_list_and_update(remote_api_env):
    ra, grpc_mod, updates = remote_api_env

    async def _run() -> None:
        server, port = await start_server(grpc_mod)
        cert_dir = repo_root / "certs"
        creds = grpc.ssl_channel_credentials(root_certificates=(cert_dir / "ca.crt").read_bytes())
        async with grpc.aio.secure_channel(f"localhost:{port}", creds) as channel:
            stub = grpc_mod.management_pb2_grpc.ManagementServiceStub(channel)
            with pytest.raises(grpc.aio.AioRpcError) as exc:
                await stub.ListBots(grpc_mod.empty_pb2.Empty())
            assert exc.value.code() == grpc.StatusCode.UNAUTHENTICATED

            metadata = (("x-api-key", "token"),)
            resp = await stub.ListBots(grpc_mod.empty_pb2.Empty(), metadata=metadata)
            assert resp.bots == {}

            cfg = grpc_mod.management_pb2.ConfigChange(key="threshold", value="0.7", reason="test")
            res = await stub.UpdateConfig(cfg, metadata=metadata)
            assert res.status == "updated"
            assert updates["threshold"] == ("0.7", "test")

        await server.stop(None)

    asyncio.get_event_loop().run_until_complete(_run())


def test_start_stop(remote_api_env):
    ra, grpc_mod, _ = remote_api_env

    async def _run() -> None:
        server, port = await start_server(grpc_mod)
        cert_dir = repo_root / "certs"
        creds = grpc.ssl_channel_credentials(root_certificates=(cert_dir / "ca.crt").read_bytes())
        async with grpc.aio.secure_channel(f"localhost:{port}", creds) as channel:
            stub = grpc_mod.management_pb2_grpc.ManagementServiceStub(channel)
            metadata = (("x-api-key", "token"),)
            resp = await stub.StartBot(
                grpc_mod.management_pb2.StartRequest(bot_id="b1"), metadata=metadata
            )
            assert resp.status == "started"
            assert "b1" in ra.bots
            resp2 = await stub.StopBot(
                grpc_mod.management_pb2.BotIdRequest(bot_id="b1"), metadata=metadata
            )
            assert resp2.status == "stopped"
            assert "b1" not in ra.bots

        await server.stop(None)

    asyncio.get_event_loop().run_until_complete(_run())


def test_status_and_logs(remote_api_env):
    ra, grpc_mod, _ = remote_api_env

    async def _run() -> None:
        server, port = await start_server(grpc_mod)
        cert_dir = repo_root / "certs"
        creds = grpc.ssl_channel_credentials(root_certificates=(cert_dir / "ca.crt").read_bytes())
        async with grpc.aio.secure_channel(f"localhost:{port}", creds) as channel:
            stub = grpc_mod.management_pb2_grpc.ManagementServiceStub(channel)
            metadata = (("x-api-key", "token"),)
            await stub.StartBot(
                grpc_mod.management_pb2.StartRequest(bot_id="b1"), metadata=metadata
            )

            status = await stub.BotStatus(
                grpc_mod.management_pb2.BotStatusRequest(bot_id="b1", lines=5), metadata=metadata
            )
            assert status.bot_id == "b1"
            assert status.running is True
            assert status.pid == 123
            assert status.returncode == 0
            assert "line1" in status.logs

            logs = await stub.GetLogs(
                grpc_mod.management_pb2.LogRequest(lines=1), metadata=metadata
            )
            assert "line2" in logs.logs

            await stub.StopBot(
                grpc_mod.management_pb2.BotIdRequest(bot_id="b1"), metadata=metadata
            )

        await server.stop(None)

    asyncio.get_event_loop().run_until_complete(_run())


def test_get_risk_status(remote_api_env, monkeypatch):
    ra, grpc_mod, _ = remote_api_env
    monkeypatch.setenv("MAX_PORTFOLIO_DRAWDOWN", "100")
    from mt5 import risk_manager as rm_mod

    rm_mod.risk_manager.reset()
    rm_mod.risk_manager.update("b1", -60)
    rm_mod.risk_manager.update("b2", -50)

    async def _run() -> None:
        server, port = await start_server(grpc_mod)
        cert_dir = repo_root / "certs"
        creds = grpc.ssl_channel_credentials(root_certificates=(cert_dir / "ca.crt").read_bytes())
        async with grpc.aio.secure_channel(f"localhost:{port}", creds) as channel:
            stub = grpc_mod.management_pb2_grpc.ManagementServiceStub(channel)
            metadata = (("x-api-key", "token"),)
            status = await stub.GetRiskStatus(grpc_mod.empty_pb2.Empty(), metadata=metadata)
            assert status.trading_halted is True
            assert status.daily_loss == -110

        await server.stop(None)

    asyncio.get_event_loop().run_until_complete(_run())
