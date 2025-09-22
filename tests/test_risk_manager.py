import sys
import types
import importlib
import asyncio
from pathlib import Path
try:
    from fastapi.testclient import TestClient
except Exception:  # pragma: no cover - fallback when FastAPI lacks testclient
    from starlette.testclient import TestClient  # type: ignore
import contextlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mt5.risk_manager import RiskManager


def test_combined_drawdown_triggers_global_stop():
    rm = RiskManager(max_drawdown=100)

    async def bot(loss):
        rm.update("bot", loss)

    asyncio.get_event_loop().run_until_complete(
        asyncio.gather(bot(-60), bot(-50))
    )
    assert rm.status()["trading_halted"] is True


def load_api(tmp_log, monkeypatch):
    monkeypatch.patch('utils.secret_manager.SecretManager.get_secret', return_value='token')
    logger = types.SimpleNamespace(warning=lambda *a, **k: None, info=lambda *a, **k: None)
    sys.modules["log_utils"] = types.SimpleNamespace(
        LOG_FILE=tmp_log,
        setup_logging=lambda: logger,
        log_exceptions=lambda f: f,
        TRADE_COUNT=types.SimpleNamespace(inc=lambda: None),
        ERROR_COUNT=types.SimpleNamespace(inc=lambda: None),
    )
    sys.modules["prometheus_client"] = types.SimpleNamespace(
        Counter=lambda *a, **k: None,
        Gauge=lambda *a, **k: None,
        generate_latest=lambda: b"",
        CONTENT_TYPE_LATEST="text/plain",
        REGISTRY=None,
    )
    sys.modules["yaml"] = types.SimpleNamespace(
        safe_load=lambda *a, **k: {},
        safe_dump=lambda *a, **k: "",
    )
    env_mod = types.ModuleType("environment")
    env_mod.ensure_environment = lambda: None
    sys.modules["utils.environment"] = env_mod
    sys.modules["utils"] = types.SimpleNamespace(update_config=lambda *a, **k: None)
    sys.modules["utils.graceful_exit"] = types.SimpleNamespace(graceful_exit=lambda *a, **k: None)
    sys.modules["metrics"] = importlib.import_module("metrics")
    sys.modules["mlflow"] = types.SimpleNamespace(
        set_tracking_uri=lambda *a, **k: None,
        set_experiment=lambda *a, **k: None,
        start_run=contextlib.nullcontext,
        log_dict=lambda *a, **k: None,
    )
    async def async_stub(*a, **k):
        return None

    sys.modules["scheduler"] = types.SimpleNamespace(
        start_scheduler=lambda *a, **k: None,
        stop_scheduler=lambda *a, **k: None,
        schedule_retrain=lambda *a, **k: None,
        resource_reprobe=async_stub,
        run_drift_detection=lambda *a, **k: None,
        run_feature_importance_drift=lambda *a, **k: None,
        run_change_point_detection=lambda *a, **k: None,
        run_trade_analysis=lambda *a, **k: None,
        run_decision_review=lambda *a, **k: None,
        run_diagnostics=lambda *a, **k: None,
        rebuild_news_vectors=lambda *a, **k: None,
        update_regime_performance=lambda *a, **k: None,
        run_backups=lambda *a, **k: None,
        cleanup_checkpoints=lambda *a, **k: None,
    )
    mod = importlib.reload(importlib.import_module("remote_api"))
    mod.init_remote_api()
    return mod


def setup_client(tmp_path, monkeypatch):
    api = load_api(tmp_path / "app.log", monkeypatch)
    api.bots.clear()
    Path(api.LOG_FILE).write_text("line1\nline2\n")
    return api, TestClient(api.app)


def test_risk_status_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("MAX_PORTFOLIO_DRAWDOWN", "100")
    from mt5 import risk_manager as rm_mod
    importlib.reload(rm_mod)
    api, client = setup_client(tmp_path, monkeypatch)
    rm = rm_mod.risk_manager
    rm.reset()
    rm.update("b1", -60)
    rm.update("b2", -50)
    resp = client.get("/risk/status", headers={"X-API-Key": "token"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["trading_halted"] is True
    assert data["daily_loss"] == -110
    rm.reset()


def test_check_fills_returns_violations():
    rm = RiskManager(max_drawdown=100)
    violations = rm.check_fills(
        placed=10,
        filled=4,
        cancels=6,
        slippage=0.02,
        min_ratio=0.5,
        max_slippage=0.01,
        max_cancel_rate=0.5,
    )
    assert set(violations) == {"fill_ratio", "cancel_rate", "slippage"}
