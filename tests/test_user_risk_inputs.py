import pytest
import pathlib
import sys
import os
from datetime import datetime, timezone, timedelta

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
TEST_CFG = pathlib.Path(__file__).with_name("test_config.yaml")
if not TEST_CFG.exists():
    TEST_CFG.write_text("risk_per_trade: 0.01\nsymbols: ['EURUSD']\n")
os.environ["CONFIG_FILE"] = str(TEST_CFG)

import types, importlib.machinery, contextlib
telemetry_mod = types.ModuleType("telemetry")
telemetry_mod.get_tracer = lambda *a, **k: types.SimpleNamespace(
    start_as_current_span=lambda *a, **k: contextlib.nullcontext()
)
telemetry_mod.get_meter = lambda *a, **k: types.SimpleNamespace(
    create_counter=lambda *a, **k: types.SimpleNamespace(add=lambda *a, **k: None),
    create_histogram=lambda *a, **k: types.SimpleNamespace(record=lambda *a, **k: None),
)
telemetry_mod.__spec__ = importlib.machinery.ModuleSpec("telemetry", loader=None)
sys.modules.setdefault("telemetry", telemetry_mod)

scheduler_mod = types.ModuleType("scheduler")
scheduler_mod.start_scheduler = lambda: None
scheduler_mod.stop_scheduler = lambda: None
scheduler_mod.__spec__ = importlib.machinery.ModuleSpec("scheduler", loader=None)
sys.modules["scheduler"] = scheduler_mod

news_mod = types.ModuleType("news.aggregator")
class _Agg:
    def fetch(self):
        return []

    def get_news(self, start, end):
        return []

news_mod.NewsAggregator = _Agg
news_mod.__spec__ = importlib.machinery.ModuleSpec("news.aggregator", loader=None)
sys.modules.setdefault("news.aggregator", news_mod)

shadow_mod = types.ModuleType("strategy.shadow_runner")
shadow_mod.ShadowRunner = lambda *a, **k: None
shadow_mod.__spec__ = importlib.machinery.ModuleSpec("strategy.shadow_runner", loader=None)
sys.modules.setdefault("strategy.shadow_runner", shadow_mod)

evo_mod = types.ModuleType("strategy.evolution_lab")
evo_mod.EvolutionLab = lambda *a, **k: None
evo_mod.__spec__ = importlib.machinery.ModuleSpec("strategy.evolution_lab", loader=None)
sys.modules.setdefault("strategy.evolution_lab", evo_mod)

signal_queue_mod = types.ModuleType("signal_queue")
signal_queue_mod.get_async_subscriber = lambda *a, **k: None
signal_queue_mod._ROUTER = types.SimpleNamespace(algorithms={})
signal_queue_mod.get_signal_backend = lambda cfg: None
signal_queue_mod.__spec__ = importlib.machinery.ModuleSpec("signal_queue", loader=None)
sys.modules.setdefault("signal_queue", signal_queue_mod)

brokers_pkg = types.ModuleType("brokers")
brokers_pkg.__path__ = []
sys.modules.setdefault("brokers", brokers_pkg)
mt5_mod = types.ModuleType("brokers.mt5_direct")
mt5_mod.COPY_TICKS_ALL = 0
mt5_mod.copy_ticks_from = lambda *a, **k: []
mt5_mod.__spec__ = importlib.machinery.ModuleSpec("brokers.mt5_direct", loader=None)
sys.modules.setdefault("brokers.mt5_direct", mt5_mod)
conn_mgr_mod = types.ModuleType("brokers.connection_manager")
conn_mgr_mod.init = lambda *a, **k: None
conn_mgr_mod.get_active_broker = lambda: mt5_mod
conn_mgr_mod.failover = lambda: None
conn_mgr_mod.__spec__ = importlib.machinery.ModuleSpec("brokers.connection_manager", loader=None)
sys.modules.setdefault("brokers.connection_manager", conn_mgr_mod)

execution_mod = types.ModuleType("execution")
execution_mod.ExecutionEngine = lambda *a, **k: None
execution_mod.place_order = lambda *a, **k: None
execution_mod.__spec__ = importlib.machinery.ModuleSpec("execution", loader=None)
sys.modules.setdefault("execution", execution_mod)

features_mod = types.ModuleType("data.features")
features_mod.make_features = lambda *a, **k: None
features_mod.__spec__ = importlib.machinery.ModuleSpec("data.features", loader=None)
sys.modules.setdefault("data.features", features_mod)

sanitize_mod = types.ModuleType("data.sanitize")
sanitize_mod.sanitize_ticks = lambda df: df
sanitize_mod.__spec__ = importlib.machinery.ModuleSpec("data.sanitize", loader=None)
sys.modules.setdefault("data.sanitize", sanitize_mod)

scaler_mod = types.ModuleType("data.feature_scaler")
scaler_mod.FeatureScaler = lambda *a, **k: types.SimpleNamespace(partial_fit=lambda *a, **k: None, transform=lambda x: x, save=lambda *a, **k: None)
scaler_mod.__spec__ = importlib.machinery.ModuleSpec("data.feature_scaler", loader=None)
sys.modules.setdefault("data.feature_scaler", scaler_mod)

metrics_mod = types.ModuleType("metrics")
for _n in [
    "RECONNECT_COUNT",
    "FEATURE_ANOMALIES",
    "RESOURCE_RESTARTS",
    "QUEUE_DEPTH",
    "BATCH_LATENCY",
    "PRED_CACHE_HIT",
    "PRED_CACHE_HIT_RATIO",
    "PLUGIN_RELOADS",
]:
    setattr(metrics_mod, _n, types.SimpleNamespace(inc=lambda *a, **k: None))
metrics_mod.__spec__ = importlib.machinery.ModuleSpec("metrics", loader=None)
sys.modules.setdefault("metrics", metrics_mod)

analysis_dq_mod = types.ModuleType("analysis.data_quality")
analysis_dq_mod.apply_quality_checks = lambda df: (df, {})
analysis_dq_mod.__spec__ = importlib.machinery.ModuleSpec("analysis.data_quality", loader=None)
sys.modules.setdefault("analysis.data_quality", analysis_dq_mod)

analysis_da_mod = types.ModuleType("analysis.domain_adapter")
analysis_da_mod.DomainAdapter = lambda *a, **k: types.SimpleNamespace(transform=lambda x: x, save=lambda *a, **k: None, load=lambda p: types.SimpleNamespace(transform=lambda x: x))
analysis_da_mod.__spec__ = importlib.machinery.ModuleSpec("analysis.domain_adapter", loader=None)
sys.modules.setdefault("analysis.domain_adapter", analysis_da_mod)

resource_mod = types.ModuleType("utils.resource_monitor")
resource_mod.monitor = types.SimpleNamespace(capabilities=None)
resource_mod.ResourceMonitor = lambda *a, **k: types.SimpleNamespace()
resource_mod.__spec__ = importlib.machinery.ModuleSpec("utils.resource_monitor", loader=None)
sys.modules.setdefault("utils.resource_monitor", resource_mod)

alert_mod = types.ModuleType("utils.alerting")
alert_mod.send_alert = lambda *a, **k: None
alert_mod.__spec__ = importlib.machinery.ModuleSpec("utils.alerting", loader=None)
sys.modules.setdefault("utils.alerting", alert_mod)

from mt5 import state_manager


def _compute_quiet_windows(agg, minutes: int) -> list[dict]:
    now = datetime.now(timezone.utc)
    try:
        agg.fetch()
    except Exception:
        pass
    events = agg.get_news(now, now + timedelta(days=1))
    windows: list[dict] = []
    for ev in events:
        imp = str(ev.get("importance", "")).lower()
        ts = ev.get("timestamp")
        if ts and (imp.startswith("high") or imp.startswith("red")):
            start = ts - timedelta(minutes=minutes)
            end = ts + timedelta(minutes=minutes)
            currencies = ev.get("currencies") or []
            if not currencies:
                cur = ev.get("currency")
                currencies = [cur] if cur else []
            symbols = ev.get("symbols", []) or []
            windows.append(
                {
                    "start": start,
                    "end": end,
                    "currencies": currencies,
                    "symbols": symbols,
                }
            )
    return windows


def test_default_limits_persist(monkeypatch, tmp_path):
    monkeypatch.setattr(state_manager, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(state_manager, "_RISK_FILE", tmp_path / "user_risk.pkl")
    monkeypatch.setenv("INITIAL_CAPITAL", "100000")
    from mt5 import user_risk_inputs as uri
    from mt5 import risk_manager as rm_mod
    rm_mod.risk_manager = rm_mod.RiskManager(
        max_drawdown=1e9, max_total_drawdown=1e9, initial_capital=100000
    )
    dd, td, nb = uri.configure_user_risk([])
    assert (dd, td, nb) == (4900.0, 9800.0, 0)
    saved = state_manager.load_user_risk()
    assert saved == {
        "daily_drawdown": 4900.0,
        "total_drawdown": 9800.0,
        "news_blackout_minutes": 0,
        "allow_hedging": False,
    }
    assert rm_mod.risk_manager.max_drawdown == 4900.0
    assert rm_mod.risk_manager.max_total_drawdown == 9800.0


def test_cli_override_persist(monkeypatch, tmp_path):
    monkeypatch.setattr(state_manager, "_STATE_DIR", tmp_path)
    monkeypatch.setattr(state_manager, "_RISK_FILE", tmp_path / "user_risk.pkl")
    monkeypatch.setenv("INITIAL_CAPITAL", "50000")
    from mt5 import user_risk_inputs as uri
    from mt5 import risk_manager as rm_mod
    rm_mod.risk_manager = rm_mod.RiskManager(
        max_drawdown=1e9, max_total_drawdown=1e9, initial_capital=50000
    )
    args = [
        "--daily-drawdown",
        "5000",
        "--total-drawdown",
        "8000",
        "--news-blackout-minutes",
        "10",
    ]
    dd, td, nb = uri.configure_user_risk(args)
    assert (dd, td, nb) == (5000.0, 8000.0, 10)
    saved = state_manager.load_user_risk()
    assert saved == {
        "daily_drawdown": 5000.0,
        "total_drawdown": 8000.0,
        "news_blackout_minutes": 10,
        "allow_hedging": False,
    }
    assert rm_mod.risk_manager.max_drawdown == 5000.0
    assert rm_mod.risk_manager.max_total_drawdown == 8000.0


def test_drawdown_enforcement():
from mt5.risk_manager import RiskManager

    rm = RiskManager(max_drawdown=5, max_total_drawdown=8, initial_capital=100)
    rm.update("bot", -6)
    assert rm.metrics.trading_halted

    rm2 = RiskManager(max_drawdown=100, max_total_drawdown=10, initial_capital=100)
    rm2.update("bot", -6)
    assert not rm2.metrics.trading_halted
    rm2.update("bot", -5)
    assert rm2.metrics.trading_halted


def test_runtime_limit_update_enforced():
from mt5.risk_manager import RiskManager

    rm = RiskManager(max_drawdown=1000, max_total_drawdown=2000, initial_capital=100000)
    rm.update("bot", -200)
    rm.update_drawdown_limits(100, 2000)
    assert rm.metrics.daily_loss == 0
    assert not rm.metrics.trading_halted
    rm.update("bot", -150)
    assert rm.metrics.trading_halted


def test_quiet_window_minutes():
from mt5.risk_manager import RiskManager
    class DummyAgg:
        def fetch(self):
            return None

        def get_news(self, start, end):
            ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
            return [{"importance": "red", "timestamp": ts, "currencies": ["USD"]}]

    windows = _compute_quiet_windows(DummyAgg(), 20)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert windows == [
        {
            "start": ts - timedelta(minutes=20),
            "end": ts + timedelta(minutes=20),
            "currencies": ["USD"],
            "symbols": [],
        }
    ]
    rm = RiskManager(max_drawdown=100, max_total_drawdown=1000)
    rm.set_quiet_windows(windows)
    size = rm.adjust_size("EURUSD", 1.0, ts, 1)
    assert size == 0.0
