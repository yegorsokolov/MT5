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
scheduler_mod.__spec__ = importlib.machinery.ModuleSpec("scheduler", loader=None)
sys.modules.setdefault("scheduler", scheduler_mod)

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

import state_manager
from core.orchestrator import _compute_quiet_windows


def test_prompt_and_persist(monkeypatch, tmp_path):
    prompts = []
    vals = iter(["10", "50", "15", "0"])

    def fake_input(msg: str) -> str:
        prompts.append(msg)
        return next(vals)

    import importlib
    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(state_manager, "_STATE_DIR", tmp_path)
    orig_loader = state_manager.load_user_risk
    monkeypatch.setattr(state_manager, "load_user_risk", lambda: None)
    monkeypatch.setenv("SKIP_USER_RISK_PROMPT", "1")
    import user_risk_inputs as uri
    dd, td, nb = uri.configure_user_risk([])
    monkeypatch.setattr(state_manager, "load_user_risk", orig_loader)
    assert dd == 10.0 and td == 50.0 and nb == 15
    saved = state_manager.load_user_risk()
    assert saved == {"daily_drawdown": 10.0, "total_drawdown": 50.0, "news_blackout_minutes": 15}
    assert any("daily" in p.lower() for p in prompts)


def test_drawdown_enforcement():
    from risk_manager import RiskManager

    rm = RiskManager(max_drawdown=5, max_total_drawdown=8, initial_capital=100)
    rm.update("bot", -6)
    assert rm.metrics.trading_halted

    rm2 = RiskManager(max_drawdown=100, max_total_drawdown=10, initial_capital=100)
    rm2.update("bot", -6)
    assert not rm2.metrics.trading_halted
    rm2.update("bot", -5)
    assert rm2.metrics.trading_halted


def test_quiet_window_minutes():
    from risk_manager import RiskManager
    class DummyAgg:
        def fetch(self):
            return None

        def get_news(self, start, end):
            ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
            return [{"importance": "High", "timestamp": ts}]

    windows = _compute_quiet_windows(DummyAgg(), 20)
    ts = datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert windows == [(ts - timedelta(minutes=20), ts + timedelta(minutes=20))]
    rm = RiskManager(max_drawdown=100, max_total_drawdown=1000)
    rm.set_quiet_windows(windows)
    size = rm.adjust_size("EURUSD", 1.0, ts, 1)
    assert size == 0.0
