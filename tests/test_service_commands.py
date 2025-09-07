import json
import types
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
from resource_tiers.utils import monitor, plugins  # noqa: E402

state_manager_stub = types.ModuleType("state_manager")
sys.modules["state_manager"] = state_manager_stub
risk_manager_stub = types.ModuleType("risk_manager")
risk_manager_stub.risk_manager = types.SimpleNamespace(status=lambda: {})
risk_manager_stub.subscribe_to_broker_alerts = lambda: None
sys.modules["risk_manager"] = risk_manager_stub
news_pkg = types.ModuleType("news")
news_agg = types.ModuleType("news.aggregator")


class _Agg:
    def fetch(self):
        return None

    def get_news(self, *a, **k):
        return []


news_agg.NewsAggregator = _Agg
sys.modules["news"] = news_pkg
sys.modules["news.aggregator"] = news_agg
strategy_pkg = types.ModuleType("strategy")
shadow_runner_mod = types.ModuleType("strategy.shadow_runner")
shadow_runner_mod.ShadowRunner = lambda name, handler: types.SimpleNamespace(
    run=lambda self: None
)
evolution_lab_mod = types.ModuleType("strategy.evolution_lab")
evolution_lab_mod.EvolutionLab = lambda base, register: types.SimpleNamespace()
sys.modules["strategy"] = strategy_pkg
sys.modules["strategy.shadow_runner"] = shadow_runner_mod
sys.modules["strategy.evolution_lab"] = evolution_lab_mod

# ensure lightweight plugin environment
plugins.FEATURE_PLUGINS.clear()
plugins.MODEL_PLUGINS.clear()
plugins.RISK_CHECKS.clear()
plugins._import_plugins(reload=True)

import core.orchestrator as orch  # noqa: E402


def test_service_commands_env(monkeypatch):
    monkeypatch.setenv(
        "SERVICE_COMMANDS",
        json.dumps({"signal_queue": ["echo", "hi"], "extra": ["python", "extra.py"]}),
    )
    monkeypatch.setattr(orch, "CanaryManager", lambda registry: types.SimpleNamespace())
    monkeypatch.setattr(
        orch, "EvolutionLab", lambda base, register: types.SimpleNamespace()
    )

    o = orch.Orchestrator(mon=monitor)
    assert o._service_cmds["signal_queue"] == ["echo", "hi"]
    assert o._service_cmds["realtime_train"] == ["python", "realtime_train.py"]
    assert o._service_cmds["extra"] == ["python", "extra.py"]
