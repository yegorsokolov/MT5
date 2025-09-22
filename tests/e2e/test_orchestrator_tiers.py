import asyncio
import types
import pandas as pd
import pytest
import sys
from pathlib import Path

# Access shared test helpers
sys.path.append(str(Path(__file__).resolve().parents[1]))
from resource_tiers.utils import ResourceCapabilities, monitor, plugins

# Remove lightweight scipy stubs installed by global test config
sys.modules.pop("scipy", None)
sys.modules.pop("scipy.stats", None)

import core.orchestrator as orch


@pytest.mark.parametrize(
    "tier,caps,expected_models,expected_features",
    [
        (
            "lite",
            ResourceCapabilities(cpus=2, memory_gb=4, has_gpu=False, gpu_count=0),
            {"sentiment": "sentiment_small_quantized", "rl_policy": "rl_small_quantized"},
            {"cpu_feature"},
        ),
        (
            "standard",
            ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=False, gpu_count=0),
            {"sentiment": "sentiment_small", "rl_policy": "rl_small"},
            {"cpu_feature"},
        ),
        (
            "gpu",
            ResourceCapabilities(cpus=8, memory_gb=32, has_gpu=True, gpu_count=1),
            {"sentiment": "sentiment_large", "rl_policy": "rl_medium"},
            {"cpu_feature", "gpu_feature"},
        ),
    ],
)

def test_orchestrator_tiers(monkeypatch, tier, caps, expected_models, expected_features):
    # Configure monitor for this tier
    monitor.capabilities = caps
    monitor.capability_tier = tier
    plugins.monitor.capabilities = caps
    plugins.monitor.capability_tier = tier
    monkeypatch.setattr(monitor, "start", lambda: None)

    # Reload test plugins
    plugins.FEATURE_PLUGINS.clear()
    plugins.MODEL_PLUGINS.clear()
    plugins.RISK_CHECKS.clear()
    plugins._import_plugins(reload=True)

    feat_names = {f.__name__ for f in plugins.FEATURE_PLUGINS}
    assert feat_names == expected_features

    # Stub heavy orchestrator dependencies
    monkeypatch.setattr(orch.state_sync, "pull_event_store", lambda: None)
    monkeypatch.setattr(orch.state_sync, "check_health", lambda max_lag: True)
    monkeypatch.setattr(orch, "subscribe_to_broker_alerts", lambda: None)
    monkeypatch.setattr(orch, "CanaryManager", lambda registry: types.SimpleNamespace(evaluate_all=lambda: None))
    monkeypatch.setattr(
        orch,
        "NewsAggregator",
        lambda: types.SimpleNamespace(fetch=lambda: None, get_news=lambda *a, **k: []),
    )
    monkeypatch.setattr(
        orch,
        "ShadowRunner",
        lambda name, handler: types.SimpleNamespace(run=lambda self: None),
    )
    monkeypatch.setattr(orch, "EvolutionLab", lambda base, register: types.SimpleNamespace())
    monkeypatch.setattr(orch, "record_metric", lambda *a, **k: None)
    monkeypatch.setattr(orch, "send_alert", lambda msg: None)
    monkeypatch.setattr(orch, "risk_manager", types.SimpleNamespace(status=lambda: {}))
    from mt5 import scheduler
    monkeypatch.setattr(scheduler, "subscribe_retrain_events", lambda *a, **k: None)

    # Avoid background coroutines
    monkeypatch.setattr(orch.Orchestrator, "_watch", lambda self: None)
    monkeypatch.setattr(orch.Orchestrator, "_sync_monitor", lambda self: None)
    monkeypatch.setattr(orch.Orchestrator, "_daily_summary", lambda self: None)
    monkeypatch.setattr(orch.Orchestrator, "_watch_services", lambda self: None)
    monkeypatch.setattr(orch.Orchestrator, "_update_quiet_windows", lambda self: None)

    # Track state replay calls
    calls: list[str] = []

    def fake_reprocess(path):
        calls.append("replay")
        return pd.DataFrame({"pnl_new": [0.0], "pnl_old": [0.0]})

    monkeypatch.setattr(orch.replay, "reprocess", fake_reprocess)

    # Avoid scheduling background tasks
    dummy_loop = types.SimpleNamespace(create_task=lambda *a, **k: None)
    monkeypatch.setattr(asyncio, "get_running_loop", lambda: dummy_loop)
    monkeypatch.setattr(asyncio, "get_event_loop", lambda: dummy_loop)

    # Boot orchestrator
    o = orch.Orchestrator(mon=monitor)
    o._start()

    # Simulate a live tick through feature plugins
    data: dict[str, object] = {}
    for feat in plugins.FEATURE_PLUGINS:
        data = feat(data)
    assert data, "Feature generation failed"
    assert expected_features <= set(data.keys())

    # Assert correct model variants selected
    for task, model_name in expected_models.items():
        assert o.registry.get(task) == model_name

    # Ensure state replay path invoked
    asyncio.run(o._run_reprocess())
    assert calls == ["replay"], "State replay not triggered"
