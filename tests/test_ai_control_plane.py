"""Tests for :mod:`mt5.ai_control`."""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import types

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

if "mt5.model_registry" not in sys.modules:
    _mr = types.ModuleType("mt5.model_registry")

    class _StubModelRegistry:
        def __init__(self, *args, **kwargs) -> None:
            self.selected: dict[str, object] = {}

        def refresh(self) -> None:  # pragma: no cover - trivial stub
            return None

    _mr.ModelRegistry = _StubModelRegistry
    sys.modules["mt5.model_registry"] = _mr

if "mt5.scheduler" not in sys.modules:
    _scheduler = types.ModuleType("mt5.scheduler")

    def _stub_schedule_retrain(*args, **kwargs):  # pragma: no cover - stub
        return None

    _scheduler.schedule_retrain = _stub_schedule_retrain
    sys.modules["mt5.scheduler"] = _scheduler

if "mt5.risk_manager" not in sys.modules:
    _risk_mod = types.ModuleType("mt5.risk_manager")

    class _StubRiskManager:  # pragma: no cover - placeholder for typing
        pass

    _risk_mod.RiskManager = _StubRiskManager
    _risk_mod.risk_manager = _StubRiskManager()

    def _stub_subscribe(*args, **kwargs):  # pragma: no cover - stub
        return None

    _risk_mod.subscribe_to_broker_alerts = _stub_subscribe
    sys.modules["mt5.risk_manager"] = _risk_mod

from mt5.ai_control import AIControlPlane


@dataclass
class _DummyMetrics:
    daily_loss: float = 0.0
    total_drawdown: float = 0.0
    tail_prob: float = 0.0
    risk_of_ruin: float = 0.0
    var: float = 0.0
    trading_halted: bool = False


class _DummyRiskManager:
    def __init__(self) -> None:
        self.max_drawdown = 1_000.0
        self.max_total_drawdown = 5_000.0
        self.tail_prob_limit = 0.1
        self.risk_of_ruin_threshold = 0.5
        self.allow_hedging = False
        self.metrics = _DummyMetrics()
        self._pnl_history = []
        self.updated_limits: list[tuple[float, float]] = []

    def update_drawdown_limits(self, daily: float, total: float) -> None:
        self.max_drawdown = daily
        self.max_total_drawdown = total
        self.updated_limits.append((daily, total))

    def set_allow_hedging(self, allow: bool) -> None:
        self.allow_hedging = allow


class _DummyRegistry:
    def __init__(self) -> None:
        self.refresh_count = 0
        self.selected: dict[str, object] = {}

    def refresh(self) -> None:
        self.refresh_count += 1


class _DummyMonitor:
    def __init__(self, tier: str) -> None:
        self.capability_tier = tier
        self.capabilities = type("Caps", (), {})()


class _DummyRouter:
    def __init__(self) -> None:
        self.consensus_threshold = 0.6
        self.algorithms = {"alpha": object(), "beta": object()}
        self.reward_sums = {"alpha": -10.0, "beta": 5.0}
        self.plays = {"alpha": 50, "beta": 50}
        self.demoted: list[str] = []

    def demote(self, name: str) -> None:
        self.demoted.append(name)
        self.algorithms.pop(name, None)


def test_control_plane_reacts_to_stress() -> None:
    risk = _DummyRiskManager()
    risk.metrics.daily_loss = -800.0
    risk.metrics.total_drawdown = 600.0
    risk.metrics.tail_prob = 0.2
    risk.metrics.risk_of_ruin = 0.8
    risk.metrics.var = 5.0
    risk._pnl_history = [-20.0] * 50
    registry = _DummyRegistry()
    monitor = _DummyMonitor("standard")
    router = _DummyRouter()
    retrain_calls: list[str] = []

    def _schedule_retrain(*, model: str) -> None:
        retrain_calls.append(model)

    plane = AIControlPlane(
        registry=registry,
        risk_manager=risk,
        monitor=monitor,
        router=router,
        retrain_callback=_schedule_retrain,
        loss_ratio_threshold=0.3,
        router_loss_threshold=0.05,
        retrain_trend_threshold=5.0,
        pnl_window=30,
        min_history=20,
    )

    snapshot = plane.step()

    # Model refresh triggered once capability tier is observed
    assert registry.refresh_count == 1
    # Drawdown tightened and hedging enabled
    assert risk.updated_limits, "drawdown limits should be adjusted"
    assert risk.allow_hedging is True
    # Consensus threshold tightened and underperformer demoted
    assert router.consensus_threshold > 0.6
    assert "alpha" in router.demoted
    # Retrain scheduled when losses persist
    assert retrain_calls == ["classic"]
    # Snapshot reflects stress state
    assert snapshot.tail_prob > snapshot.tail_limit


def test_control_plane_relaxes_after_recovery() -> None:
    risk = _DummyRiskManager()
    router = _DummyRouter()
    registry = _DummyRegistry()
    monitor = _DummyMonitor("standard")
    retrain_calls: list[str] = []

    def _schedule_retrain(*, model: str) -> None:
        retrain_calls.append(model)

    plane = AIControlPlane(
        registry=registry,
        risk_manager=risk,
        monitor=monitor,
        router=router,
        retrain_callback=_schedule_retrain,
        pnl_window=50,
        min_history=20,
    )

    # Simulate prior tightening
    risk.max_drawdown = 800.0
    risk.max_total_drawdown = 4_000.0
    plane._last_limits = (800.0, 4_000.0)
    router.consensus_threshold = 0.85
    risk.allow_hedging = True
    risk.metrics.daily_loss = 200.0
    risk.metrics.total_drawdown = 50.0
    risk.metrics.tail_prob = 0.01
    risk.metrics.risk_of_ruin = 0.05
    risk._pnl_history = [25.0] * 60
    plane._last_capability = monitor.capability_tier

    plane.step()

    assert risk.updated_limits, "limits should be adjusted back upwards"
    daily, total = risk.updated_limits[-1]
    assert daily > 800.0
    assert daily <= plane._base_drawdown
    assert total <= plane._base_total_drawdown
    assert risk.allow_hedging is False
    assert router.consensus_threshold == pytest.approx(plane._base_consensus or 0.6, rel=1e-3)
    assert retrain_calls == []
