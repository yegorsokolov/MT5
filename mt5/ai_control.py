"""Adaptive control plane that gives models holistic authority over the bot.

The module exposes :class:`AIControlPlane`, a lightweight decision layer that
continuously inspects risk metrics, resource availability and live strategy
performance.  It converts those signals into concrete actions such as
tightening drawdown limits, enabling hedging, demoting underperforming
strategies or triggering model retraining.

This is intentionally rule based—the heuristics run locally without requiring
network access or heavyweight optimisation libraries—yet the design keeps the
doors open for more sophisticated policies.  The controller aggregates
observations into :class:`ControlSnapshot` instances, allowing reinforcement
learning or evolutionary algorithms to be plugged in later without touching
the orchestrator.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional

from mt5.model_registry import ModelRegistry
from mt5.risk_manager import RiskManager, risk_manager as _DEFAULT_RISK_MANAGER
from mt5.scheduler import schedule_retrain as _schedule_retrain

try:  # pragma: no cover - optional import during tests
    from strategy.router import StrategyRouter
except Exception:  # pragma: no cover - router is optional in lightweight envs

    class StrategyRouter:  # type: ignore[misc, no-redef]
        """Fallback protocol used when the real router is unavailable."""

        consensus_threshold: float = 0.6
        champion: Optional[str] = None
        algorithms: Dict[str, Callable[[Mapping[str, float]], float]] = {}
        plays: Dict[str, int] = {}
        reward_sums: Dict[str, float] = {}

        def demote(self, name: str) -> None:  # pragma: no cover - stub
            return None


try:  # pragma: no cover - optional dependency during tests
    from utils.resource_monitor import ResourceMonitor, monitor as _DEFAULT_MONITOR
except Exception:  # pragma: no cover - lightweight fallback for tests

    class ResourceMonitor:  # type: ignore[misc, no-redef]
        """Minimal monitor exposing the attributes used by the controller."""

        def __init__(self) -> None:
            self.capability_tier = "lite"
            self.capabilities = type("Caps", (), {"cpus": 1, "memory_gb": 1.0})()

        def subscribe(self) -> asyncio.Queue[str]:  # pragma: no cover - stub
            return asyncio.Queue()

    _DEFAULT_MONITOR = ResourceMonitor()


logger = logging.getLogger(__name__)


@dataclass
class ControlSnapshot:
    """Aggregated view of the system used to decide the next action."""

    timestamp: float
    capability_tier: str
    daily_loss: float
    max_drawdown: float
    total_drawdown: float
    max_total_drawdown: float
    tail_prob: float
    tail_limit: float
    risk_of_ruin: float
    risk_limit: float
    var: float
    allows_hedging: bool
    trading_halted: bool
    pnl_trend: float
    history_depth: int
    router_losses: Dict[str, float] = field(default_factory=dict)
    consensus_threshold: Optional[float] = None
    router_champion: Optional[str] = None


@dataclass
class ControlDecision:
    """Concrete action emitted by :class:`AIControlPlane`."""

    action: str
    payload: Dict[str, Any] = field(default_factory=dict)


class AIControlPlane:
    """Reason about risk/strategy state and intervene automatically."""

    def __init__(
        self,
        registry: ModelRegistry,
        risk_manager: RiskManager | None = None,
        monitor: ResourceMonitor | None = None,
        *,
        router: StrategyRouter | None = None,
        retrain_callback: Callable[..., None] = _schedule_retrain,
        interval: float = 120.0,
        retrain_cooldown: float = 3_600.0,
        pnl_window: int = 120,
        loss_ratio_threshold: float = 0.65,
        tail_stress_ratio: float = 0.85,
        router_loss_threshold: float = 0.01,
        router_min_plays: int = 25,
        retrain_trend_threshold: float = 0.0,
        min_history: int = 30,
    ) -> None:
        self.registry = registry
        self.risk_manager = risk_manager or _DEFAULT_RISK_MANAGER
        self.monitor = monitor or _DEFAULT_MONITOR
        self.router: StrategyRouter | None = router
        self.retrain_callback = retrain_callback
        self.interval = interval
        self.retrain_cooldown = retrain_cooldown
        self.pnl_window = max(1, pnl_window)
        self.loss_ratio_threshold = loss_ratio_threshold
        self.tail_stress_ratio = tail_stress_ratio
        self.router_loss_threshold = router_loss_threshold
        self.router_min_plays = router_min_plays
        self.retrain_trend_threshold = retrain_trend_threshold
        self.min_history = min_history
        self._last_capability: Optional[str] = None
        self._last_retrain_ts: float = 0.0
        self._base_drawdown = self.risk_manager.max_drawdown
        self._base_total_drawdown = self.risk_manager.max_total_drawdown
        self._min_drawdown = 0.3 * self._base_drawdown if self._base_drawdown else 0.0
        if math.isfinite(self._base_total_drawdown):
            self._min_total_drawdown = 0.3 * self._base_total_drawdown
        else:
            self._min_total_drawdown = self._base_total_drawdown
        self._base_consensus: Optional[float] = None
        if router is not None:
            self._base_consensus = getattr(router, "consensus_threshold", None)
        self._last_limits = (
            self.risk_manager.max_drawdown,
            self.risk_manager.max_total_drawdown,
        )
        self._last_snapshot: Optional[ControlSnapshot] = None

    # ------------------------------------------------------------------
    def bind_router(self, router: StrategyRouter) -> None:
        """Attach a live :class:`~strategy.router.StrategyRouter` instance."""

        self.router = router
        self._base_consensus = getattr(router, "consensus_threshold", None)

    # ------------------------------------------------------------------
    async def run(self) -> None:
        """Continuously execute control cycles at the configured interval."""

        while True:
            try:
                self.step()
            except Exception:  # pragma: no cover - defensive loop guard
                logger.exception("AI control plane cycle failed")
            await asyncio.sleep(self.interval)

    # ------------------------------------------------------------------
    def step(self) -> ControlSnapshot:
        """Execute a single control iteration and return the observation."""

        snapshot = self.observe()
        decisions = self.plan(snapshot)
        if decisions:
            self.execute(decisions, snapshot)
        self._last_snapshot = snapshot
        return snapshot

    # ------------------------------------------------------------------
    def observe(self) -> ControlSnapshot:
        """Collect the latest metrics from risk, resources and routing."""

        metrics = self.risk_manager.metrics
        history = list(getattr(self.risk_manager, "_pnl_history", []))
        depth = len(history)
        if history:
            window = min(len(history), self.pnl_window)
            pnl_trend = sum(history[-window:]) / float(window)
        else:
            pnl_trend = 0.0

        router_losses: Dict[str, float] = {}
        consensus = None
        champion = None
        if self.router is not None:
            consensus = getattr(self.router, "consensus_threshold", None)
            champion = getattr(self.router, "champion", None)
            plays = getattr(self.router, "plays", {})
            rewards = getattr(self.router, "reward_sums", {})
            for name, count in plays.items():
                if count < self.router_min_plays or count <= 0:
                    continue
                mean = rewards.get(name, 0.0) / float(count)
                router_losses[name] = mean

        snapshot = ControlSnapshot(
            timestamp=time.time(),
            capability_tier=str(getattr(self.monitor, "capability_tier", "unknown")),
            daily_loss=float(metrics.daily_loss),
            max_drawdown=float(self.risk_manager.max_drawdown),
            total_drawdown=float(metrics.total_drawdown),
            max_total_drawdown=float(self.risk_manager.max_total_drawdown),
            tail_prob=float(metrics.tail_prob),
            tail_limit=float(getattr(self.risk_manager, "tail_prob_limit", 0.0)),
            risk_of_ruin=float(metrics.risk_of_ruin),
            risk_limit=float(getattr(self.risk_manager, "risk_of_ruin_threshold", 1.0)),
            var=float(metrics.var),
            allows_hedging=bool(getattr(self.risk_manager, "allow_hedging", False)),
            trading_halted=bool(metrics.trading_halted),
            pnl_trend=float(pnl_trend),
            history_depth=depth,
            router_losses=router_losses,
            consensus_threshold=consensus,
            router_champion=champion,
        )
        return snapshot

    # ------------------------------------------------------------------
    def plan(self, snapshot: ControlSnapshot) -> List[ControlDecision]:
        """Derive a sequence of actions from ``snapshot``."""

        decisions: List[ControlDecision] = []
        capability = snapshot.capability_tier
        if capability != self._last_capability:
            decisions.append(ControlDecision("refresh_models"))
            self._last_capability = capability

        loss_ratio = 0.0
        if snapshot.max_drawdown > 0:
            loss_ratio = max(0.0, -snapshot.daily_loss) / snapshot.max_drawdown

        tail_ratio = 0.0
        if snapshot.tail_limit > 0:
            tail_ratio = snapshot.tail_prob / snapshot.tail_limit

        ruin_ratio = 0.0
        if snapshot.risk_limit > 0:
            ruin_ratio = snapshot.risk_of_ruin / snapshot.risk_limit

        stress_score = max(loss_ratio, tail_ratio, ruin_ratio)
        if stress_score >= self.loss_ratio_threshold:
            new_daily = snapshot.max_drawdown * 0.8
            new_total = snapshot.max_total_drawdown * 0.8
            if self._min_drawdown and new_daily < self._min_drawdown:
                new_daily = self._min_drawdown
            if (
                math.isfinite(self._base_total_drawdown)
                and math.isfinite(new_total)
                and new_total < self._min_total_drawdown
            ):
                new_total = self._min_total_drawdown
            decisions.append(
                ControlDecision(
                    "set_drawdown_limits",
                    {
                        "daily": float(new_daily),
                        "total": float(new_total)
                        if math.isfinite(new_total)
                        else new_total,
                        "reason": "stress",
                    },
                )
            )
        elif (
            snapshot.history_depth >= self.min_history
            and loss_ratio < 0.2
            and tail_ratio < 0.5
            and ruin_ratio < 0.5
            and not snapshot.trading_halted
        ):
            target_daily = snapshot.max_drawdown * 1.1
            if self._base_drawdown:
                target_daily = min(target_daily, self._base_drawdown)
            target_total = snapshot.max_total_drawdown * 1.1
            if math.isfinite(self._base_total_drawdown):
                target_total = min(target_total, self._base_total_drawdown)
            decisions.append(
                ControlDecision(
                    "set_drawdown_limits",
                    {
                        "daily": float(target_daily),
                        "total": float(target_total)
                        if math.isfinite(target_total)
                        else target_total,
                        "reason": "recovery",
                    },
                )
            )

        if snapshot.tail_prob > snapshot.tail_limit and not snapshot.allows_hedging:
            decisions.append(ControlDecision("set_hedging", {"allow": True}))
        elif (
            snapshot.allows_hedging
            and snapshot.tail_limit > 0
            and snapshot.tail_prob < snapshot.tail_limit * 0.25
        ):
            decisions.append(ControlDecision("set_hedging", {"allow": False}))

        if (
            snapshot.history_depth >= self.min_history
            and snapshot.pnl_trend < -abs(self.retrain_trend_threshold)
            and self._cooldown_elapsed()
        ):
            decisions.append(
                ControlDecision(
                    "schedule_retrain",
                    {"model": "classic", "reason": "negative_trend"},
                )
            )

        if self.router is not None and snapshot.consensus_threshold is not None:
            base_consensus = self._base_consensus or snapshot.consensus_threshold
            target = base_consensus
            if snapshot.tail_limit > 0 and snapshot.tail_prob > snapshot.tail_limit:
                target = min(0.95, base_consensus + 0.2)
            elif (
                snapshot.tail_limit > 0
                and snapshot.tail_prob > snapshot.tail_limit * self.tail_stress_ratio
            ):
                target = min(0.9, base_consensus + 0.1)
            elif snapshot.tail_limit > 0 and snapshot.tail_prob < snapshot.tail_limit * 0.5:
                target = base_consensus
            if abs(target - snapshot.consensus_threshold) > 0.01:
                decisions.append(
                    ControlDecision(
                        "set_consensus_threshold", {"value": float(target)}
                    )
                )

        if self.router is not None and snapshot.router_losses:
            for name, mean in snapshot.router_losses.items():
                if mean < -self.router_loss_threshold:
                    decisions.append(
                        ControlDecision(
                            "demote_strategy",
                            {"name": name, "avg_reward": mean},
                        )
                    )

        return decisions

    # ------------------------------------------------------------------
    def execute(
        self, decisions: Iterable[ControlDecision], snapshot: ControlSnapshot
    ) -> None:
        """Apply ``decisions`` to the managed subsystems."""

        for decision in decisions:
            action = decision.action
            payload = decision.payload
            if action == "refresh_models":
                logger.info("Refreshing model registry due to capability change")
                try:
                    self.registry.refresh()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Model registry refresh failed")
            elif action == "set_drawdown_limits":
                daily = float(payload.get("daily", self.risk_manager.max_drawdown))
                total = payload.get("total", self.risk_manager.max_total_drawdown)
                if self._should_adjust_limits(daily, total):
                    logger.info(
                        "Adjusting drawdown limits to daily=%.2f total=%s (%s)",
                        daily,
                        "inf" if not math.isfinite(total) else f"{float(total):.2f}",
                        payload.get("reason", ""),
                    )
                    self.risk_manager.update_drawdown_limits(
                        daily, float(total) if math.isfinite(total) else total
                    )
                    self._last_limits = (daily, float(total))
            elif action == "set_hedging":
                allow = bool(payload.get("allow", False))
                logger.info("Setting hedging allowance to %s", allow)
                self.risk_manager.set_allow_hedging(allow)
            elif action == "schedule_retrain":
                model = payload.get("model", "classic")
                logger.info("Scheduling retrain for %s", model)
                try:
                    self.retrain_callback(model=model)
                    self._last_retrain_ts = time.time()
                except TypeError:
                    # legacy callbacks may not accept keyword arguments
                    self.retrain_callback(model)  # type: ignore[misc]
                    self._last_retrain_ts = time.time()
                except Exception:  # pragma: no cover - defensive
                    logger.exception("Retrain scheduling failed for %s", model)
            elif action == "set_consensus_threshold" and self.router is not None:
                value = float(payload["value"])
                logger.info("Updating router consensus threshold to %.3f", value)
                setattr(self.router, "consensus_threshold", value)
            elif action == "demote_strategy" and self.router is not None:
                name = payload.get("name")
                if name and name in getattr(self.router, "algorithms", {}):
                    logger.info("Demoting underperforming strategy %s", name)
                    try:
                        self.router.demote(name)
                    except Exception:  # pragma: no cover - defensive
                        logger.exception("Failed to demote strategy %s", name)

    # ------------------------------------------------------------------
    def _should_adjust_limits(self, daily: float, total: float) -> bool:
        current_daily, current_total = self._last_limits
        if abs(current_daily - daily) / max(current_daily, 1e-6) > 0.05:
            return True
        if math.isfinite(current_total) and math.isfinite(total):
            return abs(current_total - total) / max(current_total, 1e-6) > 0.05
        return not math.isfinite(current_total) and math.isfinite(total)

    # ------------------------------------------------------------------
    def _cooldown_elapsed(self) -> bool:
        if self._last_retrain_ts <= 0:
            return True
        return (time.time() - self._last_retrain_ts) >= self.retrain_cooldown


__all__ = ["AIControlPlane", "ControlSnapshot", "ControlDecision"]

