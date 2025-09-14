from __future__ import annotations

"""Train and evaluate candidate strategies before live deployment.

The :class:`StrategyLab` orchestrates the full workflow for experimental
strategies.  Candidates are trained on historical tick data via a supplied
``train_fn`` then deployed in shadow mode using :class:`strategy.shadow_runner.ShadowRunner`.

Each shadow runner publishes evaluation metrics (PnL, drawdown and Sharpe
ratio) to a queue which the lab monitors.  Once a strategy's live metrics
exceed configurable thresholds it is promoted to the
:class:`strategy.router.StrategyRouter` for live trading.  All metric updates
are persisted to ``history_path`` for auditability.
"""

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional
from datetime import datetime

import asyncio
import pandas as pd

from analytics.metrics_aggregator import record_metric
from risk_manager import RiskManager
from .shadow_runner import ShadowRunner
from .router import StrategyRouter, Algorithm


@dataclass
class StrategyLab:
    """Manage training, shadow deployment and promotion of strategies."""

    train_fn: Callable[[pd.DataFrame, Optional[Any]], Algorithm]
    router: StrategyRouter
    thresholds: Dict[str, float]
    history_path: Path = Path("reports/strategy_lab.csv")
    metrics_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    candidates: Dict[str, Algorithm] = field(default_factory=dict)
    runners: Dict[str, ShadowRunner] = field(default_factory=dict)
    tasks: Dict[str, asyncio.Task] = field(default_factory=dict)
    max_drawdown: float = float("inf")
    max_total_drawdown: float = float("inf")
    fill_ratio_threshold: float = 0.5
    max_slippage: float = 0.01
    max_cancel_rate: float = 0.5
    limit_window: int = 100
    risk_managers: Dict[str, RiskManager] = field(default_factory=dict)
    policy_versions: Dict[str, str] = field(default_factory=dict)
    _limit_stats: Dict[str, Dict[str, deque]] = field(default_factory=dict)

    async def train_and_deploy(
        self, name: str, data: pd.DataFrame, init: Optional[Any] = None
    ) -> None:
        """Train a new strategy and start its shadow runner.

        Parameters
        ----------
        name:
            Identifier for the candidate strategy.
        data:
            Historical data used for training.
        init:
            Optional reinforcement learning or meta-learned initialisation
            passed to ``train_fn`` for warm starting policies.
        """

        algo = self.train_fn(data, init)
        self.candidates[name] = algo
        runner = ShadowRunner(name=name, handler=algo, metrics_queue=self.metrics_queue)
        self.runners[name] = runner
        task = asyncio.create_task(runner.run())
        self.tasks[name] = task
        rm = RiskManager(
            max_drawdown=self.max_drawdown,
            max_total_drawdown=self.max_total_drawdown,
            initial_capital=1.0,
        )
        self.risk_managers[name] = rm
        # Record a simple timestamp based policy version for auditability
        self.policy_versions[name] = datetime.utcnow().isoformat()
        self._limit_stats[name] = {
            "placed": deque(maxlen=self.limit_window),
            "filled": deque(maxlen=self.limit_window),
            "cancels": deque(maxlen=self.limit_window),
            "slippage": deque(maxlen=self.limit_window),
        }

    # ------------------------------------------------------------------
    def _persist(self, rec: Dict[str, Any]) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        header = not self.history_path.exists()
        with self.history_path.open("a") as f:
            if header:
                f.write(
                    "name,version,pnl,drawdown,sharpe,fill_ratio,cancel_rate,slippage,violations\n"
                )
            ver = rec.get("version") or self.policy_versions.get(rec.get("name", ""), "")
            f.write(
                f"{rec['name']},{ver},{rec.get('pnl',0.0):.6f},{rec.get('drawdown',0.0):.6f},{rec.get('sharpe',0.0):.6f},{rec.get('fill_ratio',0.0):.6f},{rec.get('cancel_rate',0.0):.6f},{rec.get('slippage',0.0):.6f},{rec.get('violations','')}\n"
            )

    # ------------------------------------------------------------------
    def _meets_thresholds(self, rec: Dict[str, Any]) -> bool:
        return all(rec.get(k, float("-inf")) >= v for k, v in self.thresholds.items())

    def _demote(self, name: str) -> None:
        """Remove a candidate strategy and stop its runner."""
        runner = self.runners.pop(name, None)
        task = self.tasks.pop(name, None)
        if task is not None:
            task.cancel()
        if runner is not None:
            # Runner task already cancelled but ensure cleanup
            pass
        self.candidates.pop(name, None)
        self.risk_managers.pop(name, None)
        self._limit_stats.pop(name, None)
        self.router.algorithms.pop(name, None)

    # ------------------------------------------------------------------
    async def monitor(self) -> None:
        """Consume shadow metrics and promote successful strategies."""

        promoted: set[str] = set()
        while True:
            rec = await self.metrics_queue.get()
            name = rec.get("name")
            if not name or name in promoted:
                continue
            rm = self.risk_managers.get(name)
            if rm is not None:
                rm.check_drawdown(rec.get("pnl", 0.0))
                limit_orders = rec.get("limit_orders")
                if limit_orders:
                    stats = self._limit_stats.get(name)
                    if stats is not None:
                        stats["placed"].append(int(limit_orders.get("placed", 0)))
                        stats["filled"].append(int(limit_orders.get("filled", 0)))
                        stats["cancels"].append(int(limit_orders.get("cancels", 0)))
                        stats["slippage"].append(float(limit_orders.get("slippage", 0.0)))
                        placed_tot = sum(stats["placed"])
                        filled_tot = sum(stats["filled"])
                        cancels_tot = sum(stats["cancels"])
                        avg_slip = (
                            sum(stats["slippage"]) / len(stats["slippage"])
                            if stats["slippage"]
                            else 0.0
                        )
                        violations = rm.check_fills(
                            placed=int(placed_tot),
                            filled=int(filled_tot),
                            cancels=int(cancels_tot),
                            slippage=float(avg_slip),
                            min_ratio=self.fill_ratio_threshold,
                            max_slippage=self.max_slippage,
                            max_cancel_rate=self.max_cancel_rate,
                        )
                        rec["fill_ratio"] = rm.metrics.fill_ratio
                        rec["cancel_rate"] = rm.metrics.cancel_rate
                        rec["slippage"] = rm.metrics.slippage
                        rec["violations"] = ";".join(violations)
                        try:
                            record_metric(
                                "limit_fill_ratio", rm.metrics.fill_ratio, {"name": name}
                            )
                            record_metric(
                                "limit_cancel_rate", rm.metrics.cancel_rate, {"name": name}
                            )
                            record_metric(
                                "limit_slippage", rm.metrics.slippage, {"name": name}
                            )
                            for v in violations:
                                record_metric(
                                    "limit_violation", 1.0, {"name": name, "type": v}
                                )
                        except Exception:
                            pass
                else:
                    rec["fill_ratio"] = rm.metrics.fill_ratio
                    rec["cancel_rate"] = rm.metrics.cancel_rate
                    rec["slippage"] = rm.metrics.slippage
                    rec.setdefault("violations", "")
                if rm.metrics.trading_halted:
                    self._persist(rec)
                    try:
                        record_metric(
                            "strategy_demoted", 1.0, {"name": name, "reason": "risk"}
                        )
                    except Exception:
                        pass
                    self._demote(name)
                    continue
            self._persist(rec)
            try:
                record_metric("shadow_pnl", rec.get("pnl", 0.0), {"name": name})
                record_metric("shadow_drawdown", rec.get("drawdown", 0.0), {"name": name})
            except Exception:
                pass
            if self._meets_thresholds(rec) and rec.get("verified", True) and not rec.get("violations"):
                algo = self.candidates.get(name)
                if algo is not None:
                    self.router.promote(name, algo)
                    promoted.add(name)
                    version = self.policy_versions.get(name)
                    try:
                        record_metric(
                            "policy_version", 1.0, {"name": name, "version": version}
                        )
                        record_metric("strategy_promoted", 1.0, {"name": name})
                    except Exception:
                        pass

__all__ = ["StrategyLab"]
