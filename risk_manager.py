from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, TYPE_CHECKING
import asyncio
import os

import numpy as np
import pandas as pd
from scheduler import start_scheduler
from risk import risk_of_ruin
from risk.budget_allocator import BudgetAllocator
from risk.net_exposure import NetExposure
from analytics.metrics_store import record_metric
from analytics import decision_logger
from portfolio.robust_optimizer import RobustOptimizer
from analysis.extreme_value import estimate_tail_probability, log_evt_result
try:
    from news.impact_model import get_impact
except Exception:  # pragma: no cover - optional dependency
    def get_impact(*args, **kwargs):  # type: ignore
        return 0.0, 0.0
try:
    from utils.alerting import send_alert
except Exception:  # pragma: no cover - utils may be stubbed in tests
    def send_alert(msg: str) -> None:  # type: ignore
        return

if TYPE_CHECKING:  # pragma: no cover - used only for typing
    from risk.tail_hedger import TailHedger


_IMPACT_THRESHOLD = float(os.getenv("NEWS_IMPACT_THRESHOLD", "0.5"))
_UNCERTAINTY_THRESHOLD = float(os.getenv("NEWS_IMPACT_UNCERTAINTY", "1.0"))
_IMPACT_BOOST = float(os.getenv("NEWS_IMPACT_BOOST", "1.5"))
_DOWNSCALE = float(os.getenv("NEWS_IMPACT_DOWNSCALE", "0.5"))


@dataclass
class RiskMetrics:
    exposure: float = 0.0
    daily_loss: float = 0.0
    var: float = 0.0
    risk_of_ruin: float = 0.0
    tail_prob: float = 0.0
    trading_halted: bool = False
    factor_contributions: Dict[str, float] = field(default_factory=dict)


class RiskManager:
    """Aggregate risk metrics from trade/PnL events."""

    def __init__(
        self,
        max_drawdown: float,
        max_var: float = float("inf"),
        var_window: int = 100,
        tail_hedger: "TailHedger" | None = None,
        risk_of_ruin_threshold: float = 1.0,
        initial_capital: float = 1.0,
        optimizer: RobustOptimizer | None = None,
        tail_threshold: float | None = None,
        tail_prob_limit: float = 0.05,
        max_long_exposure: float = float("inf"),
        max_short_exposure: float = float("inf"),
    ) -> None:
        self.max_drawdown = max_drawdown
        self.max_var = max_var
        self.var_window = var_window
        self.metrics = RiskMetrics()
        self._pnl_history: List[float] = []
        self._factor_history: List[Dict[str, float]] = []
        self._bot_pnl_history: Dict[str, List[float]] = {}
        self.budget_allocator = BudgetAllocator(initial_capital)
        self.tail_hedger = tail_hedger
        self.risk_of_ruin_threshold = risk_of_ruin_threshold
        self.initial_capital = initial_capital
        self.robust_optimizer = optimizer or RobustOptimizer()
        self._regime_pnl_history: Dict[int, Dict[str, List[float]]] = {}
        self._last_regime: int | None = None
        self.tail_threshold = tail_threshold or max_drawdown
        self.tail_prob_limit = tail_prob_limit
        self.quiet_windows: list[tuple[pd.Timestamp, pd.Timestamp]] = []
        self.net_exposure = NetExposure(
            max_long=max_long_exposure, max_short=max_short_exposure
        )

    def attach_tail_hedger(self, hedger: "TailHedger") -> None:
        """Attach a :class:`~risk.tail_hedger.TailHedger` instance."""
        self.tail_hedger = hedger

    def adjust_size(
        self, symbol: str, size: float, timestamp: str | pd.Timestamp, direction: int
    ) -> float:
        """Return size adjusted for predicted news impact.

        Parameters
        ----------
        symbol: str
            Instrument symbol.
        size: float
            Proposed position size.
        timestamp: str or pandas.Timestamp
            Event timestamp.
        direction: int
            +1 for long trades, -1 for shorts.
        """
        ts = pd.Timestamp(timestamp)
        if self.metrics.trading_halted:
            try:
                record_metric("trades_skipped_halt", 1)
            except Exception:
                pass
            decision_logger.log(
                pd.DataFrame(
                    [
                        {
                            "timestamp": ts.isoformat(),
                            "symbol": symbol,
                            "event": "skip",
                            "position_size": 0.0,
                            "reason": "trading_halted",
                        }
                    ]
                )
            )
            return 0.0
        for start, end in self.quiet_windows:
            if start <= ts <= end:
                try:
                    record_metric("trades_skipped_news", 1)
                except Exception:
                    pass
                decision_logger.log(
                    pd.DataFrame(
                        [
                            {
                                "timestamp": ts.isoformat(),
                                "symbol": symbol,
                                "event": "skip",
                                "position_size": 0.0,
                                "reason": "quiet_window",
                            }
                        ]
                    )
                )
                return 0.0
        impact, uncert = get_impact(symbol, ts)
        if uncert > _UNCERTAINTY_THRESHOLD:
            size *= _DOWNSCALE
        if impact is not None and abs(impact) > _IMPACT_THRESHOLD:
            if impact * direction < 0:
                try:
                    record_metric("trades_skipped_news", 1)
                except Exception:
                    pass
                decision_logger.log(
                    pd.DataFrame(
                        [
                            {
                                "timestamp": ts.isoformat(),
                                "symbol": symbol,
                                "event": "skip",
                                "position_size": 0.0,
                                "reason": "news_impact",
                            }
                        ]
                    )
                )
                return 0.0
            size *= _IMPACT_BOOST
        proposed = size * direction
        allowed = self.net_exposure.limit(symbol, proposed)
        if allowed == 0.0:
            try:
                record_metric("trades_skipped_exposure", 1)
            except Exception:
                pass
            decision_logger.log(
                pd.DataFrame(
                    [
                        {
                            "timestamp": ts.isoformat(),
                            "symbol": symbol,
                            "event": "skip",
                            "position_size": 0.0,
                            "reason": "exposure_limit",
                        }
                    ]
                )
            )
            return 0.0
        size = abs(allowed)
        decision_logger.log(
            pd.DataFrame(
                [
                    {
                        "timestamp": ts.isoformat(),
                        "symbol": symbol,
                        "event": "trade",
                        "position_size": size,
                    }
                ]
            )
        )
        return size

    def set_quiet_windows(
        self, windows: list[tuple[pd.Timestamp, pd.Timestamp]]
    ) -> None:
        """Define trading blackout periods around high-impact news."""
        self.quiet_windows = windows

    def update(
        self,
        bot_id: str,
        pnl: float,
        exposure: float = 0.0,
        check_hedge: bool = True,
        factor_returns: Dict[str, float] | None = None,
        *,
        symbol: str | None = None,
    ) -> None:
        """Record a trade or PnL update from ``bot_id``."""
        if symbol is not None:
            self.net_exposure.update(symbol, exposure)
            totals = self.net_exposure.totals()
            self.metrics.exposure = totals["net"]
        else:
            self.metrics.exposure += exposure
        self.metrics.daily_loss += pnl
        self._pnl_history.append(pnl)
        bot_hist = self._bot_pnl_history.setdefault(bot_id, [])
        bot_hist.append(pnl)
        regime = None
        if factor_returns is not None:
            self._factor_history.append(factor_returns)
            regime = factor_returns.get("regime")
            if regime is None:
                regime = factor_returns.get("market_regime")
            if regime is not None:
                reg = int(regime)
                self._last_regime = reg
                reg_hist = self._regime_pnl_history.setdefault(reg, {})
                reg_hist.setdefault(bot_id, []).append(pnl)
        if len(self._pnl_history) > self.var_window:
            self._pnl_history.pop(0)
            if self._factor_history:
                self._factor_history.pop(0)
        if len(bot_hist) > self.var_window:
            bot_hist.pop(0)
        if self._pnl_history:
            self.metrics.var = float(-np.percentile(self._pnl_history, 1))
            returns = pd.Series(self._pnl_history) / self.initial_capital
            self.metrics.risk_of_ruin = float(
                risk_of_ruin(returns, self.initial_capital)
            )
            if self._factor_history and len(self._factor_history) == len(self._pnl_history):
                factors_df = pd.DataFrame(self._factor_history)
                from portfolio.factor_risk import FactorRisk

                fr = FactorRisk(factors_df)
                contrib = fr.factor_contributions(returns)
                self.metrics.factor_contributions = contrib.to_dict()
            tail_prob, evt = estimate_tail_probability(
                returns,
                -self.tail_threshold / self.initial_capital,
            )
            self.metrics.tail_prob = tail_prob
            log_evt_result(evt, breach=tail_prob > self.tail_prob_limit)
            if self.tail_hedger is not None:
                self.tail_hedger.hedge_ratio = 1.0 + tail_prob
        breach_reason = None
        if self.metrics.daily_loss <= -self.max_drawdown:
            breach_reason = (
                f"max drawdown exceeded: {self.metrics.daily_loss:.2f}"
                f" <= {-self.max_drawdown}"
            )
        elif self.metrics.var > self.max_var:
            breach_reason = (
                f"VaR limit exceeded: {self.metrics.var:.2f} > {self.max_var}"
            )
        elif self.metrics.risk_of_ruin > self.risk_of_ruin_threshold:
            breach_reason = (
                "risk of ruin exceeded: "
                f"{self.metrics.risk_of_ruin:.2f} > {self.risk_of_ruin_threshold}"
            )
        if breach_reason:
            self.metrics.trading_halted = True
            send_alert(f"Risk limit breached: {breach_reason}")
        if check_hedge and self.tail_hedger is not None:
            self.tail_hedger.evaluate()
        # Record metrics for analytics
        try:
            record_metric("pnl", pnl, {"bot_id": bot_id})
            record_metric("exposure", self.metrics.exposure)
            record_metric("var", self.metrics.var)
            record_metric("risk_of_ruin", self.metrics.risk_of_ruin)
        except Exception:
            pass
        decision_logger.log(
            pd.DataFrame(
                [
                    {
                        "timestamp": pd.Timestamp.utcnow().isoformat(),
                        "bot_id": bot_id,
                        "event": "trade_result",
                        "pnl": pnl,
                        "exposure": exposure,
                    }
                ]
            )
        )

    async def monitor(self, queue: "asyncio.Queue[tuple[str, float, float]]") -> None:
        """Consume trade/PnL events from ``queue`` and update metrics."""
        while True:
            bot_id, pnl, exposure = await queue.get()
            self.update(bot_id, pnl, exposure)

    async def watch_feed_divergence(self, queue: asyncio.Queue) -> None:
        """Subscribe to broker divergence alerts and halt trading when necessary."""
        while True:
            event = await queue.get()
            if getattr(event, "resolved", False):
                self.metrics.trading_halted = False
            else:
                self.metrics.trading_halted = True

    def status(self) -> Dict[str, float | bool]:
        """Return current aggregated risk metrics."""
        totals = self.net_exposure.totals()
        return {
            "exposure": self.metrics.exposure,
            "long_exposure": totals["long"],
            "short_exposure": totals["short"],
            "daily_loss": self.metrics.daily_loss,
            "var": self.metrics.var,
            "risk_of_ruin": self.metrics.risk_of_ruin,
            "tail_prob": self.metrics.tail_prob,
            "trading_halted": self.metrics.trading_halted,
            "factor_contributions": self.metrics.factor_contributions,
        }

    def halt(self) -> None:
        self.metrics.trading_halted = True

    def reset(self) -> None:
        self.metrics = RiskMetrics()
        self._pnl_history.clear()
        self._factor_history.clear()
        self._bot_pnl_history.clear()
        self.budget_allocator.budgets.clear()
        self._regime_pnl_history.clear()
        self._last_regime = None

    # Risk budgets ----------------------------------------------------
    def rebalance_budgets(
        self,
        regime: int | None = None,
        factor_exposures: Dict[str, pd.Series] | None = None,
    ) -> Dict[str, float]:
        """Recompute risk budgets using a robust optimiser.

        When ``regime`` is provided (or inferred from the most recent update) the
        allocation is computed from returns observed in that market regime using a
        :class:`~portfolio.robust_optimizer.RobustOptimizer` to guard against
        estimation error.  If insufficient data is available the allocator falls
        back to the naive :class:`~risk.budget_allocator.BudgetAllocator` weights.
        """

        returns = {
            bot: pd.Series(hist) / self.initial_capital
            for bot, hist in self._bot_pnl_history.items()
            if hist
        }
        budgets = self.budget_allocator.allocate(returns)

        # If factor exposures are supplied, derive a covariance matrix from them
        # to better reflect cross-asset correlations.  This provides a more
        # stable estimate of risk when only limited history is available.
        if factor_exposures:
            exp_df = pd.DataFrame(factor_exposures).T.fillna(0.0)
            mat = exp_df.to_numpy()
            cov = mat @ mat.T
            cov += np.eye(cov.shape[0]) * 1e-6
            inv_var = 1.0 / np.diag(cov)
            weights = inv_var / inv_var.sum()
            total = sum(budgets.values()) or self.budget_allocator.capital
            self.budget_allocator.budgets = {
                bot: total * w for bot, w in zip(exp_df.index, weights)
            }
            return self.budget_allocator.budgets

        reg = regime if regime is not None else self._last_regime
        if reg is None or reg not in self._regime_pnl_history:
            return budgets

        reg_hist = {
            bot: pd.Series(hist) / self.initial_capital
            for bot, hist in self._regime_pnl_history[reg].items()
            if len(hist) >= 2
        }
        if len(reg_hist) < 2:
            return budgets

        min_len = min(len(s) for s in reg_hist.values())
        data = np.vstack([s.tail(min_len).to_numpy() for s in reg_hist.values()])
        mu = data.mean(axis=1)
        cov = np.cov(data)
        weights = self.robust_optimizer.compute_weights(mu, cov)
        total = sum(budgets.values()) or self.budget_allocator.capital
        self.budget_allocator.budgets = {
            bot: total * w for bot, w in zip(reg_hist.keys(), weights)
        }
        return self.budget_allocator.budgets

    def adjust_position_size(self, bot_id: str, size: float) -> float:
        """Scale ``size`` by the risk budget for ``bot_id``."""
        frac = self.budget_allocator.fraction(bot_id)
        scale = max(0.0, 1.0 - self.metrics.tail_prob)
        return size * frac * scale


from risk.tail_hedger import TailHedger

MAX_DRAWDOWN = float(os.getenv("MAX_PORTFOLIO_DRAWDOWN", "1e9"))
MAX_VAR = float(os.getenv("MAX_VAR", "1e9"))
TAIL_HEDGE_VAR = float(os.getenv("TAIL_HEDGE_VAR", str(MAX_VAR)))
RISK_OF_RUIN_THRESHOLD = float(os.getenv("RISK_OF_RUIN_THRESHOLD", "1.0"))
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "1.0"))
MAX_LONG_EXPOSURE = float(os.getenv("MAX_LONG_EXPOSURE", "inf"))
MAX_SHORT_EXPOSURE = float(os.getenv("MAX_SHORT_EXPOSURE", "inf"))

risk_manager = RiskManager(
    MAX_DRAWDOWN,
    MAX_VAR,
    risk_of_ruin_threshold=RISK_OF_RUIN_THRESHOLD,
    initial_capital=INITIAL_CAPITAL,
    max_long_exposure=MAX_LONG_EXPOSURE,
    max_short_exposure=MAX_SHORT_EXPOSURE,
)
tail_hedger = TailHedger(risk_manager, TAIL_HEDGE_VAR)
risk_manager.attach_tail_hedger(tail_hedger)


def subscribe_to_broker_alerts(rm: RiskManager | None = None) -> asyncio.Task | None:
    """Subscribe ``rm`` to broker divergence alerts."""
    rm = rm or risk_manager
    try:
        from data.tick_aggregator import divergence_alerts
    except Exception:
        return None
    queue = divergence_alerts()
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.get_event_loop()
    return loop.create_task(rm.watch_feed_divergence(queue))

start_scheduler()
