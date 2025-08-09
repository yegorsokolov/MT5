from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import numpy as np

from risk_manager import risk_manager


# Historical crisis periods to replay
CRISIS_PERIODS: Dict[str, Tuple[str, str]] = {
    "2008": ("2007-10-01", "2009-03-31"),
    "covid": ("2020-02-01", "2020-05-31"),
}


@dataclass
class ScenarioResult:
    """Result of a single stress scenario run."""

    scenario: str
    max_drawdown: float
    recovery_days: int
    liquidity_impact: float
    action: str


class StressScenarioRunner:
    """Replay crisis periods and shocks on recorded strategy PnL."""

    def __init__(self, strategies: Dict[str, Path], thresholds: Dict[str, float]) -> None:
        self.strategies = strategies
        self.thresholds = thresholds
        self.report_dir = Path("reports/stress")
        self.report_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Loading and basic metric calculations
    # ------------------------------------------------------------------
    def _load_pnl(self, path: Path) -> pd.Series:
        df = pd.read_csv(path, parse_dates=["date"])
        return df.set_index("date")["pnl"].astype(float)

    @staticmethod
    def _max_drawdown(pnl: pd.Series) -> float:
        cum = pnl.cumsum()
        peak = cum.cummax()
        drawdown = cum - peak
        return float(drawdown.min())

    @staticmethod
    def _recovery_time(pnl: pd.Series) -> int:
        cum = pnl.cumsum()
        peak = cum.cummax()
        drawdown = cum - peak
        recovered = np.where(drawdown == 0)[0]
        if not len(recovered):
            return len(pnl)
        return int(recovered[-1])

    @staticmethod
    def _liquidity_impact(pnl: pd.Series) -> float:
        return float(np.abs(pnl).sum())

    # ------------------------------------------------------------------
    # Scenario running
    # ------------------------------------------------------------------
    def run(self) -> Dict[str, List[ScenarioResult]]:
        """Run all scenarios for all strategies and save reports."""
        all_results: Dict[str, List[ScenarioResult]] = {}
        for name, path in self.strategies.items():
            pnl = self._load_pnl(path)
            results: List[ScenarioResult] = []

            for label, (start, end) in CRISIS_PERIODS.items():
                segment = pnl.loc[start:end]
                if not segment.empty:
                    results.append(self._evaluate(segment, label))

            # Synthetic shock: large drop on first observation
            shock_size = self.thresholds.get("shock_size", 0.1)
            shocked = pnl.copy()
            if not shocked.empty:
                shocked.iloc[0] -= shock_size
            results.append(self._evaluate(shocked, "synthetic"))

            all_results[name] = results
            self._save_report(name, results)
            self._apply_actions(results)
        return all_results

    def _evaluate(self, pnl: pd.Series, scenario: str) -> ScenarioResult:
        max_dd = self._max_drawdown(pnl)
        recovery = self._recovery_time(pnl)
        liquidity = self._liquidity_impact(pnl)
        action = "ok"
        if max_dd < -self.thresholds["max_drawdown"] or liquidity > self.thresholds["max_liquidity"]:
            action = "disable"
        elif max_dd < -self.thresholds["max_drawdown"] * 0.5:
            action = "adjust"
        return ScenarioResult(
            scenario=scenario,
            max_drawdown=-max_dd,
            recovery_days=recovery,
            liquidity_impact=liquidity,
            action=action,
        )

    def _save_report(self, name: str, results: Iterable[ScenarioResult]) -> None:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        data = [asdict(r) for r in results]
        with (self.report_dir / f"{name}_{ts}.json").open("w") as f:
            json.dump(data, f, indent=2)

    def _apply_actions(self, results: Iterable[ScenarioResult]) -> None:
        actions = {r.action for r in results}
        if "disable" in actions:
            risk_manager.halt()
        elif "adjust" in actions:
            risk_manager.max_drawdown = min(
                risk_manager.max_drawdown, self.thresholds["max_drawdown"] * 0.5
            )

