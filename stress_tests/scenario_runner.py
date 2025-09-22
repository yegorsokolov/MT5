from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Tuple, Optional, Any

import pandas as pd
import numpy as np

from analytics.metrics_store import record_metric
from mt5.risk_manager import risk_manager


# Historical crisis periods to replay
CRISIS_PERIODS: Dict[str, Tuple[str, str]] = {
    "2008": ("2007-10-01", "2009-03-31"),
    "covid": ("2020-02-01", "2020-05-31"),
}


@dataclass
class ScenarioResult:
    """Result of a single stress scenario run."""

    scenario: str
    pnl: float
    max_drawdown: float
    recovery_days: int
    liquidity_impact: float
    hedging_effectiveness: float
    action: str


class StressScenarioRunner:
    """Replay crisis periods and shocks on recorded strategy PnL."""

    def __init__(
        self,
        strategies: Dict[str, Path],
        thresholds: Dict[str, float],
        report_dir: Path | str = "reports/stress",
        scenario_report_dir: Path | str = "reports/scenario_tests",
    ) -> None:
        self.strategies = strategies
        self.thresholds = thresholds
        self.report_dir = Path(report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        # Separate directory used for audit of generated scenario paths
        self.scenario_report_dir = Path(scenario_report_dir)
        self.scenario_report_dir.mkdir(parents=True, exist_ok=True)

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
    def run(
        self,
        scenario_generator: Optional[object] = None,
        synthetic_generator: Optional[object] = None,
        n_synthetic: int = 0,
    ) -> Dict[str, List[ScenarioResult]]:
        """Run all scenarios for all strategies and save reports.

        Parameters
        ----------
        scenario_generator:
            Optional ``ScenarioGenerator`` used to create additional shocked
            scenarios from the base PnL series.
        synthetic_generator:
            Optional object exposing a ``generate(length: int) -> ArrayLike``
            method.  When provided, additional synthetic PnL paths will be
            generated and evaluated alongside historical scenarios.
        n_synthetic:
            Number of synthetic paths to generate for each strategy.  Ignored if
            ``synthetic_generator`` is ``None``.
        """
        all_results: Dict[str, List[ScenarioResult]] = {}
        for name, path in self.strategies.items():
            pnl = self._load_pnl(path)
            results: List[ScenarioResult] = []
            generated_paths: Dict[str, List[float]] = {}

            # Baseline metrics
            base_result = self._evaluate(pnl, "baseline", None)
            base_drawdown = base_result.max_drawdown
            results.append(base_result)

            for label, (start, end) in CRISIS_PERIODS.items():
                segment = pnl.loc[start:end]
                if not segment.empty:
                    results.append(self._evaluate(segment, label, base_drawdown))

            # Synthetic shock: large drop on first observation
            shock_size = self.thresholds.get("shock_size", 0.1)
            shocked = pnl.copy()
            if not shocked.empty:
                shocked.iloc[0] -= shock_size
            results.append(self._evaluate(shocked, "synthetic", base_drawdown))

            # Additional user defined scenarios
            if scenario_generator is not None:
                generated = scenario_generator.generate_pnl(pnl)
                for label, series in generated.items():
                    results.append(self._evaluate(series, label, base_drawdown))
                    generated_paths[label] = [float(x) for x in series]

            # Optional GAN/diffusion generated paths
            if synthetic_generator is not None:
                if n_synthetic > 0:
                    for i in range(n_synthetic):
                        synthetic = synthetic_generator.generate(len(pnl))
                        syn_series = pd.Series(synthetic, index=pnl.index)
                        label = f"synthetic_path_{i}"
                        results.append(
                            self._evaluate(syn_series, label, base_drawdown)
                        )
                        generated_paths[label] = [float(x) for x in synthetic]

                # One crash/recovery path if the generator supports it
                if hasattr(synthetic_generator, "sample_crash_recovery"):
                    crash_path = synthetic_generator.sample_crash_recovery(len(pnl))
                    crash_series = pd.Series(crash_path, index=pnl.index)
                    label = "synthetic_crash_recovery"
                    results.append(self._evaluate(crash_series, label, base_drawdown))
                    generated_paths[label] = [float(x) for x in crash_path]

                # Diffusion based scenarios
                if hasattr(synthetic_generator, "sample_crash"):
                    crash = synthetic_generator.sample_crash(len(pnl))
                    crash_series = pd.Series(crash, index=pnl.index)
                    label = "diffusion_crash"
                    results.append(self._evaluate(crash_series, label, base_drawdown))
                    generated_paths[label] = [float(x) for x in crash]
                if hasattr(synthetic_generator, "sample_liquidity_freeze"):
                    freeze = synthetic_generator.sample_liquidity_freeze(len(pnl))
                    freeze_series = pd.Series(freeze, index=pnl.index)
                    label = "diffusion_liquidity_freeze"
                    results.append(self._evaluate(freeze_series, label, base_drawdown))
                    generated_paths[label] = [float(x) for x in freeze]
                if hasattr(synthetic_generator, "sample_regime_flip"):
                    flip = synthetic_generator.sample_regime_flip(len(pnl))
                    flip_series = pd.Series(flip, index=pnl.index)
                    label = "diffusion_regime_flip"
                    results.append(self._evaluate(flip_series, label, base_drawdown))
                    generated_paths[label] = [float(x) for x in flip]

            for r in results:
                record_metric(
                    "stress_pnl", r.pnl, {"strategy": name, "scenario": r.scenario}
                )
                record_metric(
                    "stress_drawdown",
                    r.max_drawdown,
                    {"strategy": name, "scenario": r.scenario},
                )
                record_metric(
                    "stress_hedge_eff",
                    r.hedging_effectiveness,
                    {"strategy": name, "scenario": r.scenario},
                )

            all_results[name] = results
            self._save_report(name, results)
            meta: Dict[str, Any] = {
                "scenario_generator": type(scenario_generator).__name__
                if scenario_generator
                else None,
                "synthetic_generator": type(synthetic_generator).__name__
                if synthetic_generator
                else None,
                "n_synthetic": n_synthetic,
            }
            self._save_scenario_report(name, results, meta, generated_paths)
            self._apply_actions(results)
        return all_results

    def _evaluate(
        self, pnl: pd.Series, scenario: str, base_drawdown: Optional[float]
    ) -> ScenarioResult:
        max_dd = self._max_drawdown(pnl)
        recovery = self._recovery_time(pnl)
        liquidity = self._liquidity_impact(pnl)
        pnl_total = float(pnl.sum())
        hedging = 1.0
        if base_drawdown and base_drawdown != 0:
            hedging = 1.0 - (-max_dd) / base_drawdown
        action = "ok"
        if max_dd < -self.thresholds["max_drawdown"] or liquidity > self.thresholds["max_liquidity"]:
            action = "disable"
        elif max_dd < -self.thresholds["max_drawdown"] * 0.5:
            action = "adjust"
        return ScenarioResult(
            scenario=scenario,
            pnl=pnl_total,
            max_drawdown=-max_dd,
            recovery_days=recovery,
            liquidity_impact=liquidity,
            hedging_effectiveness=hedging,
            action=action,
        )

    def _save_report(self, name: str, results: Iterable[ScenarioResult]) -> None:
        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        data = [asdict(r) for r in results]
        with (self.report_dir / f"{name}_{ts}.json").open("w") as f:
            json.dump(data, f, indent=2)

    def _save_scenario_report(
        self,
        name: str,
        results: Iterable[ScenarioResult],
        metadata: Dict[str, Any],
        paths: Dict[str, List[float]],
    ) -> None:
        """Persist scenario metadata and outcomes for audit."""

        ts = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        payload = {
            "metadata": metadata,
            "results": [asdict(r) for r in results],
            "paths": paths,
        }
        with (self.scenario_report_dir / f"{name}_{ts}.json").open("w") as f:
            json.dump(payload, f, indent=2)

    def _apply_actions(self, results: Iterable[ScenarioResult]) -> None:
        actions = {r.action for r in results}
        if "disable" in actions:
            risk_manager.halt()
            raise RuntimeError("critical stress scenario breach")
        elif "adjust" in actions:
            risk_manager.max_drawdown = min(
                risk_manager.max_drawdown, self.thresholds["max_drawdown"] * 0.5
            )
        for r in results:
            if r.action != "ok":
                record_metric(
                    "stress_risk_alert",
                    r.max_drawdown,
                    {"scenario": r.scenario, "action": r.action},
                )

