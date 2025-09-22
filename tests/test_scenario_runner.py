import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from stress_tests.scenario_runner import StressScenarioRunner
from mt5.risk_manager import risk_manager


def test_stress_scenario_runner_creates_report(tmp_path):
    data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=10, freq="D"),
        "pnl": [0.1, -0.2, 0.05, -0.1, 0.05, -0.05, 0.1, 0.1, -0.2, 0.1],
    })
    strat_path = tmp_path / "strat.csv"
    data.to_csv(strat_path, index=False)

    thresholds = {"max_drawdown": 0.3, "max_liquidity": 1.0, "shock_size": 0.5}
    runner = StressScenarioRunner({"strat": strat_path}, thresholds)
    runner.report_dir = tmp_path / "reports"
    runner.report_dir.mkdir()

    risk_manager.reset()
    with pytest.raises(RuntimeError):
        runner.run()

    reports = list(runner.report_dir.glob("strat_*.json"))
    assert reports, "Report file was not created"
    report = json.loads(reports[0].read_text())
    assert report and isinstance(report, list)
    assert any(r["scenario"] == "synthetic" for r in report)
    assert risk_manager.metrics.trading_halted


def test_stress_runner_injects_synthetic_paths(tmp_path):
    data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "pnl": [0.1, 0.05, -0.02, 0.03, -0.01],
    })
    strat_path = tmp_path / "strat.csv"
    data.to_csv(strat_path, index=False)

    thresholds = {"max_drawdown": 0.3, "max_liquidity": 1.0, "shock_size": 0.2}
    runner = StressScenarioRunner({"strat": strat_path}, thresholds)
    runner.report_dir = tmp_path / "reports"
    runner.report_dir.mkdir()

    class DummyGen:
        def generate(self, n):
            return np.linspace(0, 0.1, n)

    risk_manager.reset()
    runner.run(synthetic_generator=DummyGen(), n_synthetic=2)

    reports = list(runner.report_dir.glob("strat_*.json"))
    report = json.loads(reports[0].read_text())
    assert sum(r["scenario"].startswith("synthetic_path") for r in report) == 2


def test_scenario_runner_stores_metadata(tmp_path):
    data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "pnl": [0.1, 0.05, -0.02, 0.03, -0.01],
    })
    strat_path = tmp_path / "strat.csv"
    data.to_csv(strat_path, index=False)

    thresholds = {"max_drawdown": 0.3, "max_liquidity": 1.0, "shock_size": 0.2}
    runner = StressScenarioRunner({"strat": strat_path}, thresholds)
    runner.report_dir = tmp_path / "reports"
    runner.report_dir.mkdir()
    runner.scenario_report_dir = tmp_path / "scenario_reports"
    runner.scenario_report_dir.mkdir()

    class DummyGen:
        def generate(self, n):
            return np.linspace(0, 0.1, n)

    risk_manager.reset()
    runner.run(synthetic_generator=DummyGen(), n_synthetic=1)

    files = list(runner.scenario_report_dir.glob("strat_*.json"))
    assert files, "scenario report missing"
    payload = json.loads(files[0].read_text())
    assert payload["metadata"]["synthetic_generator"] == "DummyGen"
    assert any(r["scenario"].startswith("synthetic_path") for r in payload["results"])


def test_diffusion_scenarios(tmp_path):
    data = pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=5, freq="D"),
        "pnl": [0.1, 0.05, -0.02, 0.03, -0.01],
    })
    strat_path = tmp_path / "strat.csv"
    data.to_csv(strat_path, index=False)

    thresholds = {"max_drawdown": 10.0, "max_liquidity": 10.0, "shock_size": 0.2}
    runner = StressScenarioRunner({"strat": strat_path}, thresholds)
    runner.report_dir = tmp_path / "reports"
    runner.report_dir.mkdir()

    class DummyDiffusion:
        def generate(self, n):
            return np.zeros(n)

        def sample_crash(self, n):
            arr = np.zeros(n)
            arr[0] = -1
            return arr

        def sample_liquidity_freeze(self, n):
            arr = np.linspace(0, 0.1, n)
            arr[:2] = 0
            return arr

        def sample_regime_flip(self, n):
            arr = np.ones(n)
            arr[n // 2 :] *= -1
            return arr

    risk_manager.reset()
    runner.run(synthetic_generator=DummyDiffusion())

    reports = list(runner.report_dir.glob("strat_*.json"))
    report = json.loads(reports[0].read_text())
    labels = {r["scenario"] for r in report}
    assert {"diffusion_crash", "diffusion_liquidity_freeze", "diffusion_regime_flip"}.issubset(labels)


