import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from risk import risk_of_ruin
from risk_manager import RiskManager


def test_risk_of_ruin_zero_with_positive_returns():
    returns = pd.Series([0.01, 0.02, 0.03])
    assert risk_of_ruin(returns, 1000.0) == 0.0


def test_risk_of_ruin_high_when_ruin_possible():
    returns = pd.Series([-1.0, 0.1])
    prob = risk_of_ruin(returns, 1000.0)
    assert 0.7 < prob < 0.8


def test_risk_of_ruin_is_one_when_equity_zero():
    returns = pd.Series([0.01, -0.02])
    assert risk_of_ruin(returns, 0.0) == 1.0


def test_risk_manager_halts_on_high_ruin_risk():
    rm = RiskManager(
        max_drawdown=1e9, risk_of_ruin_threshold=0.5, initial_capital=1000.0
    )
    rm.update("bot", -1000.0)
    assert rm.status()["trading_halted"] is True


def test_risk_manager_downscales_on_ruin_warning():
    rm = RiskManager(
        max_drawdown=1e9,
        risk_of_ruin_threshold=0.9,
        risk_of_ruin_downscale=0.2,
        ruin_downscale_factor=0.1,
        initial_capital=1000.0,
    )
    rm.metrics.risk_of_ruin = 0.3
    size = rm.adjust_size("SYM", 100.0, pd.Timestamp("2023-01-01"), 1)
    assert size == 10.0


def test_risk_metrics_persisted():
    path = Path("reports/risk")
    for f in path.glob("*"):
        if f.name != ".gitkeep":
            f.unlink()
    rm = RiskManager(max_drawdown=1e9, initial_capital=1000.0)
    rm.update("bot", 10.0)
    assert (path / "latest.json").exists()
    for f in path.glob("*"):
        if f.name != ".gitkeep":
            f.unlink()
