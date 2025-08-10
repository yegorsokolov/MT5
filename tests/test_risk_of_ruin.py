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


def test_risk_manager_halts_on_high_ruin_risk():
    rm = RiskManager(
        max_drawdown=1e9, risk_of_ruin_threshold=0.5, initial_capital=1000.0
    )
    rm.update("bot", -1000.0)
    assert rm.status()["trading_halted"] is True
