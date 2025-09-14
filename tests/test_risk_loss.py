import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.risk_loss import (
    RiskBudget,
    cvar_penalty,
    max_drawdown_penalty,
    risk_penalty,
)


def test_cvar_penalty_increases_when_limit_tightens():
    returns = np.array([-0.2, 0.1, -0.05, 0.02], dtype=float)
    p_loose = cvar_penalty(returns, limit=0.2, level=0.5)
    p_tight = cvar_penalty(returns, limit=0.05, level=0.5)
    assert p_tight > p_loose


def test_max_drawdown_penalty_increases_when_limit_tightens():
    returns = np.array([-0.05, 0.1, -0.2, 0.05], dtype=float)
    p_loose = max_drawdown_penalty(returns, limit=0.3)
    p_tight = max_drawdown_penalty(returns, limit=0.1)
    assert p_tight > p_loose


def test_risk_budget_combined_penalty():
    returns = np.array([-0.1, 0.02, -0.15, 0.03], dtype=float)
    budget_loose = RiskBudget(max_leverage=1.0, max_drawdown=0.3, cvar_limit=0.2)
    budget_tight = RiskBudget(max_leverage=1.0, max_drawdown=0.1, cvar_limit=0.05)
    p_loose = risk_penalty(returns, budget_loose, level=0.5)
    p_tight = risk_penalty(returns, budget_tight, level=0.5)
    assert p_tight > p_loose
