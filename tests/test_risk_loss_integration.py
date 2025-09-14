import sys, os
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analysis.risk_loss import (
    RiskBudget,
    risk_penalty,
    cvar_penalty,
    max_drawdown_penalty,
)


def test_penalties_increase_when_limits_exceeded():
    """Penalties should grow as risk limits tighten."""

    returns = np.array([0.05, -0.2, 0.01, -0.15, 0.03], dtype=float)

    loose = RiskBudget(max_leverage=2.0, max_drawdown=0.5, cvar_limit=0.5)
    tight = RiskBudget(max_leverage=1.0, max_drawdown=0.1, cvar_limit=0.1)

    loose_pen = risk_penalty(returns, loose, level=0.5)
    tight_pen = risk_penalty(returns, tight, level=0.5)

    assert tight_pen > loose_pen


def test_individual_penalties_monotonic():
    returns = np.array([0.1, -0.4, 0.02, -0.3], dtype=float)

    cvar_loose = cvar_penalty(returns, limit=0.5, level=0.5)
    cvar_tight = cvar_penalty(returns, limit=0.1, level=0.5)
    assert cvar_tight > cvar_loose

    mdd_loose = max_drawdown_penalty(returns, limit=0.6)
    mdd_tight = max_drawdown_penalty(returns, limit=0.1)
    assert mdd_tight > mdd_loose

