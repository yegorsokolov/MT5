"""Regression tests for risk objective penalties."""

from __future__ import annotations

import numpy as np

from pathlib import Path
import sys

# Ensure repository root is importable when running tests standalone
sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.risk_objectives import (
    cvar_penalty,
    max_drawdown_penalty,
    risk_penalty,
)


def test_cvar_penalty_increases_loss():
    """Penalty should grow once CVaR exceeds the user limit."""

    base_loss = 1.0
    safe_returns = np.array([0.02, 0.01, -0.01, 0.0])
    risky_returns = np.array([-0.3, -0.2, 0.05, 0.1])

    safe_loss = base_loss + cvar_penalty(safe_returns, target=0.05, level=0.5)
    risky_loss = base_loss + cvar_penalty(risky_returns, target=0.05, level=0.5)

    assert risky_loss > safe_loss


def test_drawdown_penalty_increases_loss():
    """Penalty should grow once max drawdown exceeds the user limit."""

    base_loss = 1.0
    safe_returns = np.array([0.1, -0.05, 0.07, 0.03])
    risky_returns = np.array([0.1, -0.4, 0.05, -0.2])

    safe_loss = base_loss + max_drawdown_penalty(safe_returns, target=0.3)
    risky_loss = base_loss + max_drawdown_penalty(risky_returns, target=0.3)

    assert risky_loss > safe_loss


def test_combined_penalty_adds_components():
    returns = np.array([-0.3, -0.2, 0.05, 0.1])
    p1 = risk_penalty(returns, cvar_target=0.5, level=0.5)
    p2 = risk_penalty(returns, mdd_target=0.1)
    combined = risk_penalty(returns, cvar_target=0.5, mdd_target=0.1, level=0.5)
    assert np.isclose(p1 + p2, combined)
