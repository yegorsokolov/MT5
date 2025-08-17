import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import pytest

from risk.budget_allocator import BudgetAllocator
from risk_manager import RiskManager

if not hasattr(pytest.MonkeyPatch, "patch"):
    from unittest.mock import patch as _mpatch
    import types
    import sys

    def _ensure_utils_secret() -> None:
        if "utils" not in sys.modules:
            sys.modules["utils"] = types.ModuleType("utils")
        utils_mod = sys.modules["utils"]
        if "utils.secret_manager" not in sys.modules:
            sm_mod = types.ModuleType("utils.secret_manager")

            class SecretManager:
                def get_secret(self, *_, **__):
                    return None

            sm_mod.SecretManager = SecretManager
            sys.modules["utils.secret_manager"] = sm_mod
            setattr(utils_mod, "secret_manager", sm_mod)

    def _patch(self, target, *args, **kwargs):
        if target.startswith("utils.secret_manager"):
            _ensure_utils_secret()
        p = _mpatch(target, *args, **kwargs)
        started = p.start()
        try:
            self.addfinalizer(p.stop)  # type: ignore[attr-defined]
        except Exception:
            pass
        return started

    pytest.MonkeyPatch.patch = _patch  # type: ignore[attr-defined]


def test_allocator_prefers_lower_risk():
    allocator = BudgetAllocator(capital=100)
    low = pd.Series([0.01] * 100)
    high = pd.Series([0.02, -0.02] * 50)
    budgets = allocator.allocate({"low": low, "high": high})
    assert pytest.approx(sum(budgets.values())) == 100
    assert budgets["low"] > budgets["high"]


def test_rebalance_adjusts_sizes():
    rm = RiskManager(max_drawdown=100, initial_capital=100)
    for pnl in [1, -1, 1, -1]:
        rm.update("low", pnl)
    for pnl in [5, -5, 5, -5]:
        rm.update("high", pnl)
    rm.rebalance_budgets()
    base = 10
    low_size = rm.adjust_position_size("low", base)
    high_size = rm.adjust_position_size("high", base)
    assert low_size + high_size == pytest.approx(base)
    assert low_size > high_size


def test_rebalance_changes_with_new_returns():
    rm = RiskManager(max_drawdown=100, initial_capital=100)
    for pnl in [1, -1, 1, -1]:
        rm.update("s1", pnl)
        rm.update("s2", pnl)
    rm.rebalance_budgets()
    before = rm.budget_allocator.budgets["s1"]
    for pnl in [10, -10, 10, -10]:
        rm.update("s1", pnl)
    rm.rebalance_budgets()
    after = rm.budget_allocator.budgets["s1"]
    assert after < before
