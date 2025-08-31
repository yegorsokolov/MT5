from __future__ import annotations

import sys
import importlib
from pathlib import Path

# Ensure real pandas is used rather than the lightweight test stub
if "pandas" in sys.modules:
    del sys.modules["pandas"]
pd = importlib.import_module("pandas")
np = importlib.import_module("numpy")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.factor_model import FactorModel
import risk_manager as rm_module
rm_module.risk_of_ruin = lambda *a, **k: 0.0
from risk_manager import RiskManager


def max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    return float(drawdown.min())


def test_factor_aware_weights_reduce_drawdown():
    np.random.seed(42)
    n = 20
    factors = pd.DataFrame({
        "F1": np.random.randn(n),
        "F2": np.random.randn(n),
    })
    exposures_true = pd.DataFrame(
        {
            "asset1": [1.0, 0.5],
            "asset2": [1.0, -0.3],
            "asset3": [-0.5, 1.0],
        },
        index=["F1", "F2"],
    ).T
    noise = pd.DataFrame(np.random.randn(n, 3) * 0.05, columns=exposures_true.index)
    returns = factors.values @ exposures_true.T.values + noise.values
    returns_df = pd.DataFrame(returns, columns=exposures_true.index)

    fm = FactorModel(n_factors=2).fit(returns_df)
    exposures_est = fm.get_exposures()

    rm = RiskManager(max_drawdown=1e9, var_window=n, initial_capital=1.0)
    for i in range(n):
        for asset in returns_df.columns:
            rm.update(asset, pnl=float(returns_df.iloc[i][asset]), check_hedge=False)

    # Naive weights ignoring correlation
    budgets_naive = rm.rebalance_budgets()
    w_naive = np.array(list(budgets_naive.values()))
    if w_naive.sum() == 0:
        w_naive = np.array([1.0 / len(budgets_naive)] * len(budgets_naive))
    else:
        w_naive = w_naive / w_naive.sum()
    port_naive = returns_df[list(budgets_naive.keys())] @ w_naive
    dd_naive = max_drawdown(port_naive)

    # Factor-aware weights using estimated exposures
    exposures_dict = {a: exposures_est.loc[a] for a in exposures_est.index}
    budgets_factor = rm.rebalance_budgets(factor_exposures=exposures_dict)
    w_factor = np.array(list(budgets_factor.values()))
    if w_factor.sum() == 0:
        w_factor = np.array([1.0 / len(budgets_factor)] * len(budgets_factor))
    else:
        w_factor = w_factor / w_factor.sum()
    port_factor = returns_df[list(budgets_factor.keys())] @ w_factor
    dd_factor = max_drawdown(port_factor)

    assert abs(dd_factor) < abs(dd_naive)
