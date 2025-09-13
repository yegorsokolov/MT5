import numpy as np
import pandas as pd
import pytest

from evaluation.strategy_benchmark import run_benchmark


class DummyStrategy:
    def backtest(self, data: pd.DataFrame, risk_profile: float) -> pd.DataFrame:
        df = data.copy()
        df["returns"] = df["returns"] * risk_profile
        df["position"] = df["position"] * risk_profile
        return df


def test_run_benchmark_metrics():
    data = pd.DataFrame(
        {
            "returns": [0.1, -0.1, 0.05, -0.02],
            "position": [0, 1, -1, 0],
        }
    )
    datasets = {"ds": data}
    risk_profiles = [1.0]

    results = run_benchmark(DummyStrategy(), datasets, risk_profiles)
    assert list(results.columns) == [
        "dataset",
        "risk_profile",
        "sharpe",
        "cvar",
        "turnover",
    ]

    returns = data["returns"]
    expected_sharpe = np.sqrt(252) * returns.mean() / returns.std(ddof=0)
    var = returns.quantile(0.05)
    expected_cvar = returns[returns <= var].mean()
    expected_turnover = data["position"].diff().abs().sum()

    row = results.iloc[0]
    assert row["dataset"] == "ds"
    assert row["risk_profile"] == "1.0"
    assert row["sharpe"] == pytest.approx(expected_sharpe)
    assert row["cvar"] == pytest.approx(expected_cvar)
    assert row["turnover"] == pytest.approx(expected_turnover)
