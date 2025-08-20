import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis import rationale_scorer


def test_rationale_scoring(tmp_path):
    trades = pd.DataFrame(
        {
            "decision_id": [1, 2, 3, 4],
            "algorithm": ["algo1", "algo1", "algo2", "algo2"],
            "reason": ["trend", "trend", "mean_reversion", "trend"],
            "pnl": [1.0, -0.5, 0.7, -1.2],
            "feature_importance": [
                {"f1": 0.8, "f2": 0.2},
                {"f1": 0.7, "f2": 0.3},
                {"f1": 0.2, "f2": 0.8},
                {"f1": 0.1, "f2": 0.9},
            ],
        }
    )
    trade_path = tmp_path / "trade_history.parquet"
    trades.to_parquet(trade_path)
    report_dir = tmp_path / "reports"

    scores = rationale_scorer.score_rationales(trade_path, report_dir)

    acc = pd.read_parquet(report_dir / "reason_accuracy.parquet")
    win = pd.read_parquet(report_dir / "algorithm_win_rates.parquet")
    drift = pd.read_parquet(report_dir / "feature_importance_drift.parquet")

    assert abs(acc.loc["trend", "accuracy"] - (1 / 3)) < 1e-6
    assert acc.loc["mean_reversion", "accuracy"] == 1.0
    assert win.loc["algo1", "win_rate"] == 0.5
    assert win.loc["algo2", "win_rate"] == 0.5
    assert drift.loc["f1", "importance"] == 0.45
    assert drift.loc["f2", "importance"] == 0.55

    # Return values from function should match files
    assert "reason_accuracy" in scores and not scores["reason_accuracy"].empty
