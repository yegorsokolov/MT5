import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from backtesting import walk_forward


def test_rolling_windows_and_aggregate():
    data = pd.DataFrame({
        "return": [0.01, -0.02, 0.015, 0.02, -0.01, 0.03, 0.02, -0.01, 0.015, 0.025, -0.005, 0.02]
    })
    windows = walk_forward.rolling_windows(data, train_size=5, val_size=2, step=2)
    assert len(windows) == 3
    # Verify each window has expected lengths
    for train, val in windows:
        assert len(train) == 5
        assert len(val) == 2

    stats = walk_forward.aggregate_metrics(data, train_size=5, val_size=2, step=2)
    assert set(stats) == {"avg_sharpe", "worst_drawdown"}
    # Manual computation for comparison
    manual = []
    for _, val in windows:
        r = val["return"]
        sharpe = r.mean() / r.std(ddof=0)
        cum = (1 + r).cumprod()
        dd = (cum / cum.cummax() - 1).min()
        manual.append((sharpe, dd))
    avg_sharpe = sum(s for s, _ in manual) / len(manual)
    worst_dd = min(d for _, d in manual)
    assert stats["avg_sharpe"] == pytest.approx(avg_sharpe)
    assert stats["worst_drawdown"] == pytest.approx(worst_dd)
