import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategies.baseline import BaselineStrategy  # noqa: E402


def test_kalman_smoothing_reduces_whipsaws():
    np.random.seed(0)
    n = 200
    base = np.cumsum(np.random.normal(0, 0.1, n)) + 100
    noisy = base + np.random.normal(0, 0.5, n)

    raw = BaselineStrategy(short_window=3, long_window=5, rsi_window=3, atr_window=3)
    smooth = BaselineStrategy(
        short_window=3,
        long_window=5,
        rsi_window=3,
        atr_window=3,
        use_kalman_smoothing=True,
    )

    trades_raw = sum(1 for p in noisy if raw.update(p))
    trades_smooth = sum(1 for p in noisy if smooth.update(p))

    assert trades_smooth < trades_raw

