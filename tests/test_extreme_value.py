import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.extreme_value import estimate_tail_probability


def test_estimate_tail_probability_fat_tailed():
    rng = np.random.default_rng(0)
    returns = rng.standard_t(df=3, size=5000) * 0.1
    threshold = -0.3
    prob, params = estimate_tail_probability(returns, threshold)
    empirical = np.mean(returns <= threshold)
    assert params.n_exceed > 0
    assert abs(prob - empirical) < 0.02
