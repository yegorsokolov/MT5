from pathlib import Path
import sys
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
if "scipy" in sys.modules:
    del sys.modules["scipy"]
if "scipy.stats" in sys.modules:
    del sys.modules["scipy.stats"]
import scipy  # noqa: F401  # ensure real scipy is available
from analysis.evaluate import bootstrap_classification_metrics


def test_intervals_shrink_with_sample_size():
    rng = np.random.default_rng(0)
    y_small = rng.integers(0, 2, 50)
    p_small = rng.integers(0, 2, 50)
    m_small = bootstrap_classification_metrics(
        y_small, p_small, n_bootstrap=200, seed=1
    )

    rng = np.random.default_rng(1)
    y_large = rng.integers(0, 2, 500)
    p_large = rng.integers(0, 2, 500)
    m_large = bootstrap_classification_metrics(
        y_large, p_large, n_bootstrap=200, seed=1
    )

    for key in ["precision_ci", "recall_ci", "f1_ci"]:
        w_small = m_small[key][1] - m_small[key][0]
        w_large = m_large[key][1] - m_large[key][0]
        assert w_large < w_small
