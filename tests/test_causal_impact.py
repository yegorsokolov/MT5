import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis.causal_impact import estimate_causal_impact


def test_dml_recovers_effect(tmp_path):
    rng = np.random.default_rng(0)
    n = 400
    X = rng.normal(size=(n, 2))
    t = (X[:, 0] + rng.normal(scale=0.1, size=n) > 0).astype(float)
    y = 2 * t + X[:, 0] + rng.normal(scale=0.1, size=n)
    df = pd.DataFrame({
        "f1": X[:, 0],
        "f2": X[:, 1],
        "executed": t,
        "pnl": y,
    })
    ate = estimate_causal_impact(df, report_dir=tmp_path)
    assert ate is not None
    assert abs(ate - 2.0) < 0.3
    files = list(tmp_path.glob("impact_*.json"))
    assert files, "report not written"
