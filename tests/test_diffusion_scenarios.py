import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.diffusion_scenarios import MultiAssetDiffusion


def test_multiasset_diffusion_samples():
    df = pd.DataFrame({"a": np.random.randn(20), "b": np.random.randn(20)})
    model = MultiAssetDiffusion(seq_len=5)
    model.fit(df, epochs=1)
    path = model.generate(10)
    assert len(path) == 10
    crash = model.sample_crash(10)
    assert crash.shape == (10,)
    freeze = model.sample_liquidity_freeze(10, freeze_days=2)
    assert np.allclose(freeze[:2], 0)
    flip = model.sample_regime_flip(10)
    assert np.sign(flip[0]) != np.sign(flip[-1])

