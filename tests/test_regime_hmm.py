from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.regime_hmm import fit_regime_hmm
from strategies.baseline import BaselineStrategy, IndicatorBundle


def test_hmm_detects_regime_shift():
    pytest.importorskip("hmmlearn.hmm")
    rng = np.random.default_rng(0)
    n = 50
    ret1 = rng.normal(0.01, 0.001, size=n)
    vol1 = rng.normal(0.01, 0.001, size=n)
    ret2 = rng.normal(-0.01, 0.001, size=n)
    vol2 = rng.normal(0.05, 0.001, size=n)
    df = pd.DataFrame(
        {
            "return": np.concatenate([ret1, ret2]),
            "volatility_30": np.concatenate([vol1, vol2]),
        }
    )
    labeled = fit_regime_hmm(df, n_states=2)
    regimes = labeled["regime"].to_numpy()
    first = regimes[:n]
    second = regimes[n:]
    assert np.bincount(first).argmax() != np.bincount(second).argmax()


def test_baseline_regime_gating():
    strat = BaselineStrategy(short_window=2, long_window=3, atr_window=2, long_regimes={1}, short_regimes=set())
    strat.update(1, IndicatorBundle(regime=1))
    strat.update(2, IndicatorBundle(regime=1))
    sig = strat.update(3, IndicatorBundle(regime=1))
    assert sig == 1 and strat.position == 1
    # Regime disallows long
    exit_sig = strat.update(4, IndicatorBundle(regime=0))
    assert exit_sig == -1 and strat.position == 0
    # Attempt long in disallowed regime
    strat2 = BaselineStrategy(short_window=2, long_window=3, atr_window=2, long_regimes={1}, short_regimes=set())
    strat2.update(1, IndicatorBundle(regime=0))
    strat2.update(2, IndicatorBundle(regime=0))
    sig2 = strat2.update(3, IndicatorBundle(regime=0))
    assert sig2 == 0 and strat2.position == 0
