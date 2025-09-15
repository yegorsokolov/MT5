import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
from analysis import cross_spectral


def _make_df(x: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    t = pd.date_range("2020-01-01", periods=len(x), freq="D")
    return pd.DataFrame(
        {
            "Timestamp": list(t) + list(t),
            "Symbol": ["A"] * len(x) + ["B"] * len(y),
            "Close": np.concatenate([x, y]),
        }
    )


def test_high_coherence_for_correlated_series() -> None:
    n = 128
    t = np.linspace(0, 10, n)
    x = np.sin(t)
    y = x + 0.01 * np.random.randn(n)
    df = _make_df(x, y)
    out = cross_spectral.compute(df, window=32)
    coh = out[out["Symbol"] == "A"]["coh_B"].dropna()
    assert coh.mean() > 0.8


def test_low_coherence_for_uncorrelated_series() -> None:
    n = 128
    t = np.linspace(0, 10, n)
    x = np.sin(t)
    y = np.random.randn(n)
    df = _make_df(x, y)
    out = cross_spectral.compute(df, window=32)
    coh = out[out["Symbol"] == "A"]["coh_B"].dropna()
    assert coh.mean() < 0.5


def test_multi_asset_coherence_ranks_pairs() -> None:
    n = 128
    t = np.linspace(0, 10, n)
    base = np.sin(t)
    correlated = base + 0.01 * np.random.randn(n)
    random = np.random.randn(n)
    t_index = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame(
        {
            "Timestamp": list(t_index) * 3,
            "Symbol": ["A"] * n + ["B"] * n + ["C"] * n,
            "Close": np.concatenate([base, correlated, random]),
        }
    )
    out = cross_spectral.compute(df, window=32)
    coh_ab = out[out["Symbol"] == "A"]["coh_B"].dropna().mean()
    coh_ac = out[out["Symbol"] == "A"]["coh_C"].dropna().mean()
    assert coh_ab > coh_ac
