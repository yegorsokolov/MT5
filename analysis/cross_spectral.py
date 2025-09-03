from __future__ import annotations

"""Cross-spectral relationship features.

This module provides a :func:`compute` function that derives rolling
coherence metrics between pairs of symbols.  The current implementation
uses a simple FFT based magnitude-squared coherence which provides a
proxy for wavelet coherence but without the heavy dependency footprint.
"""

import numpy as np
import pandas as pd
from itertools import combinations

# ``ResourceCapabilities`` is only needed for gating based on hardware.  The
# analysis module can fall back to a lightweight local definition when the
# full utilities package (and its optional dependencies) is unavailable.
try:  # pragma: no cover - fallback for minimal test environments
    from utils.resource_monitor import ResourceCapabilities
except Exception:  # pragma: no cover - utils may not be importable in tests
    from dataclasses import dataclass

    @dataclass
    class ResourceCapabilities:  # type: ignore
        cpus: int = 1
        memory_gb: float = 0.0
        has_gpu: bool = False
        gpu_count: int = 0

# Minimum resources required to enable this heavy computation
REQUIREMENTS = ResourceCapabilities(cpus=4, memory_gb=8.0, has_gpu=False, gpu_count=0)


def _coherence_fft(x: np.ndarray, y: np.ndarray) -> float:
    """Return the average magnitude-squared coherence between ``x`` and ``y``."""
    fx = np.fft.rfft(x - np.mean(x))
    fy = np.fft.rfft(y - np.mean(y))
    num = np.abs(np.vdot(fx, fy)) ** 2
    denom = np.vdot(fx, fx) * np.vdot(fy, fy)
    if denom == 0:
        return 0.0
    return float((num / denom).real)


def _rolling_coherence(x: pd.Series, y: pd.Series, window: int) -> pd.Series:
    values = np.full(len(x), np.nan, dtype=float)
    arr_x = x.to_numpy()
    arr_y = y.to_numpy()
    for i in range(window - 1, len(x)):
        values[i] = _coherence_fft(
            arr_x[i - window + 1 : i + 1], arr_y[i - window + 1 : i + 1]
        )
    return pd.Series(values, index=x.index)


def compute(df: pd.DataFrame, window: int = 64) -> pd.DataFrame:
    """Compute rolling coherence features between symbol pairs.

    Parameters
    ----------
    df:
        Input dataframe containing ``Timestamp``, ``Symbol`` and ``Close``
        columns.
    window:
        Number of observations per rolling coherence window.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with ``coh_{other}`` columns for each symbol
        pair.
    """

    required = {"Symbol", "Timestamp", "Close"}
    if not required.issubset(df.columns) or df["Symbol"].nunique() < 2:
        return df

    prices = df.pivot(index="Timestamp", columns="Symbol", values="Close").sort_index()
    returns = prices.pct_change().fillna(0.0)

    features: list[pd.DataFrame] = []
    for s1, s2 in combinations(returns.columns, 2):
        coh = _rolling_coherence(returns[s1], returns[s2], window)
        f1 = pd.DataFrame(
            {"Timestamp": coh.index, "Symbol": s1, f"coh_{s2}": coh.values}
        )
        f2 = pd.DataFrame(
            {"Timestamp": coh.index, "Symbol": s2, f"coh_{s1}": coh.values}
        )
        features.extend([f1, f2])

    if not features:
        return df

    feat_df = pd.concat(features, ignore_index=True)
    return df.merge(feat_df, on=["Timestamp", "Symbol"], how="left")


__all__ = ["compute", "REQUIREMENTS"]
