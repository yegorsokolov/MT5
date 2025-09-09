from __future__ import annotations

import pandas as pd

from analysis.regime_hmm import fit_regime_hmm


def compute(df: pd.DataFrame) -> pd.DataFrame:
    """Append discrete market regime labels using an HMM."""
    return fit_regime_hmm(df)


__all__ = ["compute"]
