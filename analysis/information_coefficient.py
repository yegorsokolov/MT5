"""Compute information coefficient (rank correlation)."""
from __future__ import annotations

from typing import Iterable
import numpy as np
import pandas as pd


def information_coefficient(preds: Iterable[float], returns: Iterable[float]) -> float:
    """Return Spearman rank correlation between predictions and returns.

    Parameters
    ----------
    preds:
        Iterable of model predictions.
    returns:
        Iterable of realized returns or outcomes.

    Returns
    -------
    float
        Spearman rank correlation. ``NaN`` if not computable.
    """

    s = pd.Series(preds)
    r = pd.Series(returns)
    df = pd.DataFrame({"pred": s, "ret": r}).dropna()
    if df.empty or df["pred"].nunique() < 2 or df["ret"].nunique() < 2:
        return float("nan")
    return float(df["pred"].corr(df["ret"], method="spearman"))
