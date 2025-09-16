"""Compute information coefficient (rank correlation)."""
from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

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


def information_coefficient_series(
    predictions: Mapping[str, Iterable[float]],
    returns: Iterable[float],
) -> pd.Series:
    """Return information coefficients for multiple prediction series.

    ``predictions`` is converted into a :class:`pandas.DataFrame` and aligned
    with ``returns``; coefficients are computed column-wise. Any rows containing
    missing values in either the predictions or the returns are discarded.
    """

    if not predictions:
        return pd.Series(dtype=float)

    truth = pd.Series(returns, name="ret")
    preds_df = pd.DataFrame(predictions)
    aligned = pd.concat([preds_df, truth], axis=1).dropna()
    if aligned.empty:
        return pd.Series({name: float("nan") for name in preds_df.columns})
    result = {
        name: information_coefficient(aligned[name], aligned["ret"])
        for name in preds_df.columns
    }
    return pd.Series(result, dtype=float)


def grouped_information_coefficient(
    predictions: Mapping[str, Iterable[float]],
    returns: Iterable[float],
    regimes: Sequence[float | int | str],
) -> pd.DataFrame:
    """Return information coefficients grouped by ``regimes``.

    Parameters
    ----------
    predictions:
        Mapping of model name to predicted values aligned with ``returns``.
    returns:
        Iterable of realized outcomes.
    regimes:
        Grouping labels of the same length as ``returns`` used to compute
        regime-specific coefficients.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by unique regime label with one column per model
        containing the respective information coefficient. ``NaN`` values are
        returned for regimes with insufficient data.
    """

    if not predictions:
        return pd.DataFrame()

    df = pd.DataFrame(predictions)
    df["ret"] = list(returns)
    df["regime"] = list(regimes)
    df = df.dropna(subset=["ret", "regime"])
    if df.empty:
        return pd.DataFrame(columns=df.columns[:-2], dtype=float)

    records: dict[float | int | str, dict[str, float]] = {}
    for regime, group in df.groupby("regime"):
        res = {
            name: information_coefficient(group[name], group["ret"])
            for name in predictions
        }
        records[regime] = res
    result = pd.DataFrame.from_dict(records, orient="index")
    # Ensure deterministic ordering for testing
    return result.sort_index()


__all__ = [
    "information_coefficient",
    "information_coefficient_series",
    "grouped_information_coefficient",
]
