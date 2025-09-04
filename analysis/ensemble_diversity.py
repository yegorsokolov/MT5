"""Tools to analyse model diversity via error correlations."""

from __future__ import annotations

from typing import Iterable, Mapping

import numpy as np
import pandas as pd


def error_correlation_matrix(
    predictions: Mapping[str, Iterable[float]],
    y_true: Iterable[float],
) -> pd.DataFrame:
    """Return pairwise correlations of model prediction errors.

    Parameters
    ----------
    predictions:
        Mapping of model name to an iterable of predicted values. Each iterable
        must be aligned with ``y_true``.
    y_true:
        Iterable of true target values.

    Returns
    -------
    pandas.DataFrame
        Symmetric matrix of Pearson correlations where ``(i, j)`` denotes the
        correlation between the errors of model ``i`` and model ``j``. Errors are
        defined as ``prediction - y_true``. ``NaN`` is returned for pairs with
        constant errors.
    """

    truth = np.asarray(y_true)
    data = {name: np.asarray(pred) - truth for name, pred in predictions.items()}
    df = pd.DataFrame(data)
    return df.corr()


__all__ = ["error_correlation_matrix"]
