"""Causal impact estimation utilities.

This module implements a lightweight double machine learning (DML)
approach to estimate treatment effects from historical trade data.
The analysis expects a data set containing a binary treatment column,

``treatment_col``
    Indicator whether the trade was executed (1) or represents a
    counterfactual (0).

``outcome_col``
    Observed outcome for the trade such as realised PnL or return.

All other columns are treated as features used for the nuisance models
within the DML procedure.  Results are written to
``reports/causal_impact`` for later inspection.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports/causal_impact")


def _double_ml_effect(
    X: np.ndarray,
    treatment: np.ndarray,
    outcome: np.ndarray,
    *,
    n_splits: int = 2,
    random_state: int = 0,
) -> float:
    """Estimate average treatment effect using double machine learning.

    Parameters
    ----------
    X:
        Feature matrix.
    treatment:
        Binary treatment indicator array.
    outcome:
        Outcome array.
    n_splits:
        Number of cross–fitting splits.
    random_state:
        Seed for the cross–fitting shuffler.
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    t_hat = np.zeros_like(treatment, dtype=float)
    y_hat = np.zeros_like(outcome, dtype=float)

    for train, test in kf.split(X):
        t_model = LinearRegression().fit(X[train], treatment[train])
        y_model = LinearRegression().fit(X[train], outcome[train])
        t_hat[test] = t_model.predict(X[test])
        y_hat[test] = y_model.predict(X[test])

    resid_t = treatment - t_hat
    resid_y = outcome - y_hat
    denom = np.dot(resid_t, resid_t)
    if denom == 0:  # pragma: no cover - degenerate case
        return float("nan")
    ate = float(np.dot(resid_t, resid_y) / denom)
    return ate


def estimate_causal_impact(
    df: pd.DataFrame,
    *,
    treatment_col: str = "executed",
    outcome_col: str = "pnl",
    feature_cols: Iterable[str] | None = None,
    report_dir: Path = REPORT_DIR,
) -> float | None:
    """Compute and log the causal impact of executed trades.

    Parameters
    ----------
    df:
        DataFrame containing trade observations and counterfactuals.
    treatment_col:
        Name of the binary treatment column.
    outcome_col:
        Name of the outcome column.
    feature_cols:
        Optional iterable of feature column names.  If not provided all
        columns except ``treatment_col`` and ``outcome_col`` are used.
    report_dir:
        Directory where the analysis result is written.  The file name
        contains the current date for traceability.

    Returns
    -------
    float | None
        Estimated average treatment effect or ``None`` if the required
        columns are missing.
    """

    if treatment_col not in df.columns or outcome_col not in df.columns:
        logger.info(
            "Causal impact requires '%s' and '%s' columns", treatment_col, outcome_col
        )
        return None

    feature_cols = list(
        feature_cols
        or [c for c in df.columns if c not in {treatment_col, outcome_col}]
    )
    if not feature_cols:
        logger.info("No feature columns available for causal impact analysis")
        return None

    X = df[feature_cols].to_numpy()
    treatment = df[treatment_col].to_numpy().astype(float)
    outcome = df[outcome_col].to_numpy().astype(float)

    ate = _double_ml_effect(X, treatment, outcome)

    report_dir.mkdir(parents=True, exist_ok=True)
    out_file = report_dir / f"impact_{pd.Timestamp.utcnow():%Y-%m-%d}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"ate": ate}, f)
    logger.info("Wrote causal impact report %s", out_file)
    return ate


__all__ = ["estimate_causal_impact"]
