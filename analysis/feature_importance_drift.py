from __future__ import annotations

"""Detect drift in feature importances.

This module computes rolling feature importances using SHAP if available
(or sklearn's permutation importance as a fallback) and compares them to
baseline importances derived from training data.  Features whose relative
importance deviates beyond a configurable threshold are flagged and
reported.
"""

import json
import logging
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance

from analytics.metrics_store import record_metric
from monitor_drift import DRIFT_METRICS

try:  # pragma: no cover - optional dependency
    import shap  # type: ignore
except Exception:  # pragma: no cover - shap is optional
    shap = None

logger = logging.getLogger(__name__)

REPORT_DIR = Path("reports/feature_drift")


def _compute_importances(model: RandomForestRegressor, X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Return feature importances using SHAP if available.

    Falls back to permutation importance when SHAP is not installed or
    fails.  Returned series is indexed by feature name.
    """

    if shap is not None:  # pragma: no cover - heavy optional dependency
        try:
            explainer = shap.Explainer(model, X)
            values = explainer(X)
            importance = np.abs(values.values).mean(axis=0)
            return pd.Series(importance, index=X.columns)
        except Exception:
            logger.exception("SHAP importance computation failed, falling back")

    result = permutation_importance(model, X, y, n_repeats=5, random_state=0)
    return pd.Series(result.importances_mean, index=X.columns)


def analyze(
    window: int = 1000,
    threshold: float = 0.5,
    baseline_file: Path | str = REPORT_DIR / "baseline.json",
) -> Dict[str, float]:
    """Compute rolling importances and flag drifts.

    Parameters
    ----------
    window:
        Number of most recent rows to use when computing importances.
    threshold:
        Relative change threshold for flagging feature drift.  For example,
        ``0.5`` flags features whose importance changed by more than 50%.
    baseline_file:
        JSON file containing training baseline importances as
        ``{feature: importance}``.

    Returns
    -------
    Dict[str, float]
        Mapping of features to relative importance change for those that
        exceeded ``threshold``.
    """

    baseline_path = Path(baseline_file)
    if not (baseline_path.exists() and DRIFT_METRICS.exists()):
        return {}

    try:
        baseline = pd.Series(json.loads(baseline_path.read_text()))
    except Exception:
        logger.exception("Failed loading baseline importances from %s", baseline_path)
        return {}

    try:
        df = pd.read_parquet(DRIFT_METRICS)
    except Exception:
        logger.exception("Failed loading feature data from %s", DRIFT_METRICS)
        return {}

    if df.empty or "prediction" not in df.columns:
        return {}

    X = df.drop(columns=["prediction"]).tail(window)
    y = df["prediction"].iloc[-len(X):]

    model = RandomForestRegressor(n_estimators=100, random_state=0)
    try:
        model.fit(X, y)
    except Exception:
        logger.exception("Failed fitting surrogate model for importances")
        return {}

    current = _compute_importances(model, X, y)
    rel_change = (current - baseline) / baseline.replace(0, np.nan)
    rel_change = rel_change.replace([np.inf, -np.inf], np.nan).dropna()
    flagged = rel_change[rel_change.abs() > threshold].to_dict()

    report = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "current": current.to_dict(),
        "baseline": baseline.to_dict(),
        "relative_change": rel_change.to_dict(),
        "flagged": flagged,
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out_file = REPORT_DIR / f"{pd.Timestamp.utcnow():%Y-%m-%d}.json"
    out_file.write_text(json.dumps(report, indent=2))
    (REPORT_DIR / "latest.json").write_text(json.dumps(report, indent=2))

    for feat, change in flagged.items():
        try:  # pragma: no cover - metrics store may be stubbed
            record_metric("feature_importance_drift", float(change), tags={"feature": feat})
        except Exception:
            logger.exception("Failed recording metric for feature %s", feat)

    return flagged


__all__ = ["analyze"]
