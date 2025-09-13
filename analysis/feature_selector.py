"""Feature selection utilities."""

from __future__ import annotations

from typing import Iterable, Sequence

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:  # pragma: no cover - optional dependency
    from sklearn.linear_model import LogisticRegression
except Exception:  # pragma: no cover
    LogisticRegression = None  # type: ignore

try:  # pragma: no cover - shap is optional
    import shap  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    shap = None


def select_features(
    df: pd.DataFrame,
    target: Sequence[float] | pd.Series,
    top_k: int | None = None,
    corr_threshold: float | None = 0.95,
) -> list[str]:
    """Select informative features using L1 regularization or SHAP.

    Parameters
    ----------
    df:
        Candidate feature dataframe.
    target:
        Target values aligned with ``df``.
    top_k:
        If provided and SHAP is available, keep only the ``top_k`` features
        ranked by mean absolute SHAP value.  Otherwise, features with non-zero
        L1 coefficients are returned.
    corr_threshold:
        If provided, compute the pairwise correlation matrix and drop one of
        any feature pairs whose absolute correlation exceeds this threshold
        before fitting the model.  ``None`` disables this filtering.
    """

    X = df.copy()
    y = pd.Series(target, index=df.index, name="target").copy()
    data = pd.concat([X, y], axis=1).dropna()
    X = data[df.columns]
    y = data["target"]

    if corr_threshold is not None and len(X.columns) > 1:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        to_drop = [col for col in upper.columns if (upper[col] > corr_threshold).any()]
        if to_drop:
            X = X.drop(columns=to_drop)

    if y.nunique() < 2:
        return list(X.columns)

    if LogisticRegression is not None:
        model = LogisticRegression(
            penalty="l1",
            solver="liblinear",
            max_iter=200,
            random_state=0,
        )
        model.fit(X, y)
        coef = np.abs(model.coef_)[0]
        importance = pd.Series(coef, index=X.columns)
    else:  # pragma: no cover - correlation-based fallback
        corr = {}
        for col in X.columns:
            if X[col].std() == 0 or y.std() == 0:
                corr[col] = 0.0
            else:
                corr[col] = float(np.corrcoef(X[col], y)[0, 1])
        importance = pd.Series(np.abs(pd.Series(corr)), index=X.columns)

    # If shap is available and top_k requested, use shap-based ranking
    if shap is not None and top_k is not None:
        try:  # pragma: no cover - heavy but optional
            explainer = shap.LinearExplainer(model, X, feature_perturbation="interventional")
            shap_values = explainer.shap_values(X)
            importance = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns)
        except Exception:  # pragma: no cover - fall back to coef based selection
            pass

    if top_k is not None:
        selected = importance.nlargest(top_k)
    else:
        thresh = 0.01 if LogisticRegression is not None else 0.1
        selected = importance[importance > thresh]

    if selected.empty:  # fallback to all features
        selected = importance

    return selected.index.tolist()


def _feature_version(features: Iterable[str]) -> str:
    """Compute a short hash for a feature list."""
    text = ",".join(sorted(features))
    return hashlib.sha1(text.encode()).hexdigest()[:8]


def save_feature_set(features: Iterable[str], path: Path) -> str:
    """Persist selected features with a version hash.

    Parameters
    ----------
    features:
        Iterable of selected feature names.
    path:
        Destination path where the feature list is written as JSON.

    Returns
    -------
    str
        Version hash computed from the feature list.
    """

    feats = list(features)
    version = _feature_version(feats)
    payload = {"version": version, "features": feats}
    path.write_text(json.dumps(payload))
    return version


def load_feature_set(path: Path) -> tuple[list[str], str | None]:
    """Load a persisted feature set.

    Parameters
    ----------
    path:
        Path to the JSON file produced by :func:`save_feature_set`.

    Returns
    -------
    list[str], str | None
        Tuple of feature list and version hash (``None`` if missing).
    """

    if not path.exists():
        return [], None
    data = json.loads(path.read_text())
    return data.get("features", []), data.get("version")
