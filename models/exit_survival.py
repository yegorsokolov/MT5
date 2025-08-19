from __future__ import annotations

"""Simple survival model estimating trade continuation probability.

This module provides :class:`ExitSurvivalModel` which fits a logistic
regression model on historical trade data to estimate the probability that a
trade remains viable given its current age and regime features.  The model is
intentionally lightweight so it can run during real-time trade management.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class ExitSurvivalModel:
    """Estimate probability of a trade continuing to be profitable.

    The model is trained on a dataframe containing trade attributes and a
    ``survived`` column indicating whether the trade ultimately reached its
    profit target (1) or not (0).
    """

    estimator: Optional[LogisticRegression] = None

    def __post_init__(self) -> None:  # pragma: no cover - simple init
        self.model = self.estimator or LogisticRegression()
        self.features: list[str] | None = None

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, target: str = "survived") -> "ExitSurvivalModel":
        """Fit the underlying estimator."""

        self.features = [c for c in df.columns if c != target]
        X = df[self.features]
        y = df[target]
        self.model.fit(X, y)
        return self

    # ------------------------------------------------------------------
    def predict_survival(self, features: Dict[str, float]) -> float:
        """Return probability the trade continues to have profit potential."""

        if self.features is None:
            raise ValueError("Model has not been fit")
        X = pd.DataFrame([features], columns=self.features)
        prob = self.model.predict_proba(X)[0, 1]
        return float(prob)
