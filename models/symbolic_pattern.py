from __future__ import annotations

"""Symbolic regression based pattern recognition model."""

from functools import cached_property

import numpy as np


class SymbolicPatternModel:
    """Discover explicit mathematical patterns using symbolic regression.

    The model leverages :class:`gplearn.genetic.SymbolicRegressor` to evolve
    analytical expressions that best explain the relationship between market
    features and a target variable.  Such expressions provide transparent
    pattern recognition that can generalise across regimes.
    """

    def __init__(self, **kwargs) -> None:
        params = {
            "population_size": 500,
            "generations": 20,
            "tournament_size": 20,
            "stopping_criteria": 0.0,
            "p_crossover": 0.7,
            "p_subtree_mutation": 0.1,
            "p_hoist_mutation": 0.05,
            "p_point_mutation": 0.1,
            "max_samples": 0.9,
            "verbose": 0,
            "parsimony_coefficient": 0.01,
            "random_state": 0,
        }
        params.update(kwargs)
        self._params = params
        self._model = None

    @cached_property
    def model(self):
        from gplearn.genetic import SymbolicRegressor

        return SymbolicRegressor(**self._params)

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the symbolic regressor on training data."""

        self.model.fit(X, y)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for ``X``."""

        return self.model.predict(X)

    def export_expression(self) -> str:
        """Return the evolved symbolic expression as a string."""

        return str(self.model._program)


__all__ = ["SymbolicPatternModel"]
