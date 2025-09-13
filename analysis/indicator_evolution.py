from __future__ import annotations

"""Genetic-programming based indicator evolution utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import json
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer  # type: ignore
from gplearn.functions import make_function  # type: ignore


@dataclass
class EvolvedIndicator:
    """Record of an evolved indicator."""

    name: str
    formula: str
    score: float


class IndicatorEvolver:
    """Evolve new indicators from base columns using genetic programming."""

    def __init__(self, random_state: int = 0) -> None:
        diff = make_function(lambda x: np.append([0], np.diff(x)), name="diff", arity=1)
        ratio = make_function(
            lambda x, y: np.divide(x, np.where(np.abs(y) > 1e-9, y, np.nan)),
            name="ratio",
            arity=2,
        )
        conv = make_function(
            lambda x: np.convolve(x, np.ones(3) / 3, mode="same"),
            name="conv",
            arity=1,
        )
        self.function_set = [diff, ratio, conv]
        self.random_state = random_state

    # ------------------------------------------------------------------
    @staticmethod
    def _sharpe(x: np.ndarray) -> float:
        r = x - x.mean()
        denom = np.std(r)
        return float(r.mean() / denom) if denom else 0.0

    # ------------------------------------------------------------------
    def evolve(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        generations: int = 5,
        population_size: int = 50,
        n_components: int = 3,
    ) -> List[EvolvedIndicator]:
        """Return ``n_components`` evolved indicators ranked by validation Sharpe."""

        gp = SymbolicTransformer(
            generations=generations,
            population_size=population_size,
            hall_of_fame=n_components,
            n_components=n_components,
            function_set=self.function_set,
            random_state=self.random_state,
            n_jobs=1,
        )
        gp.fit(X.to_numpy(), y.to_numpy())
        indicators: List[EvolvedIndicator] = []
        for i, prog in enumerate(gp._best_programs[:n_components]):
            values = prog.execute(X.to_numpy())
            sharpe = self._sharpe(np.sign(values) * y.to_numpy())
            indicators.append(EvolvedIndicator(f"evo_ind_{i}", str(prog), sharpe))
        indicators.sort(key=lambda ind: ind.score, reverse=True)
        return indicators

    # ------------------------------------------------------------------
    def save(self, inds: List[EvolvedIndicator], path: Path) -> None:
        data = [ind.__dict__ for ind in inds]
        path.write_text(json.dumps(data, indent=2))


__all__ = ["IndicatorEvolver", "EvolvedIndicator"]
