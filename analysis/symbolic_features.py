from __future__ import annotations

"""Genetic-programming based symbolic feature evolution utilities."""

from dataclasses import dataclass
from pathlib import Path
from typing import List

import json
import numpy as np
import pandas as pd
from gplearn.genetic import SymbolicTransformer  # type: ignore
from gplearn.functions import make_function  # type: ignore


@dataclass
class EvolvedSymbol:
    """Record of an evolved symbolic feature."""

    name: str
    formula: str
    score: float


class SymbolicFeatureEvolver:
    """Evolve new symbolic features using genetic programming."""

    def __init__(self, random_state: int = 0) -> None:
        roll_mean = make_function(
            lambda x: pd.Series(x).rolling(3).mean().to_numpy(),
            name="roll_mean",
            arity=1,
        )
        roll_std = make_function(
            lambda x: pd.Series(x).rolling(3).std().fillna(0.0).to_numpy(),
            name="roll_std",
            arity=1,
        )
        ratio = make_function(
            lambda x, y: np.divide(x, np.where(np.abs(y) > 1e-9, y, np.nan)),
            name="ratio",
            arity=2,
        )
        self.function_set = [roll_mean, roll_std, ratio]
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
    ) -> List[EvolvedSymbol]:
        """Return ``n_components`` evolved symbolic features ranked by Sharpe."""

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
        symbols: List[EvolvedSymbol] = []
        for i, prog in enumerate(gp._best_programs[:n_components]):
            values = prog.execute(X.to_numpy())
            sharpe = self._sharpe(np.sign(values) * y.to_numpy())
            symbols.append(EvolvedSymbol(f"evo_sym_{i}", str(prog), sharpe))
        symbols.sort(key=lambda s: s.score, reverse=True)
        return symbols

    # ------------------------------------------------------------------
    def save(self, symbols: List[EvolvedSymbol], path: Path) -> None:
        data = [s.__dict__ for s in symbols]
        path.write_text(json.dumps(data, indent=2))


__all__ = ["SymbolicFeatureEvolver", "EvolvedSymbol"]
