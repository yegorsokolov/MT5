from __future__ import annotations

"""Evolutionary multi-objective search using a lightweight NSGA-II approach.

The implementation focuses on optimising trading parameters across multiple
objectives such as total return, maximum drawdown and trade count.  It is a
minimal, dependency free variant of NSGA-II suitable for quick experiments and
unit tests.  The algorithm expects an evaluation function that returns a
sequence of objective values to *minimise*.  Users can wrap metrics they wish to
maximise (e.g. returns) by negating them before returning.
"""

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence, Any
import math
import random

try:  # pragma: no cover - optional dependency during tests
    from models import model_store, hot_reload
except Exception:  # pragma: no cover - model persistence is optional
    model_store = None  # type: ignore
    hot_reload = None  # type: ignore


ParamSpace = Mapping[str, tuple[float, float, str]]


@dataclass
class Individual:
    """Container for a candidate solution."""

    params: dict
    objectives: Sequence[float]


def _sample_param(low: float, high: float, kind: str) -> Any:
    if kind == "int":
        return random.randint(int(low), int(high))
    if kind == "log":
        return 10 ** random.uniform(math.log10(low), math.log10(high))
    return random.uniform(low, high)


def _sample_params(space: ParamSpace) -> dict:
    return {k: _sample_param(*v) for k, v in space.items()}


def _mutate(params: dict, space: ParamSpace, rate: float = 0.2) -> dict:
    new_params = params.copy()
    for k, bounds in space.items():
        if random.random() < rate:
            new_params[k] = _sample_param(*bounds)
    return new_params


def _dominates(a: Individual, b: Individual) -> bool:
    return all(x <= y for x, y in zip(a.objectives, b.objectives)) and any(
        x < y for x, y in zip(a.objectives, b.objectives)
    )


def _pareto_front(pop: list[Individual]) -> list[Individual]:
    front: list[Individual] = []
    for ind in pop:
        if not any(_dominates(other, ind) for other in pop if other is not ind):
            front.append(ind)
    return front


def run_evolutionary_search(
    eval_fn: Callable[[dict], Sequence[float]],
    param_space: ParamSpace,
    *,
    generations: int = 5,
    population_size: int = 20,
) -> list[dict]:
    """Execute a simple evolutionary search.

    Parameters
    ----------
    eval_fn:
        Function receiving a parameter dictionary and returning a sequence of
        objective values to minimise.  Typically this function will train a
        model on a recent data snapshot and return ``(-return, max_drawdown,
        -trade_count)``.
    param_space:
        Mapping of parameter name to ``(low, high, kind)`` tuples where ``kind``
        is ``"int"``, ``"log"`` or ``"float"``.
    generations:
        Number of evolutionary generations to run.
    population_size:
        Number of individuals per generation.

    Returns
    -------
    list[dict]
        The Pareto-optimal parameter sets.
    """

    population = [
        Individual(params=_sample_params(param_space), objectives=())
        for _ in range(population_size)
    ]
    for ind in population:
        ind.objectives = tuple(eval_fn(ind.params))

    for _ in range(generations):
        offspring: list[Individual] = []
        for parent in population:
            child_params = _mutate(parent.params, param_space)
            child_obj = tuple(eval_fn(child_params))
            offspring.append(Individual(child_params, child_obj))
        population.extend(offspring)
        population = _pareto_front(population)
        population = population[:population_size]

    pareto = _pareto_front(population)
    if model_store is not None:  # Persist Pareto front for offline analysis
        for ind in pareto:
            model_store.save_tuned_params(
                {"params": ind.params, "objectives": list(ind.objectives)}
            )
    if hot_reload is not None and pareto:  # Activate best candidate
        best = min(pareto, key=lambda i: i.objectives[0])
        hot_reload.hot_reload(best.params)
    return [ind.params for ind in pareto]


__all__ = ["run_evolutionary_search"]
