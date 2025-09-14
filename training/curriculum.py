"""Curriculum learning utilities.

This module defines a simple :class:`CurriculumScheduler` that orchestrates a
set of training stages.  Each stage provides a callable which performs the
training for that stage and returns a validation metric.  The scheduler
progresses to the next stage only if the metric meets a configurable
threshold.  Metrics for each completed stage are stored in
:attr:`metrics`.

The scheduler is intentionally lightweight so it can be used from both the
standard supervised trainer in :mod:`train` and the reinforcement learning
trainer in :mod:`train_rl`.  External callers typically create stage
configuration dictionaries and then use
:meth:`CurriculumScheduler.from_config` to build the scheduler.  This keeps the
integration minimal while allowing advanced curriculum strategies to be
implemented outside of this module.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Tuple, Any, Sequence
import asyncio


@dataclass
class CurriculumStage:
    """Description of a single curriculum stage.

    Parameters
    ----------
    name:
        Friendly name for the stage.  Used mainly for logging.
    train_fn:
        Callable that executes the training for the stage and returns a
        validation metric.  Higher is assumed to be better.
    threshold:
        Minimum metric value required to progress to the next stage.
    """

    name: str
    train_fn: Callable[[], float]
    threshold: float = 0.0


class CurriculumScheduler:
    """Simple scheduler that progresses through stages based on metrics."""

    def __init__(self, stages: Iterable[CurriculumStage]):
        self.stages: List[CurriculumStage] = list(stages)
        self.current_stage: int = 0
        self.metrics: List[Tuple[str, float]] = []

    def run(self) -> List[Tuple[str, float]]:
        """Execute all curriculum stages sequentially.

        Returns
        -------
        list of (stage name, metric) tuples for each attempted stage.  The
        scheduler stops when a stage fails to meet its threshold.
        """

        for idx, stage in enumerate(self.stages):
            metric = stage.train_fn()
            self.metrics.append((stage.name, metric))
            if metric < stage.threshold:
                break
            self.current_stage = idx + 1
        return self.metrics

    @property
    def final_metric(self) -> float:
        """Validation metric from the last attempted stage."""

        return self.metrics[-1][1] if self.metrics else 0.0

    @classmethod
    def from_config(
        cls, stages_cfg: Iterable[dict] | None, build_fn: Callable[[dict], Callable[[], float]]
    ) -> "CurriculumScheduler | None":
        """Construct a scheduler from configuration.

        Parameters
        ----------
        stages_cfg:
            Iterable of dictionaries with ``name``, ``threshold`` and any
            additional keys understood by ``build_fn``.
        build_fn:
            Callable that takes a stage configuration dictionary and returns a
            ``train_fn`` compatible with :class:`CurriculumStage`.
        """

        if not stages_cfg:
            return None
        stages: List[CurriculumStage] = []
        for st in stages_cfg:
            name = st.get("name", f"stage-{len(stages)}")
            threshold = float(st.get("threshold", 0.0))
            train_fn = build_fn(st)
            stages.append(CurriculumStage(name, train_fn, threshold))
        return cls(stages)


def build_strategy_curriculum(
    simple_fn: Callable[[], float],
    combo_fn: Callable[[], float],
    graph_fn: Callable[[], float],
    thresholds: Sequence[float] | None = None,
) -> "CurriculumScheduler":
    """Create a three stage curriculum used in tests and examples.

    The stages progress from simple single-indicator strategies to
    combinations of multiple indicators and finally graph based strategies.
    Each ``*_fn`` should execute the training for the respective stage and
    return a validation metric between 0 and 1.  The scheduler will
    automatically advance to the next stage once the metric meets the
    associated threshold.
    """

    thresholds = list(thresholds or (0.6, 0.7, 0.8))
    stages = [
        CurriculumStage("simple", simple_fn, thresholds[0]),
        CurriculumStage("combo", combo_fn, thresholds[1]),
        CurriculumStage("graph", graph_fn, thresholds[2]),
    ]
    return CurriculumScheduler(stages)


def _evaluate_live(
    strategy: Callable[[dict], float],
    messages: Iterable[dict],
) -> float:
    """Run ``strategy`` on ``messages`` using :class:`ShadowRunner`.

    This helper is intentionally lightweight and used by
    :func:`build_live_strategy_curriculum` to gate curriculum progression based
    on live performance metrics such as Sharpe ratio.
    """

    from strategy.shadow_runner import ShadowRunner  # Local import to avoid heavy dependency

    class _Bus:
        def __init__(self, data: Iterable[dict]):
            self._data = list(data)

        async def subscribe(self, _topic: str):
            for m in self._data:
                yield m

    async def _run() -> float:
        metrics_q: asyncio.Queue = asyncio.Queue()
        runner = ShadowRunner("curriculum", strategy, bus=_Bus(messages), metrics_queue=metrics_q)
        await runner.run()
        last: dict | None = None
        while not metrics_q.empty():
            last = await metrics_q.get()
        return float(last.get("sharpe", 0.0)) if last else 0.0

    return asyncio.run(_run())


def build_live_strategy_curriculum(
    simple_fn: Callable[[], Callable[[dict], float]],
    combo_fn: Callable[[], Callable[[dict], float]],
    graph_fn: Callable[[], Callable[[dict], float]],
    messages: Iterable[dict],
    thresholds: Sequence[float] | None = None,
) -> "CurriculumScheduler":
    """Create a curriculum that evaluates stages on live data.

    ``*_fn`` callables should return a strategy function.  Each stage is
    evaluated using :class:`ShadowRunner` on ``messages`` and the Sharpe ratio is
    compared against the corresponding threshold before progressing.
    """

    thresholds = list(thresholds or (0.6, 0.7, 0.8))

    def _wrap(fn: Callable[[], Callable[[dict], float]], name: str, thr: float) -> CurriculumStage:
        return CurriculumStage(name, lambda: _evaluate_live(fn(), messages), thr)

    stages = [
        _wrap(simple_fn, "simple", thresholds[0]),
        _wrap(combo_fn, "combo", thresholds[1]),
        _wrap(graph_fn, "graph", thresholds[2]),
    ]
    return CurriculumScheduler(stages)


__all__ = [
    "CurriculumStage",
    "CurriculumScheduler",
    "build_strategy_curriculum",
    "build_live_strategy_curriculum",
]
