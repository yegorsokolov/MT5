from __future__ import annotations

"""Strategy evolution laboratory and tournament management.

This module generates mutated variants of a base strategy, runs each in
shadow mode using :class:`strategy.shadow_runner.ShadowRunner` and tracks
rolling performance metrics.  Results are persisted to
``reports/strategy_tournament.parquet`` and the best performing strategy is
promoted to live trading once it exceeds a significance threshold.  Poorly
performing live strategies are demoted back to shadow mode but remain in the
pool for further evolution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, TYPE_CHECKING
import asyncio
import random

import pandas as pd

from .shadow_runner import ShadowRunner

if TYPE_CHECKING:  # pragma: no cover - typing only
    from model_registry import ModelRegistry

try:  # pragma: no cover - optional dependency during tests
    from signal_queue import _ROUTER
except Exception:  # pragma: no cover - defensive fallback
    _ROUTER = None  # type: ignore


Strategy = Callable[[Dict[str, Any]], float]


@dataclass
class EvolutionLab:
    """Spawn strategy variants and manage a performance tournament."""

    base: Strategy
    register: Callable[[str, Strategy], None] | None = None
    out_path: Path = Path("reports/strategy_tournament.parquet")
    significance: float = 2.0
    variants: Dict[str, Strategy] = field(default_factory=dict)
    registry: ModelRegistry | None = None

    # ------------------------------------------------------------------
    def _mutate(self, scale: float = 0.1) -> Strategy:
        """Return a perturbed or composite variant of :attr:`base`.

        Historically mutation simply injected Gaussian noise into the base
        strategy's output.  To allow the evolution laboratory to explore more
        complex behaviours we now support three simple operators:

        ``noise``
            add Gaussian noise to the base output (original behaviour)
        ``scale``
            multiply the base output by a random factor
        ``mix``
            blend the base with a previously spawned variant, enabling
            higher‑order strategy combinations

        Real applications may override this with domain specific logic but the
        built in operators already yield richer search spaces for tests and
        lightweight deployments.
        """

        op_choices = ["noise", "scale"]
        if self.variants:
            op_choices.append("mix")
        op = random.choice(op_choices)

        if op == "scale":
            factor = 1.0 + random.gauss(0.0, scale)

            def variant(features: Dict[str, Any], *, _factor=factor) -> float:
                return float(self.base(features)) * _factor

            return variant

        if op == "mix" and self.variants:
            other = random.choice(list(self.variants.values()))
            weight = random.random()

            def variant(
                features: Dict[str, Any], *, _other=other, _w=weight
            ) -> float:
                base_val = float(self.base(features))
                return _w * base_val + (1.0 - _w) * float(_other(features))

            return variant

        # Default: additive noise mutation
        def variant(features: Dict[str, Any]) -> float:
            base_val = float(self.base(features))
            return base_val + random.gauss(0.0, scale)

        return variant

    # ------------------------------------------------------------------
    def spawn(self, n: int = 3) -> None:
        """Create ``n`` mutated variants and start their shadow runners."""

        start = len(self.variants)
        for i in range(start, start + n):
            name = f"{getattr(self.base, '__name__', 'strategy')}_var{i}"
            handler = self._mutate()
            chosen = handler
            if self.registry is not None:
                caps = self.registry.monitor.capabilities
                if caps.cpus < 4 or caps.memory_gb < 8:
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "remote_client",
                        Path(__file__).resolve().parents[1] / "models" / "remote_client.py",
                    )
                    rc = importlib.util.module_from_spec(spec)
                    assert spec and spec.loader
                    spec.loader.exec_module(rc)  # type: ignore

                    def remote_handler(features: Dict[str, Any], *, _name=name, _rc=rc) -> float:
                        preds = _rc.predict_remote(_name, [features])
                        return float(preds[0]) if preds else 0.0

                    chosen = remote_handler
            self.variants[name] = chosen
            if self.register is not None:
                self.register(name, chosen)
            else:  # pragma: no cover - fallback when orchestrator absent
                runner = ShadowRunner(name=name, handler=chosen)
                try:
                    import asyncio

                    loop = asyncio.get_running_loop()
                except RuntimeError:  # pragma: no cover - no running loop
                    loop = asyncio.get_event_loop()
                loop.create_task(runner.run())

    # ------------------------------------------------------------------
    async def evolve_queue(
        self,
        seeds: Iterable[int],
        messages: Iterable[Dict[str, Any]],
    ) -> list[Dict[str, float]]:
        """Run multiple evolution jobs concurrently using a task queue.

        Each ``seed`` spawns a mutated variant which is then evaluated on the
        provided ``messages`` using :class:`ShadowRunner`.  The final metrics for
        each variant are returned once all jobs have completed.
        """

        queue: asyncio.Queue[int] = asyncio.Queue()
        for s in seeds:
            queue.put_nowait(int(s))

        results: list[Dict[str, float]] = []
        msgs = list(messages)

        class _ListBus:
            def __init__(self, data: list[Dict[str, Any]]):
                self.data = data

            async def subscribe(self, _topic: str):
                for m in self.data:
                    yield m

        async def worker() -> None:
            while True:
                try:
                    seed = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                random.seed(seed)
                name = f"{getattr(self.base, '__name__', 'strategy')}_{seed}"
                handler = self._mutate()
                metrics_q: asyncio.Queue = asyncio.Queue()
                runner = ShadowRunner(
                    name=name,
                    handler=handler,
                    bus=_ListBus(msgs),
                    metrics_queue=metrics_q,
                )
                await runner.run()
                last: Dict[str, float] | None = None
                while not metrics_q.empty():
                    last = await metrics_q.get()
                if last is not None:
                    last["name"] = name
                    results.append(last)
                queue.task_done()

        tasks = [asyncio.create_task(worker()) for _ in range(queue.qsize())]
        await asyncio.gather(*tasks)
        return results

    # ------------------------------------------------------------------
    def _shadow_metrics(self, name: str) -> Dict[str, float] | None:
        """Load latest metrics from the shadow report for ``name``."""

        path = Path("reports/shadow") / f"{name}.csv"
        if not path.exists():
            return None
        try:
            df = pd.read_csv(
                path,
                names=["timestamp", "pnl", "equity", "drawdown", "sharpe"],
            )
            if df.empty:
                return None
            last = df.iloc[-1]
            return {
                "pnl": float(last["pnl"]),
                "drawdown": float(last["drawdown"]),
                "sharpe": float(last["sharpe"]),
            }
        except Exception:  # pragma: no cover - robust to malformed files
            return None

    # ------------------------------------------------------------------
    def record(self) -> pd.DataFrame:
        """Append latest metrics for all variants to the parquet report."""

        rows = []
        for name in self.variants:
            met = self._shadow_metrics(name)
            if met is None:
                continue
            met["name"] = name
            rows.append(met)
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            prev = pd.read_parquet(self.out_path)
            df = pd.concat([prev, df], ignore_index=True)
        except Exception:
            pass
        df.to_parquet(self.out_path, index=False)
        return df

    # ------------------------------------------------------------------
    def _demote_poor_performers(self, df: pd.DataFrame) -> None:
        """Remove underperforming live strategies from the router."""

        if _ROUTER is None or df.empty:
            return
        for name in list(_ROUTER.algorithms.keys()):
            sub = df[df["name"] == name]
            if sub.empty:
                continue
            sharpe = float(sub.iloc[-1]["sharpe"])
            if sharpe < 0.0:  # simple demotion criterion
                _ROUTER.algorithms.pop(name, None)
                _ROUTER.A.pop(name, None)
                _ROUTER.b.pop(name, None)

    # ------------------------------------------------------------------
    def promote(self) -> None:
        """Promote the best variant to live trading if significant."""

        df = self.record()
        if df.empty:
            return
        best = df.sort_values("sharpe", ascending=False).iloc[0]
        if best["sharpe"] < self.significance:
            self._demote_poor_performers(df)
            return
        name = str(best["name"])
        handler = self.variants.get(name)
        if handler is None or _ROUTER is None:
            return
        if name not in _ROUTER.algorithms:
            _ROUTER.register(name, handler)
        self._demote_poor_performers(df)

    # ------------------------------------------------------------------
    def run(self, n_variants: int = 3) -> None:
        """Spawn new variants and update tournament rankings."""

        self.spawn(n_variants)
        self.promote()


__all__ = ["EvolutionLab"]
