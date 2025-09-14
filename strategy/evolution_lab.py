from __future__ import annotations

"""Strategy evolution laboratory and tournament management.

This module generates mutated variants of a base strategy, including graph
based strategies which can request new indicators from the feature store.  The
variants are evaluated concurrently in shadow mode using
:class:`strategy.shadow_runner.ShadowRunner` with an asynchronous task pool
whose size is capped based on available system resources.  Lineage and
performance metrics for each variant are recorded so that the most promising
candidate can be promoted to live trading once it exceeds a significance
threshold.  Poorly performing live strategies are demoted back to shadow mode
but remain in the pool for further evolution.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, TYPE_CHECKING
import asyncio
import os
import random

import pandas as pd

from strategies.graph_dsl import Indicator, StrategyGraph
from feature_store import latest_version, request_indicator
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

    base: Strategy | StrategyGraph
    register: Callable[[str, Strategy], None] | None = None
    out_path: Path = Path("reports/strategy_tournament.parquet")
    lineage_path: Path = Path("reports/mutation_lineage.parquet")
    significance: float = 2.0
    variants: Dict[str, Strategy] = field(default_factory=dict)
    lineage: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    registry: ModelRegistry | None = None

    # ------------------------------------------------------------------
    def _mutate(
        self, scale: float = 0.1, *, with_info: bool = False
    ) -> Strategy | StrategyGraph | tuple[Strategy | StrategyGraph, Dict[str, Any]]:
        """Return a perturbed variant of :attr:`base`.

        When ``with_info`` is ``True`` a tuple of ``(variant, info)`` is
        returned, otherwise only the variant callable/graph is produced for
        backward compatibility with older callers.
        """

        if isinstance(self.base, StrategyGraph):
            variant, info = self._mutate_graph(self.base)
            return (variant, info) if with_info else variant

        op_choices = ["noise", "scale"]
        if self.variants:
            op_choices.append("mix")
        op = random.choice(op_choices)

        if op == "scale":
            factor = 1.0 + random.gauss(0.0, scale)

            def variant(features: Dict[str, Any], *, _factor=factor) -> float:
                return float(self.base(features)) * _factor

            info = {"type": "scale", "factor": factor}
            return (variant, info) if with_info else variant

        if op == "mix" and self.variants:
            other = random.choice(list(self.variants.values()))
            weight = random.random()

            def variant(features: Dict[str, Any], *, _other=other, _w=weight) -> float:
                base_val = float(self.base(features))
                return _w * base_val + (1.0 - _w) * float(_other(features))

            info = {"type": "mix", "weight": weight}
            return (variant, info) if with_info else variant

        def variant(features: Dict[str, Any]) -> float:
            base_val = float(self.base(features))
            return base_val + random.gauss(0.0, scale)

        info = {"type": "noise"}
        return (variant, info) if with_info else variant

    # ------------------------------------------------------------------
    def _mutate_graph(
        self, graph: StrategyGraph
    ) -> tuple[StrategyGraph, Dict[str, Any]]:
        """Return a mutated copy of ``graph`` and mutation metadata."""

        g = StrategyGraph(
            nodes=dict(graph.nodes), edges=list(graph.edges), entry=graph.entry
        )
        op = random.choice(["insert", "remove"])
        info: Dict[str, Any] = {"type": op}
        if op == "insert":
            version = latest_version()
            cols = request_indicator(version) if version else []
            if cols:
                lhs = random.choice(cols)
                rhs = random.choice(cols)
                node = Indicator(
                    lhs=lhs, op=random.choice(list(Indicator.OPS)), rhs=rhs
                )
                new_id = g.insert_node(g.entry, None, node)
                info["node"] = new_id
        else:  # remove
            removable = [n for n in g.nodes if n != g.entry]
            if removable:
                node_id = random.choice(removable)
                g.remove_node(node_id)
                info["removed"] = node_id
        return g, info

    # ------------------------------------------------------------------
    def spawn(self, n: int = 3) -> None:
        """Create ``n`` mutated variants and start their shadow runners."""

        start = len(self.variants)
        for i in range(start, start + n):
            name = f"{getattr(self.base, '__name__', 'strategy')}_var{i}"
            handler, info = self._mutate(with_info=True)
            chosen: Strategy
            if isinstance(handler, StrategyGraph):

                def _exec(features: Dict[str, Any], *, _g=handler) -> float:
                    return _g.run([features])

                chosen = _exec
            else:
                chosen = handler
            self.lineage[name] = info
            if self.registry is not None:
                caps = self.registry.monitor.capabilities
                if caps.cpus < 4 or caps.memory_gb < 8:
                    import importlib.util

                    spec = importlib.util.spec_from_file_location(
                        "remote_client",
                        Path(__file__).resolve().parents[1]
                        / "models"
                        / "remote_client.py",
                    )
                    rc = importlib.util.module_from_spec(spec)
                    assert spec and spec.loader
                    spec.loader.exec_module(rc)  # type: ignore

                    def remote_handler(
                        features: Dict[str, Any], *, _name=name, _rc=rc
                    ) -> float:
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
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                loop.create_task(runner.run())

    # ------------------------------------------------------------------
    async def evolve_queue(
        self,
        seeds: Iterable[int],
        messages: Iterable[Dict[str, Any]],
        concurrency: int | None = None,
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

        if concurrency is None:
            cpu = os.cpu_count() or 1
            gpu = 0
            try:
                import torch

                gpu = torch.cuda.device_count()
            except Exception:  # pragma: no cover - optional dependency
                pass
            if gpu > 0:
                cpu = min(cpu, gpu)
            concurrency = max(1, cpu)

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
                handler, info = self._mutate(with_info=True)
                chosen: Strategy
                if isinstance(handler, StrategyGraph):

                    def _exec(features: Dict[str, Any], *, _g=handler) -> float:
                        return _g.run([features])

                    chosen = _exec
                else:
                    chosen = handler
                metrics_q: asyncio.Queue = asyncio.Queue()
                runner = ShadowRunner(
                    name=name,
                    handler=chosen,
                    bus=_ListBus(msgs),
                    metrics_queue=metrics_q,
                )
                await runner.run()
                last: Dict[str, float] | None = None
                while not metrics_q.empty():
                    last = await metrics_q.get()
                if last is not None:
                    record = {"name": name, "seed": seed, **info, **last}
                    results.append(record)
                    # Track lineage and spawn the variant for live testing
                    self.lineage[name] = record
                    self.variants[name] = chosen
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:  # pragma: no cover - no running loop
                        loop = asyncio.get_event_loop()
                    if self.register is not None:
                        self.register(name, chosen)
                    else:  # pragma: no cover - fallback when orchestrator absent
                        loop.create_task(ShadowRunner(name=name, handler=chosen).run())
                queue.task_done()

        tasks = [
            asyncio.create_task(worker())
            for _ in range(min(queue.qsize(), concurrency))
        ]
        await asyncio.gather(*tasks)
        self._persist_lineage(results)
        return results

    # ------------------------------------------------------------------
    def _persist_lineage(self, records: list[Dict[str, Any]]) -> None:
        """Append mutation lineage and metrics to :attr:`lineage_path`."""

        if not records:
            return
        df = pd.DataFrame(records)
        self.lineage_path.parent.mkdir(parents=True, exist_ok=True)
        # When pandas lacks parquet/CSV support (e.g. test stubs), fall back to
        # the built-in ``csv`` module for persistence.
        if not (hasattr(df, "to_parquet") or hasattr(df, "to_csv")):
            import csv

            with self.lineage_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(records)
            return

        # Load previous lineage if available, falling back to CSV when parquet
        # support is missing (common in the lightweight test environment).
        if self.lineage_path.exists():
            try:  # pragma: no cover - depends on optional parquet deps
                prev = pd.read_parquet(self.lineage_path)
                df = pd.concat([prev, df], ignore_index=True)
            except Exception:
                try:
                    prev = pd.read_csv(self.lineage_path)
                    df = pd.concat([prev, df], ignore_index=True)
                except Exception:
                    pass
        try:  # pragma: no cover - parquet requires optional deps
            df.to_parquet(self.lineage_path, index=False)
        except Exception:  # Fall back to CSV if parquet is unavailable
            df.to_csv(self.lineage_path, index=False)

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
        if not (hasattr(df, "to_parquet") or hasattr(df, "to_csv")):
            import csv

            with self.out_path.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerows(rows)
            return pd.DataFrame(rows)
        # Attempt to append to an existing parquet report, falling back to CSV
        # when parquet support is unavailable.
        if self.out_path.exists():
            try:  # pragma: no cover - optional parquet dependency
                prev = pd.read_parquet(self.out_path)
                df = pd.concat([prev, df], ignore_index=True)
            except Exception:
                try:
                    prev = pd.read_csv(self.out_path)
                    df = pd.concat([prev, df], ignore_index=True)
                except Exception:
                    pass
        try:  # pragma: no cover - parquet requires optional deps
            df.to_parquet(self.out_path, index=False)
        except Exception:
            df.to_csv(self.out_path, index=False)
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
