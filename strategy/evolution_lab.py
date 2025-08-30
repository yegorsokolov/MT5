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
from typing import Any, Callable, Dict, TYPE_CHECKING
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
        """Return a slightly perturbed variant of :attr:`base`.

        The mutation simply adds Gaussian noise to the base strategy's output.
        Real implementations can override this with more sophisticated
        hyperparameter or architecture search.
        """

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
