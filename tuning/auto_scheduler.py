"""Nightly hyperparameter search and deployment scheduler."""

from __future__ import annotations

from dataclasses import dataclass, field
import asyncio
import logging
from typing import Any, Callable, Dict
import json

import pandas as pd

try:  # pragma: no cover - ``ray`` is optional
    from ray import tune
    from ray.tune.search.optuna import OptunaSearch
except Exception:  # pragma: no cover - fallback to light stub
from mt5.ray_stub import tune  # type: ignore

    OptunaSearch = tune.search.optuna.OptunaSearch  # type: ignore

from analysis import regime_detection
from models import model_store
from models.hot_reload import hot_reload

logger = logging.getLogger(__name__)


def _current_regime(data: pd.DataFrame) -> int:
    labels = regime_detection.detect_regimes(data.tail(200))
    return int(labels.iloc[-1]) if not labels.empty else 0


def _incumbent_score(regime: int) -> float:
    """Return best recorded score for *regime* from the model store."""
    store = model_store._ensure_store()  # type: ignore[attr-defined]
    best = float("-inf")
    for path in sorted(store.glob("tuned_*.json")):
        try:
            with open(path) as fh:
                meta = json.load(fh)
        except Exception:
            continue
        if meta.get("regime") == regime:
            score = float(meta.get("score", float("-inf")))
            if score > best:
                best = score
    return best


@dataclass
class AutoScheduler:
    """Launch nightly Optuna/Ray Tune searches.

    Parameters
    ----------
    objective_fn:
        Function returning a performance score given ``(params, data)``.
    search_space:
        Ray Tune configuration dictionary describing the parameter space.
    data_loader:
        Callable returning the latest :class:`pandas.DataFrame` snapshot.
    n_samples:
        Number of Ray Tune samples to evaluate nightly.
    margin:
        Relative improvement required over the incumbent to deploy.
    """

    objective_fn: Callable[[Dict[str, Any], pd.DataFrame], float]
    search_space: Dict[str, Any]
    data_loader: Callable[[], pd.DataFrame]
    n_samples: int = 20
    margin: float = 0.01
    metric: str = "score"
    mode: str = "max"
    _task: asyncio.Task | None = field(default=None, init=False, repr=False)

    def _objective(self, config: Dict[str, Any], data: pd.DataFrame) -> float:
        score = float(self.objective_fn(config, data))
        try:  # pragma: no cover - ``tune.report`` may be a no-op in stub
            tune.report(**{self.metric: score})
        except Exception:
            pass
        return score

    async def run_nightly(self) -> None:
        """Run the search once every UTC midnight."""
        while True:
            now = asyncio.get_event_loop().time()
            # seconds until next midnight
            from datetime import datetime, timedelta

            utc_now = datetime.utcnow()
            tomorrow = (utc_now + timedelta(days=1)).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
            sleep_for = (tomorrow - utc_now).total_seconds()
            await asyncio.sleep(max(0, sleep_for))
            await self.run_once()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self.run_nightly())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            self._task = None

    async def run_once(self) -> Dict[str, Any]:
        """Execute a single tuning cycle.

        This function loads the latest data snapshot, detects the current
        market regime and runs an Optuna/Ray Tune search across
        ``search_space``.  The best configuration is compared against the
        incumbent performance stored in :mod:`model_store`.  If the
        relative improvement exceeds ``margin`` the parameters are
        persisted and immediately hot reloaded.  The
        :class:`model_registry.ModelRegistry` will pick up the new
        parameters during its next daily resource probe.
        """

        data = self.data_loader()
        regime = _current_regime(data)
        logger.info("Starting tuning run for regime %s", regime)

        search_alg = OptunaSearch(metric=self.metric, mode=self.mode)

        def obj(cfg: Dict[str, Any]) -> float:
            return self._objective(cfg, data)

        analysis = tune.run(
            tune.with_parameters(obj),
            config=self.search_space,
            num_samples=self.n_samples,
            search_alg=search_alg,
        )
        best_config = getattr(analysis, "best_config", {})
        best_score = getattr(analysis, "best_result", {}).get(
            self.metric, getattr(analysis, "best_score", 0.0)
        )
        logger.info("Tuning completed with best %.4f", best_score)

        incumbent = _incumbent_score(regime)
        if incumbent == float("-inf") or best_score >= incumbent * (1 + self.margin):
            model_store.save_tuned_params(
                {
                    "regime": regime,
                    "params": best_config,
                    "score": best_score,
                }
            )
            hot_reload(best_config)
            logger.info(
                "Deployed new params for regime %s score %.4f", regime, best_score
            )
        else:
            logger.info(
                "Best %.4f did not beat incumbent %.4f by margin %.3f",
                best_score,
                incumbent,
                self.margin,
            )
        return best_config


__all__ = ["AutoScheduler"]
