import asyncio
from pathlib import Path
import sys

import pandas as pd

pd.MultiIndex = type("MultiIndex", (), {"from_tuples": staticmethod(lambda *a, **k: [])})

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy.strategy_lab import StrategyLab
from strategy.router import StrategyRouter
from services import message_bus
from services.message_bus import Topics


def test_strategy_promoted_after_threshold(tmp_path):
    async def _run():
        # Reset message bus for isolation
        message_bus._message_bus = message_bus.MessageBus(backend="inmemory")
        bus = message_bus.get_message_bus()

        StrategyRouter._load_scoreboard = lambda self: pd.DataFrame()  # type: ignore
        StrategyRouter._load_elo_ratings = lambda self: {}  # type: ignore
        StrategyRouter._load_rationale_scores = lambda self: {}  # type: ignore
        StrategyRouter._load_regime_performance = lambda self: pd.DataFrame()  # type: ignore

        router = StrategyRouter(
            algorithms={},
            scoreboard_path=tmp_path / "scores.parquet",
            elo_path=tmp_path / "elo.parquet",
            rationale_path=tmp_path / "rationale.parquet",
            regime_perf_path=tmp_path / "regime.parquet",
        )

        def train_fn(df: pd.DataFrame):
            return lambda msg: float(msg["value"])

        lab = StrategyLab(
            train_fn=train_fn,
            router=router,
            thresholds={"sharpe": 1.0},
            history_path=tmp_path / "history.csv",
        )

        data = pd.DataFrame({"price": [1, 2, 3]})
        await lab.train_and_deploy("cand", data)
        monitor_task = asyncio.create_task(lab.monitor())

        assert "cand" not in router.algorithms

        await bus.publish(Topics.SIGNALS, {"Timestamp": 1, "value": 0.0})
        await asyncio.sleep(0.01)
        assert "cand" not in router.algorithms

        await bus.publish(Topics.SIGNALS, {"Timestamp": 2, "value": 1.0})
        await asyncio.sleep(0.01)
        assert "cand" not in router.algorithms

        await bus.publish(Topics.SIGNALS, {"Timestamp": 3, "value": 1.0})
        await asyncio.sleep(0.05)
        assert "cand" in router.algorithms

        monitor_task.cancel()
        for task in lab.tasks.values():
            task.cancel()
        await asyncio.sleep(0)

    asyncio.run(_run())
