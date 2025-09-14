import asyncio
import sys
from pathlib import Path

import pandas as pd
pd.MultiIndex = type("MultiIndex", (), {"from_tuples": staticmethod(lambda *a, **k: [])})

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy.strategy_lab import StrategyLab
from strategy.router import StrategyRouter
from services import message_bus


def _setup_router(tmp_path):
    StrategyRouter._load_scoreboard = lambda self: pd.DataFrame()  # type: ignore
    StrategyRouter._load_elo_ratings = lambda self: {}  # type: ignore
    StrategyRouter._load_rationale_scores = lambda self: {}  # type: ignore
    StrategyRouter._load_regime_performance = lambda self: pd.DataFrame()  # type: ignore
    return StrategyRouter(
        algorithms={},
        scoreboard_path=tmp_path / "scores.parquet",
        elo_path=tmp_path / "elo.parquet",
        rationale_path=tmp_path / "rationale.parquet",
        regime_perf_path=tmp_path / "regime.parquet",
    )


def test_drawdown_violation_demotes(tmp_path):
    async def _run():
        message_bus._message_bus = message_bus.MessageBus(backend="inmemory")
        router = _setup_router(tmp_path)

        def train_fn(df: pd.DataFrame):
            return lambda msg: float(msg.get("value", 0.0))

        lab = StrategyLab(
            train_fn=train_fn,
            router=router,
            thresholds={},
            history_path=tmp_path / "history.csv",
            max_drawdown=1.0,
            max_total_drawdown=2.0,
        )

        data = pd.DataFrame({"price": [1, 2, 3]})
        await lab.train_and_deploy("cand", data)
        monitor_task = asyncio.create_task(lab.monitor())

        await lab.metrics_queue.put({"name": "cand", "pnl": -1.5, "drawdown": 1.5, "sharpe": -1.0})
        await asyncio.sleep(0.1)
        assert "cand" not in lab.risk_managers
        assert "cand" not in lab.runners

        monitor_task.cancel()
        for task in lab.tasks.values():
            task.cancel()
        await asyncio.sleep(0)

    asyncio.run(_run())


def test_fill_rate_violation_blocks_promotion(tmp_path):
    async def _run():
        message_bus._message_bus = message_bus.MessageBus(backend="inmemory")
        router = _setup_router(tmp_path)

        def train_fn(df: pd.DataFrame):
            return lambda msg: 0.0

        lab = StrategyLab(
            train_fn=train_fn,
            router=router,
            thresholds={"sharpe": -1.0},
            history_path=tmp_path / "history.csv",
            fill_ratio_threshold=0.8,
            max_cancel_rate=0.2,
            max_slippage=0.01,
        )

        data = pd.DataFrame({"price": [1, 2, 3]})
        await lab.train_and_deploy("cand", data)
        monitor_task = asyncio.create_task(lab.monitor())

        await lab.metrics_queue.put(
            {
                "name": "cand",
                "pnl": 0.0,
                "drawdown": 0.0,
                "sharpe": 0.0,
                "limit_orders": {"placed": 10, "filled": 2, "cancels": 8, "slippage": 0.02},
            }
        )
        await asyncio.sleep(0.1)
        # Strategy continues running but is not promoted
        assert "cand" in lab.runners
        assert "cand" not in router.algorithms
        lines = (tmp_path / "history.csv").read_text().strip().splitlines()
        assert (
            lines[0]
            == "name,version,pnl,drawdown,sharpe,fill_ratio,cancel_rate,slippage,violations"
        )
        last = lines[-1].split(",")
        assert abs(float(last[5]) - 0.2) < 1e-6
        assert abs(float(last[6]) - 0.8) < 1e-6
        assert abs(float(last[7]) - 0.02) < 1e-6
        assert last[8] != ""

        monitor_task.cancel()
        for task in lab.tasks.values():
            task.cancel()
        await asyncio.sleep(0)

    asyncio.run(_run())


