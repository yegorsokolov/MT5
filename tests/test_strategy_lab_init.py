import asyncio
from pathlib import Path
import sys
import pandas as pd

pd.MultiIndex = type("MultiIndex", (), {"from_tuples": staticmethod(lambda *a, **k: [])})

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy.strategy_lab import StrategyLab
from strategy.router import StrategyRouter
from services import message_bus
from model_registry import save_model


def test_init_from_registry(tmp_path):
    async def _run():
        message_bus._message_bus = message_bus.MessageBus(backend="inmemory")
        router = StrategyRouter(
            algorithms={},
            scoreboard_path=tmp_path / "scores.parquet",
            elo_path=tmp_path / "elo.parquet",
            rationale_path=tmp_path / "rationale.parquet",
            regime_perf_path=tmp_path / "regime.parquet",
        )
        payload = {"w": 1}
        save_model("test_policy", payload, {}, tmp_path / "policy.pkl")
        captured = {}

        def train_fn(df: pd.DataFrame, init=None):
            captured["init"] = init
            return lambda msg: 0.0

        lab = StrategyLab(
            train_fn=train_fn,
            router=router,
            thresholds={},
            history_path=tmp_path / "history.csv",
        )
        data = pd.DataFrame({"price": [1, 2, 3]})
        await lab.train_and_deploy("cand", data, policy="test_policy")
        assert captured["init"] == payload
        for task in lab.tasks.values():
            task.cancel()
        await asyncio.sleep(0)

    asyncio.run(_run())
