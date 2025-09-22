import asyncio
import pandas as pd
import pytest

import sys
from pathlib import Path

import types
sys.modules["utils.environment"] = types.SimpleNamespace(ensure_environment=lambda: None)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from mt5 import signal_queue
import risk.position_sizer as ps
from strategies.trade_exit_policy import TradeExitPolicy


def test_training_and_exit_decision():
    df = pd.DataFrame({"feat": [0, 1], "future_return": [-0.2, 0.3]})
    policy = TradeExitPolicy()
    policy.train(df)
    policy.register_trade("A", {"feat": 0})
    assert policy.should_exit("A", {"feat": 0})
    policy.register_trade("B", {"feat": 1})
    assert not policy.should_exit("B", {"feat": 1})


@pytest.mark.asyncio
async def test_exit_policy_integration(monkeypatch):
    def fake_record(name, val, tags=None):
        pass

    monkeypatch.setattr(signal_queue, "record_metric", fake_record)
    monkeypatch.setattr(ps, "record_metric", fake_record)

    class DummyModel:
        import numpy as np

        def predict_proba(self, X):
            return self.np.array([[0.8, 0.2]])  # negative expectation => trigger exit

    policy = TradeExitPolicy(DummyModel())
    policy.register_trade("XYZ", {"prob": 0.6, "confidence": 0.7})

    async with signal_queue.get_async_publisher("tcp://127.0.0.1:6010") as pub, signal_queue.get_async_subscriber(
        "tcp://127.0.0.1:6010"
    ) as sub:
        df = pd.DataFrame({"Timestamp": ["2024"], "Symbol": ["XYZ"], "prob": [0.6], "confidence": [0.7]})
        await signal_queue.publish_dataframe_async(pub, df)
        await asyncio.sleep(0.1)
        gen = signal_queue.iter_messages(sub, exit_policy=policy)
        msg = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert msg["Symbol"] == "XYZ"
        exit_msg = await asyncio.wait_for(gen.__anext__(), timeout=1)
        assert exit_msg["action"] == "close"
        assert exit_msg["Symbol"] == "XYZ"
