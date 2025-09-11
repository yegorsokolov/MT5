import importlib.util
from pathlib import Path
import sys

import pandas as pd

spec_tm = importlib.util.spec_from_file_location(
    "trade_manager", Path(__file__).resolve().parents[1] / "risk" / "trade_manager.py"
)
trade_manager = importlib.util.module_from_spec(spec_tm)
sys.modules["trade_manager"] = trade_manager
assert spec_tm.loader is not None
spec_tm.loader.exec_module(trade_manager)
TradeManager = trade_manager.TradeManager


class DummyExecution:
    def __init__(self):
        self.closed = []
        self.updated = []

    def update_order(self, order_id, stop_loss, take_profit):
        self.updated.append((order_id, stop_loss, take_profit))

    def close_order(self, order_id):
        self.closed.append(order_id)


class DummyLog:
    def __init__(self):
        self.survival = []
        self.thresholds = []

    def record_survival(self, order_id, prob):
        self.survival.append((order_id, prob))

    def record_thresholds(self, order_id, base_tp, base_sl, adaptive_tp, adaptive_sl):
        self.thresholds.append((order_id, base_tp, base_sl, adaptive_tp, adaptive_sl))


class StubSurvival:
    def __init__(self, probs):
        self.probs = list(probs)
        self.i = 0

    def predict_survival(self, features):
        p = self.probs[self.i]
        self.i += 1
        return p


def test_exit_when_survival_prob_low():
    prices = pd.Series([100, 101, 102, 103, 104])
    features = {}

    surv = StubSurvival([0.9, 0.1])
    exec_client = DummyExecution()
    log = DummyLog()
    tm = TradeManager(
        exec_client,
        log,
        survival_model=surv,
        survival_threshold=0.2,
        atr_period=2,
    )

    res_high = tm.update_trade(1, prices, features)
    assert exec_client.closed == []
    wide_width = res_high["adaptive_tp"] - res_high["adaptive_sl"]

    res_low = tm.update_trade(1, prices, features)
    assert exec_client.closed == [1]
    tight_width = res_low["adaptive_tp"] - res_low["adaptive_sl"]

    assert tight_width < wide_width
    assert log.survival[0][1] == 0.9
    assert log.survival[1][1] == 0.1
