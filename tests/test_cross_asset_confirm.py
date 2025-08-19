import pandas as pd
import numpy as np
import sys
from pathlib import Path
import importlib.util

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from analysis import cross_asset_confirm as cac

spec = importlib.util.spec_from_file_location(
    "trade_log", ROOT / "data" / "trade_log.py"
)
trade_log = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(trade_log)
TradeLog = trade_log.TradeLog

import risk.trade_manager as tm_mod
TradeManager = tm_mod.TradeManager


def test_cross_asset_confirm_reduces_false_signals():
    np.random.seed(0)
    steps = 200
    base = np.cumsum(np.random.normal(scale=0.1, size=steps))
    related = base + np.random.normal(scale=0.05, size=steps)
    prices = pd.DataFrame({"A": base, "B": related})

    false_raw = 0
    trades_raw = 0
    false_filtered = 0
    trades_filtered = 0
    for t in range(30, steps - 1):
        ret = prices["A"].pct_change().iloc[t]
        next_ret = prices["A"].pct_change().iloc[t + 1]
        if ret > 0:
            trades_raw += 1
            if next_ret < 0:
                false_raw += 1
            score = cac.compute_score(prices.iloc[: t + 1], "A", ["B"])
            if cac.should_open(score):
                trades_filtered += 1
                if next_ret < 0:
                    false_filtered += 1
    assert trades_filtered > 0
    assert false_filtered / trades_filtered < false_raw / trades_raw


def test_trade_manager_confirmation(tmp_path):
    class Exec:
        def __init__(self):
            self.opened = []
            self.closed = []

        def open_order(self, order):
            self.opened.append(order)

        def close_order(self, order_id):
            self.closed.append(order_id)

    exec_client = Exec()
    log = TradeLog(tmp_path / "t.db")
    tm = TradeManager(exec_client, log, atr_period=2, atr_mult=1.0)

    order = {"timestamp": "0", "symbol": "A", "side": "BUY", "volume": 1, "price": 1}
    oid = tm.open_trade(order, confirm_score=0.5)
    assert oid is not None
    assert exec_client.opened

    oid2 = tm.open_trade(order, confirm_score=-0.1)
    assert oid2 is None

    prices = pd.Series([1.0, 1.1, 1.2])
    tm.update_trade(oid, prices, {}, confirm_score=0.5)
    assert not exec_client.closed

    tm.update_trade(oid, prices, {}, confirm_score=-0.5)
    assert exec_client.closed == [oid]

    cur = log.conn.cursor()
    row = cur.execute("SELECT score FROM confirmations WHERE order_id=?", (oid,)).fetchone()
    assert row is not None and abs(row[0] + 0.5) < 1e-9
