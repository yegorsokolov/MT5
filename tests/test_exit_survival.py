import importlib.util
from pathlib import Path
import pandas as pd
import numpy as np
import sys

spec = importlib.util.spec_from_file_location("exit_survival", Path(__file__).resolve().parents[1] / "models" / "exit_survival.py")
exit_survival = importlib.util.module_from_spec(spec)
sys.modules["exit_survival"] = exit_survival
assert spec.loader is not None
spec.loader.exec_module(exit_survival)
ExitSurvivalModel = exit_survival.ExitSurvivalModel
from risk.trade_manager import TradeManager
spec_tl = importlib.util.spec_from_file_location("trade_log", Path(__file__).resolve().parents[1] / "data" / "trade_log.py")
trade_log = importlib.util.module_from_spec(spec_tl)
assert spec_tl.loader is not None
spec_tl.loader.exec_module(trade_log)
TradeLog = trade_log.TradeLog


def sharpe(returns):
    arr = np.array(returns, dtype=float)
    return arr.mean() / (arr.std(ddof=1) + 1e-9)


def test_survival_model_fit_predict():
    df = pd.DataFrame({
        'age': [1, 2, 1, 2],
        'regime': [0, 0, 1, 1],
        'survived': [0, 0, 1, 1],
    })
    model = ExitSurvivalModel().fit(df)
    prob = model.predict_survival({'age': 1, 'regime': 1})
    assert 0 <= prob <= 1


def test_trade_manager_survival_integration(tmp_path):
    class DummyExecution:
        def __init__(self):
            self.closed = []

        def update_order(self, order_id, stop_loss, take_profit):
            pass

        def close_order(self, order_id):
            self.closed.append(order_id)

    class DummySurvival:
        def predict_survival(self, feat):
            # quickly decay for regime 0, stay high for regime 1
            if feat['regime'] == 0 and feat['age'] >= 3:
                return 0.1
            return 0.9

    exec_client = DummyExecution()
    log = TradeLog(tmp_path / 'trades.db')
    tm = TradeManager(exec_client, log, survival_model=DummySurvival(), survival_threshold=0.2)

    prices1 = pd.Series([100, 99, 98, 97, 96])
    features1 = {'regime': 0, 'volatility': 1.0}
    tm.update_trade(1, prices1[:3], features1)
    assert 1 in exec_client.closed
    assert log.get_survival(1)[0] == 0.1

    prices2 = pd.Series([100, 101, 102, 103, 104])
    features2 = {'regime': 1, 'volatility': 1.0}
    tm.update_trade(2, prices2, features2)
    assert 2 not in exec_client.closed

    baseline = [prices1.iloc[-1] - prices1.iloc[0], prices2.iloc[-1] - prices2.iloc[0]]
    improved = [prices1.iloc[2] - prices1.iloc[0], prices2.iloc[-1] - prices2.iloc[0]]
    assert sharpe(improved) > sharpe(baseline)
