import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from strategies.external_adapter import load_strategy, run_external_strategy


def test_load_backtrader_strategy(tmp_path):
    code = """import backtrader as bt
class TestStrategy(bt.Strategy):
    pass
"""
    f = tmp_path / "bt.py"
    f.write_text(code)
    framework, cls = load_strategy(str(f))
    assert framework == "backtrader"
    import backtrader as bt
    assert issubclass(cls, bt.Strategy)


def test_run_backtrader_strategy(tmp_path):
    code = """import backtrader as bt
class TestStrategy(bt.Strategy):
    def next(self):
        if not self.position:
            self.buy()
        elif len(self) > 1:
            self.sell()
"""
    f = tmp_path / "bt.py"
    f.write_text(code)
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=5, freq="D"),
        "mid": [1, 1.1, 1.2, 1.1, 1.0],
    })
    metrics = run_external_strategy(df, str(f))
    assert set(metrics.keys()) == {
        "sharpe",
        "max_drawdown",
        "total_return",
        "win_rate",
        "sharpe_p_value",
    }

