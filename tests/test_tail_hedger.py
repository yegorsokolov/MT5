import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.modules["prometheus_client"] = types.SimpleNamespace(
    Counter=lambda *a, **k: None,
    Gauge=lambda *a, **k: None,
)
from mt5.risk_manager import RiskManager
from risk.tail_hedger import TailHedger


def test_tail_hedger_triggers_on_var_spike():
    rm = RiskManager(max_drawdown=1e9, max_var=1e9)
    hedger = TailHedger(rm, var_threshold=5.0)
    rm.attach_tail_hedger(hedger)

    rm.update("bot", -10.0, 100.0)

    assert len(hedger.hedges) == 1
    assert rm.metrics.exposure == 0.0


def test_tail_hedger_no_trigger_below_threshold():
    rm = RiskManager(max_drawdown=1e9, max_var=1e9)
    hedger = TailHedger(rm, var_threshold=20.0)
    rm.attach_tail_hedger(hedger)

    rm.update("bot", -10.0, 100.0)

    assert len(hedger.hedges) == 0
    assert rm.metrics.exposure == 100.0
