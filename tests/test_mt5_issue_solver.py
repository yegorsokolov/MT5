import importlib
import sys
from pathlib import Path

import pytest


@pytest.fixture
def solver_module():
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    import brokers.mt5_direct as mt5_direct
    module = importlib.import_module("brokers.mt5_issue_solver")
    module = importlib.reload(module)
    return module, mt5_direct


def test_solver_known_symbol_issue(solver_module):
    module, mt5_direct = solver_module
    solver = module.MT5IssueSolver()
    error = mt5_direct.MT5Error("unknown symbol", code=4301, details={"request": {"symbol": "EURUSD"}})
    plan = solver.solve(error, create_issue=False)
    assert plan["category"] == "configuration"
    assert any("Market Watch" in step for step in plan["steps"])


def test_solver_volume_pattern(solver_module):
    module, mt5_direct = solver_module
    solver = module.MT5IssueSolver()
    error = mt5_direct.MT5OrderError(
        "MetaTrader5 order_send failed", code=1111, details={"comment": "volume too high", "request": {"symbol": "EURUSD", "volume": 100}}
    )
    plan = solver.solve(error, create_issue=False)
    assert plan["category"] == "risk"
    assert any("volume" in step.lower() for step in plan["steps"])


def test_solver_posts_issue_for_unknown(solver_module):
    module, mt5_direct = solver_module

    class DummyClient:
        def __init__(self):
            self.events = []

        def post_event(self, event, details, severity="info"):
            self.events.append((event, details, severity))
            return "ISSUE-1"

    client = DummyClient()
    solver = module.MT5IssueSolver(issue_client=client)
    error = mt5_direct.MT5Error("strange failure", code=9999)
    plan = solver.solve(error)
    assert plan["issue_id"] == "ISSUE-1"
    assert client.events[0][1]["code"] == 9999

