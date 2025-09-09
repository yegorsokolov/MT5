import os
import sys

# Ensure repository root is on sys.path when tests are executed in isolation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from src.main import main
from src.modes import Mode
from src.strategy.executor import StrategyExecutor


def _executor_for(mode: Mode, approved: bool) -> StrategyExecutor:
    return StrategyExecutor(mode=mode, strategy={"approved": approved})


def test_main_passes_mode_to_executor():
    executor = main(["--mode", Mode.PAPER_TRADING.value])
    assert executor.mode is Mode.PAPER_TRADING


def test_live_orders_blocked_in_training_mode():
    executor = _executor_for(Mode.TRAINING, approved=True)
    with pytest.raises(PermissionError):
        executor.place_live_order(order={})


def test_live_orders_blocked_in_paper_mode():
    executor = _executor_for(Mode.PAPER_TRADING, approved=True)
    with pytest.raises(PermissionError):
        executor.place_live_order(order={})


def test_live_orders_blocked_when_not_approved():
    executor = _executor_for(Mode.LIVE_TRADING, approved=False)
    with pytest.raises(PermissionError):
        executor.place_live_order(order={})


def test_live_orders_allowed_in_live_mode_with_approval():
    executor = _executor_for(Mode.LIVE_TRADING, approved=True)
    assert executor.place_live_order(order={}) == "order_placed"
