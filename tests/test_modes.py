import os
import sys

# Ensure repository root is on sys.path when tests are executed in isolation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
from unittest.mock import Mock

from src.main import main
from src.modes import Mode
from src.strategy.executor import StrategyExecutor


def _executor_for(mode: Mode, approved: bool) -> StrategyExecutor:
    return StrategyExecutor(mode=mode, strategy={"approved": approved})


def _strategy(approved: bool):
    generate = Mock(return_value={"quantity": 1})
    update = Mock()
    return {"approved": approved, "generate_order": generate, "update": update}


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


def test_on_tick_updates_without_placing_in_training_mode():
    strat = _strategy(True)
    executor = StrategyExecutor(mode=Mode.TRAINING, strategy=strat)
    executor.place_live_order = Mock(side_effect=RuntimeError("should not place"))
    tick = {"price": 100, "next_price": 105}
    executor.on_tick(tick)
    strat["generate_order"].assert_called_once_with(tick)
    strat["update"].assert_called_once_with({"quantity": 1}, 5)
    executor.place_live_order.assert_not_called()


def test_on_tick_places_and_updates_in_live_mode():
    strat = _strategy(True)
    executor = StrategyExecutor(mode=Mode.LIVE_TRADING, strategy=strat)
    executor.place_live_order = Mock(return_value="order_placed")
    tick = {"price": 100, "next_price": 101}
    executor.on_tick(tick)
    executor.place_live_order.assert_called_once_with({"quantity": 1})
    strat["update"].assert_called_once_with({"quantity": 1}, 1)


def test_on_tick_updates_without_placing_when_not_approved():
    strat = _strategy(False)
    executor = StrategyExecutor(mode=Mode.LIVE_TRADING, strategy=strat)
    executor.place_live_order = Mock(side_effect=RuntimeError("should not place"))
    tick = {"price": 100, "next_price": 99}
    executor.on_tick(tick)
    strat["generate_order"].assert_called_once_with(tick)
    strat["update"].assert_called_once_with({"quantity": 1}, -1)
    executor.place_live_order.assert_not_called()
