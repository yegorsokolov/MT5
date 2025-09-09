import os
import sys

# Ensure repository root is on sys.path when tests are executed in isolation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.main import main
from src.modes import Mode
from src.strategy.executor import StrategyExecutor


@pytest.fixture()
def metadata_file(tmp_path: Path) -> Path:
    return tmp_path / "metadata.json"


def _executor_for(mode: Mode, approved: bool, metadata_path: Path) -> StrategyExecutor:
    metadata_path.write_text(json.dumps({"strat": {"approved": approved}}))
    return StrategyExecutor(mode=mode, strategy={"name": "strat"}, metadata_path=metadata_path)


def _strategy():
    generate = Mock(return_value={"quantity": 1})
    update = Mock()
    return {"name": "dummy", "generate_order": generate, "update": update}


def test_main_passes_mode_to_executor(metadata_file: Path):
    executor = main(["--mode", Mode.PAPER_TRADING.value])
    assert executor.mode is Mode.PAPER_TRADING


def test_live_orders_blocked_in_training_mode(metadata_file: Path):
    executor = _executor_for(Mode.TRAINING, approved=True, metadata_path=metadata_file)
    with pytest.raises(PermissionError):
        executor.place_live_order(order={})


def test_live_orders_blocked_in_paper_mode(metadata_file: Path):
    executor = _executor_for(Mode.PAPER_TRADING, approved=True, metadata_path=metadata_file)
    with pytest.raises(PermissionError):
        executor.place_live_order(order={})


def test_live_orders_blocked_when_not_approved(metadata_file: Path):
    executor = _executor_for(Mode.LIVE_TRADING, approved=False, metadata_path=metadata_file)
    with pytest.raises(PermissionError):
        executor.place_live_order(order={})


def test_live_orders_allowed_in_live_mode_with_approval(metadata_file: Path):
    executor = _executor_for(Mode.LIVE_TRADING, approved=True, metadata_path=metadata_file)
    assert executor.place_live_order(order={}) == "order_placed"


def test_on_tick_updates_without_placing_in_training_mode(metadata_file: Path):
    strat = _strategy()
    metadata_file.write_text(json.dumps({"dummy": {"approved": True}}))
    executor = StrategyExecutor(mode=Mode.TRAINING, strategy=strat, metadata_path=metadata_file)
    executor.place_live_order = Mock(side_effect=RuntimeError("should not place"))
    tick = {"price": 100, "next_price": 105}
    executor.on_tick(tick)
    strat["generate_order"].assert_called_once_with(tick)
    strat["update"].assert_called_once_with({"quantity": 1}, 5)
    executor.place_live_order.assert_not_called()


def test_on_tick_places_and_updates_in_live_mode(metadata_file: Path):
    strat = _strategy()
    metadata_file.write_text(json.dumps({"dummy": {"approved": True}}))
    executor = StrategyExecutor(mode=Mode.LIVE_TRADING, strategy=strat, metadata_path=metadata_file)
    executor.place_live_order = Mock(return_value="order_placed")
    tick = {"price": 100, "next_price": 101}
    executor.on_tick(tick)
    executor.place_live_order.assert_called_once_with({"quantity": 1})
    strat["update"].assert_called_once_with({"quantity": 1}, 1)


def test_on_tick_updates_without_placing_when_not_approved(metadata_file: Path):
    strat = _strategy()
    metadata_file.write_text(json.dumps({"dummy": {"approved": False}}))
    executor = StrategyExecutor(mode=Mode.LIVE_TRADING, strategy=strat, metadata_path=metadata_file)
    executor.place_live_order = Mock(side_effect=RuntimeError("should not place"))
    tick = {"price": 100, "next_price": 99}
    executor.on_tick(tick)
    strat["generate_order"].assert_called_once_with(tick)
    strat["update"].assert_called_once_with({"quantity": 1}, -1)
    executor.place_live_order.assert_not_called()
