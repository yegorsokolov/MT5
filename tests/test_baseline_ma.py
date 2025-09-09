import os
import sys

# Ensure repository root on path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from src.modes import Mode
from src.strategy.executor import StrategyExecutor
from src.strategy.registry import get_strategy


@pytest.fixture()
def metadata_file(tmp_path: Path) -> Path:
    return tmp_path / "metadata.json"


def test_baseline_ma_crossover_places_orders(metadata_file: Path):
    metadata_file.write_text(json.dumps({"baseline_ma": {"approved": True}}))
    strategy = get_strategy(short_window=2, long_window=3)
    executor = StrategyExecutor(mode=Mode.LIVE_TRADING, strategy=strategy, metadata_path=metadata_file)
    executor.place_live_order = Mock()

    prices = [1, 2, 3, 2, 1]
    for price in prices:
        tick = {"price": price, "next_price": price}
        executor.on_tick(tick)

    orders = [
        call.args[0]["quantity"]
        for call in executor.place_live_order.call_args_list
        if call.args[0]["quantity"]
    ]
    assert orders == [1, -1]


def test_min_diff_filters_noisy_signals():
    strategy = get_strategy(short_window=2, long_window=3, min_diff=1.0)
    executor = StrategyExecutor(mode=Mode.LIVE_TRADING, strategy=strategy)
    executor.place_live_order = Mock()

    prices = [1, 2, 3, 2.5, 2]
    for price in prices:
        tick = {"price": price, "next_price": price}
        executor.on_tick(tick)

    assert all(
        call.args[0]["quantity"] == 0
        for call in executor.place_live_order.call_args_list
    )
