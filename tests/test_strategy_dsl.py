import sys
from pathlib import Path

# Ensure project root on path for direct execution
sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy_dsl import Buy, Indicator, Sell, StrategyInterpreter
from models.strategy_controller import (
    evaluate_controller,
    train_strategy_controller,
)


def test_dsl_execution():
    program = [
        Indicator("price", "<", "ma"),
        Buy(),
        Indicator("price", ">", "ma"),
        Sell(),
    ]
    data = [
        {"price": 1.0, "ma": 2.0},
        {"price": 3.0, "ma": 2.0},
        {"price": 2.0, "ma": 2.0},
    ]
    interp = StrategyInterpreter(program)
    pnl = interp.run(data)
    assert pnl == 2.0


def test_controller_generates_strategy():
    market_data = [
        {"price": 1.0, "ma": 2.0},
        {"price": 3.0, "ma": 2.0},
        {"price": 1.0, "ma": 2.0},
    ]
    controller = train_strategy_controller(market_data, epochs=200)
    pnl = evaluate_controller(controller, market_data)
    assert pnl > 0
