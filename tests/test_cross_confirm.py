import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest


def prepare_strategy():
    from strategies.baseline import BaselineStrategy

    # short_window < long_window and minimal ATR window for quick priming
    return BaselineStrategy(
        short_window=2,
        long_window=3,
        atr_window=1,
        session_position_limits={},
        default_position_limit=1,
    )


def prime_strategy(strat):
    # Provide enough history so that the next update can generate a signal
    for price in [1.0, 1.0, 1.0]:
        from strategies.baseline import IndicatorBundle

        strat.update(
            price,
            indicators=IndicatorBundle(
                atr=1.0, rsi=50, boll_upper=1e9, boll_lower=-1e9
            ),
            cross_confirm={"PEER": 1.0},
        )


def test_cross_confirm_allows_signal():
    from strategies.baseline import IndicatorBundle

    strat = prepare_strategy()
    prime_strategy(strat)
    sig = strat.update(
        2.0,
        indicators=IndicatorBundle(atr=1.0, rsi=50, boll_upper=1e9, boll_lower=-1e9),
        cross_confirm={"PEER": 0.5},
    )
    assert sig == 1


def test_cross_confirm_blocks_divergence():
    from strategies.baseline import IndicatorBundle

    strat = prepare_strategy()
    prime_strategy(strat)
    sig = strat.update(
        2.0,
        indicators=IndicatorBundle(atr=1.0, rsi=50, boll_upper=1e9, boll_lower=-1e9),
        cross_confirm={"PEER": -0.5},
    )
    assert sig == 0
