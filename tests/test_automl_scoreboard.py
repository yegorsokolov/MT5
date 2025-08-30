import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.automl_scoreboard import build_scoreboard
from strategy.router import StrategyRouter


def test_algorithms_ranked_by_regime(tmp_path):
    regimes = {
        0: np.linspace(0.005, 0.015, 30),
        1: np.linspace(-0.015, -0.005, 30),
    }
    strategies = {
        "long": lambda data: data,
        "short": lambda data: -data,
    }
    path = tmp_path / "scoreboard.parquet"
    df = build_scoreboard(strategies, regimes, path=path)
    assert df.loc[(0, "long"), "sharpe"] > df.loc[(0, "short"), "sharpe"]
    assert df.loc[(1, "short"), "sharpe"] > df.loc[(1, "long"), "sharpe"]

    router = StrategyRouter(
        algorithms={"long": lambda _: 1.0, "short": lambda _: -1.0},
        scoreboard_path=path,
    )
    bull = {
        "volatility": 0.0,
        "trend_strength": 0.0,
        "regime": 0,
        "market_basket": 0,
    }
    bear = {
        "volatility": 0.0,
        "trend_strength": 0.0,
        "regime": 1,
        "market_basket": 1,
    }
    assert router.select(bull) == "long"
    assert router.select(bear) == "short"
