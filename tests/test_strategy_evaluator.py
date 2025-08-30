import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy.router import StrategyRouter
from analysis.strategy_evaluator import StrategyEvaluator


def test_instrument_aware_weight_updates(tmp_path):
    router = StrategyRouter(
        algorithms={"long": lambda f: 1.0, "short": lambda f: -1.0},
        alpha=0.0,
    )

    returns_a = np.array([0.02] * 25 + [-0.01] * 25)
    returns_b = np.array([-0.02] * 25 + [-0.01] * 25)
    history = pd.DataFrame(
        {
            "volatility": [0.1] * 100,
            "trend_strength": [0.5] * 100,
            "regime": [0] * 100,
            "market_basket": [0] * 100,
            "instrument": ["A"] * 50 + ["B"] * 50,
            "return": np.concatenate([returns_a, returns_b]),
        }
    )

    evaluator = StrategyEvaluator(window=50)
    evaluator.evaluate(history, router)

    features_a = {
        "volatility": 0.1,
        "trend_strength": 0.5,
        "regime": 0,
        "market_basket": 0,
        "instrument": "A",
    }
    features_b = {**features_a, "instrument": "B"}

    assert router.select(features_a) == "long"
    assert router.select(features_b) == "short"

    for _ in range(20):
        router.update(features_a, 0.05, "short", smooth=0.3)
        router.update(features_a, -0.05, "long", smooth=0.3)

    assert router.select(features_a) == "short"
    assert router.select(features_b) == "short"

