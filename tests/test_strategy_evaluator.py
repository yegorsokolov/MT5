import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy.router import StrategyRouter
from analysis.strategy_evaluator import StrategyEvaluator


def test_weight_updates_with_synthetic_regimes(tmp_path):
    # Router with two simple algorithms: long and short.
    router = StrategyRouter(
        algorithms={"long": lambda f: 1.0, "short": lambda f: -1.0},
        alpha=0.0,
        scoreboard_path=tmp_path / "score.parquet",
    )

    # Generate synthetic history where long strategy outperforms.
    returns = np.array([0.02] * 25 + [-0.01] * 25)
    history = pd.DataFrame(
        {
            "volatility": [0.1] * len(returns),
            "trend_strength": [0.5] * len(returns),
            "regime": [0] * len(returns),
            "market_basket": [0] * len(returns),
            "return": returns,
        }
    )

    evaluator = StrategyEvaluator(window=len(returns))
    evaluator.evaluate(history, router)

    features = {
        "volatility": 0.1,
        "trend_strength": 0.5,
        "regime": 0,
        "market_basket": 0,
    }
    assert router.select(features) == "long"

    # Live data now favours the short algorithm; ensure weights adapt.
    for _ in range(20):
        router.update(features, 0.05, "short", smooth=0.3)
        router.update(features, -0.05, "long", smooth=0.3)

    assert router.select(features) == "short"
