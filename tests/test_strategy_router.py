import sys
from pathlib import Path

# Remove stubbed SciPy modules so real implementations load
for mod in ["scipy", "scipy.stats"]:
    sys.modules.pop(mod, None)

sys.path.append(str(Path(__file__).resolve().parents[1]))

from strategy.router import StrategyRouter


def test_algorithm_switching_across_regimes():
    # Initialise router with dummy algorithms that simply return constant values.
    router = StrategyRouter(
        algorithms={
            "mean_reversion": lambda f: -1.0,
            "trend_following": lambda f: 1.0,
        },
        alpha=0.1,
    )

    # Regime with strong positive trend should favour trend-following strategy.
    features_trend = {"volatility": 0.2, "trend_strength": 1.0, "regime": 1.0}
    reward_map_trend = {
        "trend_following": 1.0,
        "mean_reversion": -1.0,
    }
    for _ in range(20):
        name = router.select(features_trend)
        reward = reward_map_trend[name]
        router.update(features_trend, reward, name)
    assert router.select(features_trend) == "trend_following"

    # Switch to mean-reverting regime.
    features_revert = {"volatility": 0.2, "trend_strength": -1.0, "regime": -1.0}
    reward_map_revert = {
        "trend_following": -1.0,
        "mean_reversion": 1.0,
    }
    for _ in range(20):
        name = router.select(features_revert)
        reward = reward_map_revert[name]
        router.update(features_revert, reward, name)
    assert router.select(features_revert) == "mean_reversion"
