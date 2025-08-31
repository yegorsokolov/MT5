import sys
import types
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub out heavy dependencies before importing the router
sys.modules.setdefault(
    "state_manager",
    types.SimpleNamespace(load_router_state=lambda: {}, save_router_state=lambda *a, **k: None),
)
sys.modules.setdefault(
    "analysis.algorithm_rating", types.SimpleNamespace(load_ratings=lambda *a, **k: types.SimpleNamespace(to_dict=lambda: {}))
)
sys.modules.setdefault(
    "analysis.rationale_scorer", types.SimpleNamespace(load_algorithm_win_rates=lambda *a, **k: types.SimpleNamespace(to_dict=lambda: {}))
)
sys.modules.setdefault(
    "analytics.regime_performance_store", types.SimpleNamespace(RegimePerformanceStore=lambda *a, **k: types.SimpleNamespace(get=lambda *a, **k: 0.0))
)
sys.modules.setdefault(
    "utils.resource_monitor", types.SimpleNamespace(monitor=types.SimpleNamespace(capability_tier="lite"))
)
class _DF(list):
    def to_dict(self, orient="records"):
        return list(self)

class _MI:
    @staticmethod
    def from_tuples(tuples, names=None):
        return []

sys.modules["pandas"] = types.SimpleNamespace(
    DataFrame=lambda data=None, columns=None, index=None: _DF(data or []),
    MultiIndex=_MI,
    read_parquet=lambda *a, **k: _DF(),
)

from strategy.router import StrategyRouter


def test_ftrl_router_on_lite_tier():
    router = StrategyRouter(
        algorithms={"a": lambda f: 1.0, "b": lambda f: -1.0},
        alpha=0.0,
    )
    assert router.use_ftrl
    features = {"volatility": 1.0, "trend_strength": 0.0, "regime": 0.0}
    rewards = {"a": 1.0, "b": -1.0}
    for _ in range(20):
        name = router.select(features)
        router.update(features, rewards[name], name)
    assert router.select(features) == "a"
