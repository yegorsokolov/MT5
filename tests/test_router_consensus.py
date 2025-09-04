import sys
import logging
import types
from pathlib import Path
import pandas as pd

# Ensure project root is on path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Stub heavy dependencies before importing the router
sys.modules.setdefault(
    "state_manager",
    types.SimpleNamespace(load_router_state=lambda: {}, save_router_state=lambda *a, **k: None),
)
sys.modules.setdefault(
    "analysis.algorithm_rating",
    types.SimpleNamespace(load_ratings=lambda *a, **k: types.SimpleNamespace(to_dict=lambda: {})),
)
sys.modules.setdefault(
    "analysis.rationale_scorer",
    types.SimpleNamespace(load_algorithm_win_rates=lambda *a, **k: {}),
)
sys.modules.setdefault(
    "analytics.regime_performance_store",
    types.SimpleNamespace(
        RegimePerformanceStore=lambda *a, **k: types.SimpleNamespace(load=lambda: pd.DataFrame())
    ),
)
sys.modules.setdefault(
    "analytics.metrics_aggregator",
    types.SimpleNamespace(record_metric=lambda *a, **k: None),
)
sys.modules.setdefault(
    "utils.resource_monitor", types.SimpleNamespace(monitor=None))
sys.modules.setdefault(
    "strategy.pair_trading", types.SimpleNamespace(signal_from_features=lambda f: 0.0)
)

from strategy.router import StrategyRouter


def _base_features():
    return {"volatility": 0.0, "trend_strength": 0.0, "regime": 0.0}


def test_consensus_blocks_conflicting_signals(caplog):
    router = StrategyRouter(
        algorithms={"a": lambda f: 1.0, "b": lambda f: -1.0},
        alpha=0.0,
        consensus_threshold=0.6,
    )
    features = _base_features()
    signals = {n: alg(features) for n, alg in router.algorithms.items()}
    score, _ = router.consensus.score(signals)
    assert score < router.consensus_threshold
    with caplog.at_level(logging.INFO):
        name, action = router.act(features)
    assert action == 0.0
    assert any("Consensus score" in rec.message for rec in caplog.records)


def test_consensus_allows_aligned_signals(caplog):
    router = StrategyRouter(
        algorithms={"a": lambda f: 1.0, "b": lambda f: 0.8, "c": lambda f: 0.9},
        alpha=0.0,
        consensus_threshold=0.6,
    )
    features = _base_features()
    signals = {n: alg(features) for n, alg in router.algorithms.items()}
    score, _ = router.consensus.score(signals)
    assert score >= router.consensus_threshold
    with caplog.at_level(logging.INFO):
        name, action = router.act(features)
    assert action == signals[name]
    assert any("Consensus score" in rec.message for rec in caplog.records)
