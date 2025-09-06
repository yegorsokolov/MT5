import numpy as np

from rl.meta_controller import MetaControllerDataset, train_meta_controller
from strategy.router import StrategyRouter


def test_meta_controller_routing():
    # synthetic regimes 0 then 1
    regimes = np.concatenate([np.zeros(50), np.ones(50)])
    returns = np.column_stack([
        np.where(regimes == 0, 1.0, -1.0),
        np.where(regimes == 0, -1.0, 1.0),
    ])
    states = np.stack([
        np.zeros_like(regimes),
        np.zeros_like(regimes),
        regimes,
    ], axis=1)
    dataset = MetaControllerDataset(returns, states)
    controller = train_meta_controller(dataset, epochs=200)

    # base RL agents
    def agent_a(features):
        return 1.0 if features.get("regime", 0) == 0 else -1.0

    def agent_b(features):
        return -1.0 if features.get("regime", 0) == 0 else 1.0

    router = StrategyRouter(rl_agents={"a": agent_a, "b": agent_b}, meta_controller=controller, algorithms={})

    # regime 0 should favour agent_a
    f0 = {"volatility": 0.0, "trend_strength": 0.0, "regime": 0.0}
    weights0 = controller.predict(np.array([agent_a(f0), agent_b(f0)]), np.array([0.0, 0.0, 0.0]))
    assert weights0[0] > weights0[1]
    action0 = router.algorithms["rl_policy"](f0)
    assert action0 > 0.5

    # regime 1 should favour agent_b
    f1 = {"volatility": 0.0, "trend_strength": 0.0, "regime": 1.0}
    weights1 = controller.predict(np.array([agent_a(f1), agent_b(f1)]), np.array([0.0, 0.0, 1.0]))
    assert weights1[1] > weights1[0]
    action1 = router.algorithms["rl_policy"](f1)
    assert action1 > 0.5
