"""Neural policy for searching simple strategy graphs.

The :class:`StrategySearchNet` combines a tiny graph neural network with a
policy head producing discrete actions.  Each action corresponds to a very small
predefined strategy graph.  The network can therefore be trained with
REINFORCE-style policy gradients where the reward is the PnL obtained by
executing the generated graph on historical data.

The implementation here is intentionally lightweight – it is *not* intended to
be a production ready optimisation engine.  The goal is merely to provide a
concise, testable example of how such a component could be structured.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical

from models.graph_net import GraphNet
from strategies.graph_dsl import IndicatorNode, PositionNode, RiskNode, StrategyGraph


class StrategySearchNet(nn.Module):
    """Graph neural network with a categorical policy head."""

    def __init__(self, in_channels: int = 1, hidden: int = 16, actions: int = 2):
        super().__init__()
        self.gnn = GraphNet(
            in_channels=in_channels,
            hidden_channels=hidden,
            out_channels=hidden,
            num_layers=2,
        )
        self.policy = nn.Linear(hidden, actions)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = self.gnn(x, edge_index)
        pooled = h.mean(dim=0)
        return self.policy(pooled)

    def sample(self, x: torch.Tensor, edge_index: torch.Tensor):
        logits = self.forward(x, edge_index)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action)

    # ------------------------------------------------------------------
    # Graph helpers
    def build_graph(self, action: int) -> StrategyGraph:
        """Return a :class:`StrategyGraph` for ``action``.

        Action ``0`` corresponds to going long when ``price > ma`` while action
        ``1`` uses the opposite indicator.  Both use a unit risk and position
        size.
        """

        if action == 0:
            indicator = IndicatorNode("price", ">", "ma")
        else:
            indicator = IndicatorNode("price", "<", "ma")
        risk = RiskNode(1.0)
        position = PositionNode(1.0)
        nodes = {0: indicator, 1: risk, 2: position}
        edges = [(0, 1, None), (1, 2, True)]
        return StrategyGraph(nodes=nodes, edges=edges)


def train_strategy_search(
    data: List[dict],
    episodes: int = 100,
    lr: float = 0.01,
    seed: int = 0,
) -> StrategySearchNet:
    """Train :class:`StrategySearchNet` using policy gradients."""

    torch.manual_seed(seed)
    x = torch.zeros((2, 1))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    model = StrategySearchNet(in_channels=1)
    optim = Adam(model.parameters(), lr=lr)

    for _ in range(episodes):
        action, log_prob = model.sample(x, edge_index)
        graph = model.build_graph(int(action.item()))
        reward = graph.run(data)
        loss = -log_prob * reward
        optim.zero_grad()
        loss.backward()
        optim.step()

    return model


__all__ = ["StrategySearchNet", "train_strategy_search"]

