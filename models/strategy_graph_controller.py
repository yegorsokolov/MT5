"""Neural policy emitting tiny strategy graphs.

The :class:`StrategyGraphController` combines a small graph neural network with
a policy head that selects between a set of pre-defined strategy graphs.  The
network can therefore be trained using REINFORCE style policy gradients where
the reward is the profit and loss (PnL) obtained by executing the generated
graph on historical market data.

This implementation is intentionally lightweight – it is not intended to be a
production ready trading system but rather a concise, testable example used in
the unit tests.
"""

from __future__ import annotations

from typing import List

import torch
from torch import nn
from torch.optim import Adam
from torch.distributions import Categorical

from models.graph_net import GraphNet
from strategies.graph_dsl import (
    ExitRule,
    Filter,
    Indicator,
    PositionSizer,
    StrategyGraph,
)


class StrategyGraphController(nn.Module):
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
        """Return a :class:`StrategyGraph` for ``action``."""

        if action == 0:
            indicator = Indicator("price", ">", "ma")
        else:
            indicator = Indicator("price", "<", "ma")
        filt = Filter()
        sizer = PositionSizer(1.0)
        exit_rule = ExitRule()
        nodes = {0: indicator, 1: filt, 2: sizer, 3: exit_rule}
        edges = [(0, 1, None), (1, 2, True), (1, 3, False)]
        return StrategyGraph(nodes=nodes, edges=edges)


def train_strategy_graph_controller(
    data: List[dict],
    episodes: int = 100,
    lr: float = 0.1,
    seed: int = 0,
) -> StrategyGraphController:
    """Train :class:`StrategyGraphController` using policy gradients."""

    torch.manual_seed(seed)
    x = torch.zeros((2, 1))
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    model = StrategyGraphController(in_channels=1)
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


__all__ = ["StrategyGraphController", "train_strategy_graph_controller"]

