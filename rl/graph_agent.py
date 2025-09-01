"""Lightweight graph-based reinforcement learning agent."""

from __future__ import annotations

from typing import List

import os
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import numpy as np
import torch
import torch.nn.functional as F

try:  # handle importing when package is not installed
    from models.graph_net import GraphNet
except Exception:  # pragma: no cover - fallback for direct file import
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "graph_net", Path(__file__).resolve().parents[1] / "models" / "graph_net.py"
    )
    graph_net_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph_net_mod)  # type: ignore
    GraphNet = graph_net_mod.GraphNet  # type: ignore


def _adjacency_to_edge_index(mat: np.ndarray) -> torch.Tensor:
    """Convert adjacency matrix to COO edge index tensor."""
    rows, cols = np.nonzero(mat)
    edge_index = np.vstack([rows, cols])
    return torch.tensor(edge_index, dtype=torch.long)


class Transition:
    """Simple container for experience tuples."""

    def __init__(self, x: np.ndarray, adj: np.ndarray, action: int, reward: float) -> None:
        self.x = x
        self.adj = adj
        self.action = action
        self.reward = reward


class GraphAgent:
    """Minimal policy gradient agent using :class:`GraphNet` embeddings."""

    def __init__(self, in_features: int, hidden_dim: int, num_actions: int, lr: float = 0.01) -> None:
        self.gnn = GraphNet(in_features, hidden_dim, hidden_dim)
        self.policy = torch.nn.Linear(hidden_dim, num_actions)
        self.opt = torch.optim.Adam(list(self.gnn.parameters()) + list(self.policy.parameters()), lr=lr)
        self.memory: List[Transition] = []

    def act(self, x: np.ndarray, adj: np.ndarray) -> int:
        """Return greedy action given features ``x`` and adjacency ``adj``."""
        edge_index = _adjacency_to_edge_index(adj)
        h = self.gnn(torch.tensor(x, dtype=torch.float), edge_index)
        g = h.mean(dim=0)
        logits = self.policy(g)
        return int(torch.argmax(logits).item())

    def store(self, x: np.ndarray, adj: np.ndarray, action: int, reward: float) -> None:
        self.memory.append(Transition(x, adj, action, reward))

    def train(self) -> None:
        if not self.memory:
            return
        losses = []
        for tr in self.memory:
            edge_index = _adjacency_to_edge_index(tr.adj)
            h = self.gnn(torch.tensor(tr.x, dtype=torch.float), edge_index)
            g = h.mean(dim=0)
            logits = self.policy(g)
            logp = F.log_softmax(logits, dim=-1)[tr.action]
            losses.append(-logp * tr.reward)
        loss = torch.stack(losses).mean()
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self.memory.clear()


__all__ = ["GraphAgent"]

