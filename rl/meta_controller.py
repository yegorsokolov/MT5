"""Simple meta-controller to allocate between multiple RL agents.

The controller is trained on logged sub-policy returns and regime/state
embeddings.  Given the recent performance of each base agent and the current
state embedding, the controller outputs allocation weights that can be used to
blend agent actions or to enable/disable agents.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

try:  # optional dependency - torch may be stubbed in minimal environments
    import torch
    from torch import nn
except Exception:  # pragma: no cover - torch not installed in some tests
    torch = None  # type: ignore
    nn = None  # type: ignore


if nn is not None:
    class _MLP(nn.Module):  # pragma: no cover - trivial wrapper
        def __init__(self, in_dim: int, hidden: Iterable[int], out_dim: int) -> None:
            super().__init__()
            layers = []
            last = in_dim
            for h in hidden:
                layers.append(nn.Linear(last, h))
                layers.append(nn.ReLU())
                last = h
            layers.append(nn.Linear(last, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return self.net(x)
else:  # pragma: no cover - torch not available
    class _MLP:  # type: ignore
        def __init__(self, *_, **__):
            pass

        def forward(self, x):  # pragma: no cover
            raise RuntimeError("torch is required for MetaController")


class MetaController(nn.Module if nn is not None else object):
    """A small network that outputs allocation weights for base agents."""

    def __init__(
        self,
        num_agents: int,
        state_dim: int,
        hidden: Optional[Iterable[int]] = None,
    ) -> None:
        self.num_agents = num_agents
        self.state_dim = state_dim
        if torch is None:
            # simple lookup table for states -> best agent
            self.state_to_agent: dict[tuple, int] = {}
        else:
            super().__init__()
            if hidden is None:
                hidden = (32, 16)
            self.mlp = _MLP(num_agents + state_dim, hidden, num_agents)

    def forward(
        self, returns: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:  # type: ignore[override]
        """Return unnormalised logits for agent weights."""
        if returns.dim() == 1:
            returns = returns.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        x = torch.cat([returns, state], dim=-1)
        return self.mlp(x)

    def predict(self, returns: np.ndarray, state: np.ndarray) -> np.ndarray:
        """Predict allocation weights for ``returns`` and ``state``.

        Parameters
        ----------
        returns:
            Array of shape ``(num_agents,)`` containing estimated returns from
            each base agent.
        state:
            Array of shape ``(state_dim,)`` representing the current state
            embedding.
        """
        if torch is None:
            key = tuple(np.asarray(state).tolist())
            idx = self.state_to_agent.get(key, int(np.argmax(returns)))
            weights = np.zeros(self.num_agents, dtype=float)
            weights[idx] = 1.0
            return weights
        with torch.no_grad():
            r_t = torch.as_tensor(returns, dtype=torch.float32)
            s_t = torch.as_tensor(state, dtype=torch.float32)
            logits = self.forward(r_t, s_t)
            weights = torch.softmax(logits, dim=-1)
        return weights.cpu().numpy().squeeze()

    # ------------------------------------------------------------------
    def select(
        self, returns: np.ndarray, state: np.ndarray, threshold: float = 0.5
    ) -> np.ndarray:
        """Return a boolean mask indicating which agents to enable.

        Parameters
        ----------
        returns:
            Estimated returns from each agent.
        state:
            State embedding aligned with ``returns``.
        threshold:
            Minimum allocation weight required for an agent to be enabled.
        """
        weights = self.predict(returns, state)
        return weights >= threshold


@dataclass
class MetaControllerDataset:
    """Container holding logged returns and state embeddings."""

    returns: np.ndarray  # shape (N, num_agents)
    states: np.ndarray  # shape (N, state_dim)


def train_meta_controller(
    dataset: MetaControllerDataset, epochs: int = 100, lr: float = 1e-3
) -> MetaController:
    """Train a :class:`MetaController` on the given dataset.

    The training objective is to pick the agent with the highest observed
    return.  A small neural network is trained with a cross-entropy loss where
    the target class is the index of the best-performing agent for each sample.
    """
    if torch is None:  # pragma: no cover - simple fallback without torch
        model = MetaController(dataset.returns.shape[1], dataset.states.shape[1])
        for r, s in zip(dataset.returns, dataset.states):
            model.state_to_agent[tuple(s.tolist())] = int(np.argmax(r))
        return model

    returns = torch.as_tensor(dataset.returns, dtype=torch.float32)
    states = torch.as_tensor(dataset.states, dtype=torch.float32)
    targets = returns.argmax(dim=1)

    model = MetaController(dataset.returns.shape[1], dataset.states.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for _ in range(max(1, epochs)):
        opt.zero_grad()
        logits = model(returns, states)
        loss = loss_fn(logits, targets)
        loss.backward()
        opt.step()

    return model


__all__ = ["MetaController", "MetaControllerDataset", "train_meta_controller"]
