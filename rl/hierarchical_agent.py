"""Simple hierarchical reinforcement learning agent.

A high-level manager chooses among multiple low-level policies.  All
policies share a replay buffer so they can be trained jointly.  The
implementation here is intentionally lightweight and framework agnostic so it
can be used in tests without heavy dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Any
import random


@dataclass
class Transition:
    """Single transition stored in the replay buffer."""

    obs: Any
    manager_action: str
    worker_action: Any
    reward: float
    next_obs: Any
    done: bool


class ReplayBuffer:
    """A very small replay buffer suitable for unit tests."""

    def __init__(self, capacity: int = 10000) -> None:
        self.capacity = capacity
        self.buffer: List[Transition] = []

    def add(self, transition: Transition) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Transition]:
        if not self.buffer:
            return []
        return random.sample(self.buffer, min(len(self.buffer), batch_size))

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)


class BasePolicy:
    """Interface for low level policies."""

    def act(self, obs: Any) -> Any:  # pragma: no cover - interface method
        raise NotImplementedError

    def update(self, batch: Iterable[Transition]) -> None:  # pragma: no cover
        pass


class ConstantPolicy(BasePolicy):
    """A policy returning a constant action."""

    def __init__(self, action: Any) -> None:
        self.action = action
        self.updates = 0

    def act(self, obs: Any) -> Any:  # pragma: no cover - trivial
        return self.action

    def update(self, batch: Iterable[Transition]) -> None:
        self.updates += 1


class TrendPolicy(ConstantPolicy):
    """Always goes with the trend (positive action)."""

    def __init__(self) -> None:
        super().__init__(1.0)


class MeanReversionPolicy(ConstantPolicy):
    """Always goes against the trend (negative action)."""

    def __init__(self) -> None:
        super().__init__(-1.0)


class EpsilonGreedyManager:
    """Selects a policy using a simple epsilon-greedy strategy."""

    def __init__(self, policies: List[str], epsilon: float = 0.1) -> None:
        self.policies = list(policies)
        self.epsilon = epsilon
        self.q_values: Dict[str, float] = {p: 0.0 for p in self.policies}
        self.updated = False

    def select_policy(self, obs: Any) -> str:
        if random.random() < self.epsilon:
            return random.choice(self.policies)
        # choose policy with highest estimated value
        return max(self.q_values, key=self.q_values.get)

    def update(self, batch: Iterable[Transition]) -> None:
        self.updated = True
        for tr in batch:
            q = self.q_values[tr.manager_action]
            self.q_values[tr.manager_action] = q + 0.1 * (tr.reward - q)


class HierarchicalAgent:
    """Coordinator between a manager and multiple worker policies."""

    def __init__(self, manager: Any, workers: Dict[str, BasePolicy]) -> None:
        self.manager = manager
        self.workers = workers
        self.replay_buffer = ReplayBuffer()

    def act(self, obs: Any) -> Dict[str, Any]:
        policy_key = self.manager.select_policy(obs)
        action = self.workers[policy_key].act(obs)
        return {"manager": policy_key, "worker": action}

    def store(
        self,
        obs: Any,
        action: Dict[str, Any],
        reward: float,
        next_obs: Any,
        done: bool,
    ) -> None:
        tr = Transition(obs, action["manager"], action["worker"], reward, next_obs, done)
        self.replay_buffer.add(tr)

    def train(self, batch_size: int = 32) -> None:
        batch = self.replay_buffer.sample(batch_size)
        if not batch:
            return
        # update manager with whole batch
        if hasattr(self.manager, "update"):
            self.manager.update(batch)
        # update each worker with the transitions where it was selected
        for key, policy in self.workers.items():
            sub_batch = [tr for tr in batch if tr.manager_action == key]
            if sub_batch and hasattr(policy, "update"):
                policy.update(sub_batch)


__all__ = [
    "HierarchicalAgent",
    "ReplayBuffer",
    "BasePolicy",
    "ConstantPolicy",
    "TrendPolicy",
    "MeanReversionPolicy",
    "EpsilonGreedyManager",
]
