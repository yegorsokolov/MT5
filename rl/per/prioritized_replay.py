"""Prioritized experience replay buffer.

This implementation follows the scheme described in
`Schaul et al., 2015 <https://arxiv.org/abs/1511.05952>`_.  The buffer stores
arbitrary transition objects and allows sampling with probability proportional
to the transition priority.  It also computes importance sampling (IS) weights
that can be used to correct the bias introduced by prioritized sampling.

The API is intentionally lightweight so it can be used by our minimal
hierarchical agents as well as external libraries that only expect ``add`` and
``sample`` methods.  By default ``sample`` behaves exactly like the simple
uniform replay buffer returning only a list of transitions.  When
``with_weights`` is set to ``True`` it additionally returns the indices of the
sampled transitions and their IS weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, List, Sequence, Tuple
import numpy as np


@dataclass
class Transition:
    """Container for a single experience."""

    obs: Any
    action: Any
    reward: float
    next_obs: Any
    done: bool


class PrioritizedReplayBuffer:
    """Simple prioritized replay buffer.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to store.
    alpha:
        How much prioritization is used (0 corresponds to uniform sampling).
    beta:
        Exponent used to compute importance sampling weights.
    epsilon:
        Small amount added to priorities to avoid zero probability.
    """

    def __init__(
        self,
        capacity: int = 100000,
        *,
        alpha: float = 0.6,
        beta: float = 0.4,
        epsilon: float = 1e-6,
    ) -> None:
        self.capacity = int(capacity)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.epsilon = float(epsilon)
        self.buffer: List[Any] = []
        self.priorities = np.zeros(self.capacity, dtype=np.float32)
        self.pos = 0

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.buffer)

    # ------------------------------------------------------------------
    # Experience management
    # ------------------------------------------------------------------
    def add(self, transition: Any, priority: float | None = None) -> None:
        """Add a transition with optional priority.

        If ``priority`` is omitted, the transition receives the maximum current
        priority so that it will be sampled at least once.
        """

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        if priority is None:
            max_prio = self.priorities.max() if self.buffer else 1.0
            self.priorities[self.pos] = max_prio
        else:
            self.priorities[self.pos] = float(priority)

        self.pos = (self.pos + 1) % self.capacity

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------
    def sample(
        self, batch_size: int, *, with_weights: bool = False
    ) -> Tuple[List[Any], np.ndarray, np.ndarray] | List[Any]:
        """Sample a batch of transitions.

        Parameters
        ----------
        batch_size:
            Number of transitions to sample.
        with_weights:
            When ``True`` also return sample indices and importance sampling
            weights.  When ``False`` (default) only returns the list of
            transitions for compatibility with simple agents.
        """

        if not self.buffer:
            return ([], np.array([]), np.array([])) if with_weights else []

        prios = self.priorities[: len(self.buffer)] + self.epsilon
        scaled = prios ** self.alpha
        probs = scaled / scaled.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        if not with_weights:
            return samples

        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # normalize for stability
        return samples, indices, weights.astype(np.float32)

    # ------------------------------------------------------------------
    # Priority updates
    # ------------------------------------------------------------------
    def update_priorities(
        self, indices: Sequence[int], priorities: Sequence[float]
    ) -> None:
        """Update priorities of sampled transitions."""

        for idx, prio in zip(indices, priorities):
            self.priorities[int(idx)] = float(prio) + self.epsilon


__all__ = ["PrioritizedReplayBuffer", "Transition"]
