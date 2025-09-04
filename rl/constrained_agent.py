"""Primal-dual constrained RL agent enforcing risk budgets."""

from __future__ import annotations

from typing import List, Any

try:  # optional dependency
    import mlflow
except Exception:  # pragma: no cover - mlflow may be unavailable
    mlflow = None  # type: ignore


class ConstrainedAgent:
    """Lightweight agent applying a Lagrangian penalty to risky actions.

    The agent maintains estimates of reward and risk cost for each action and
    updates a dual variable (Lagrange multiplier) so that the expected cost does
    not exceed ``risk_budget``.  This mirrors the primal–dual updates used in
    algorithms such as Constrained Policy Optimisation (CPO).

    Parameters
    ----------
    n_actions:
        Number of discrete actions available.
    risk_budget:
        Maximum allowed expected cost (e.g. Value at Risk or drawdown).
    lr: float, default 0.01
        Step size for the dual variable update.
    risk_manager: optional
        If supplied, the agent scales ``risk_budget`` by the capital fraction
        allocated to ``strategy_id`` via ``risk_manager.budget_allocator`` so
        that live and training agents share budgets.
    strategy_id: str, default "rl"
        Identifier used when querying the budget allocator.
    """

    def __init__(
        self,
        n_actions: int,
        risk_budget: float,
        lr: float = 0.01,
        risk_manager: Any | None = None,
        strategy_id: str = "rl",
    ) -> None:
        self.n_actions = n_actions
        self.lr = lr
        self.lambda_ = 0.0
        self.rewards: List[float] = [0.0 for _ in range(n_actions)]
        self.costs: List[float] = [0.0 for _ in range(n_actions)]
        self.counts: List[int] = [0 for _ in range(n_actions)]
        self.steps = 0
        self.cost_history: List[float] = []
        self.violations = 0

        budget = risk_budget
        if risk_manager is not None:
            try:
                frac = risk_manager.budget_allocator.fraction(strategy_id)
                budget *= float(frac)
            except Exception:
                pass
        self.risk_budget = budget

    # Policy ---------------------------------------------------------------
    def act(self, obs: Any | None) -> int:  # pragma: no cover - trivial action selection
        """Return action index maximising Lagrangian-adjusted value."""
        scores = [r - self.lambda_ * c for r, c in zip(self.rewards, self.costs)]
        return int(scores.index(max(scores)))

    # Learning -------------------------------------------------------------
    def update(self, action: int, reward: float, cost: float) -> None:
        """Update empirical returns and dual variable from transition."""
        self.steps += 1
        self.counts[action] += 1
        n = self.counts[action]
        self.rewards[action] += (reward - self.rewards[action]) / n
        self.costs[action] += (cost - self.costs[action]) / n

        self.cost_history.append(cost)
        violation = float(cost > self.risk_budget)
        if violation:
            self.violations += 1
        self.lambda_ = max(0.0, self.lambda_ + self.lr * (cost - self.risk_budget))

        if mlflow is not None:  # pragma: no cover - logging side effect
            try:
                mlflow.log_metric("dual_variable", self.lambda_, step=self.steps)
                mlflow.log_metric("constraint_violation", violation, step=self.steps)
            except Exception:
                pass

    # Diagnostics ---------------------------------------------------------
    def avg_cost(self, window: int | None = None) -> float:
        """Return average cost over all or ``window`` most recent steps."""
        history = self.cost_history[-window:] if window else self.cost_history
        if not history:
            return 0.0
        return float(sum(history) / len(history))


__all__ = ["ConstrainedAgent"]
