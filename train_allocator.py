"""Train the :class:`~models.capital_allocator.CapitalAllocator` policy network."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np

from models.capital_allocator import CapitalAllocator


def _generate_synthetic(n_strategies: int, n_samples: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return synthetic (pnl, risk, target) arrays for training.

    Targets are derived from a simple exponential scoring rule where strategies
    with higher PnL and lower risk receive larger weights.
    """
    pnl = np.random.randn(n_samples, n_strategies) * 0.05
    risk = np.abs(np.random.randn(n_samples, n_strategies) * 0.05)
    scores = pnl - risk
    exp = np.exp(scores - scores.max(axis=1, keepdims=True))
    target = exp / exp.sum(axis=1, keepdims=True)
    return pnl, risk, target


def train(output: Path, n_strategies: int = 3, n_samples: int = 1000) -> None:
    pnl, risk, target = _generate_synthetic(n_strategies, n_samples)
    allocator = CapitalAllocator()
    allocator.train(pnl, risk, target, lr=1e-2, epochs=500)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, alpha=allocator.alpha, beta=allocator.beta)
    print(f"Saved policy to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train capital allocator")
    parser.add_argument("--out", type=Path, default=Path("allocator_policy.npz"))
    parser.add_argument("--strategies", type=int, default=3)
    parser.add_argument("--samples", type=int, default=1000)
    args = parser.parse_args()
    train(args.out, args.strategies, args.samples)


if __name__ == "__main__":
    main()
