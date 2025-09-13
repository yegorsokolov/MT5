from __future__ import annotations

"""Hypernetwork for dynamic indicator generation without external deps."""

from typing import Sequence, Tuple


class IndicatorHyperNet:
    """Deterministic hypernetwork producing simple indicator parameters."""

    def __init__(self, in_dim: int, seed: int = 0) -> None:  # pragma: no cover - trivial
        self.in_dim = in_dim
        self.seed = seed

    def __call__(self, x: Sequence[float]) -> Tuple[int, int]:
        s = float(sum(x)) + self.seed
        lag = int(abs(round(s))) % 5 + 1
        window = int(abs(round(s + 1))) % 9 + 2
        return lag, window


__all__ = ["IndicatorHyperNet"]
