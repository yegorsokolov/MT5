from __future__ import annotations

"""Hierarchical risk parity portfolio optimiser."""

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np


def _cov_to_corr(cov: np.ndarray) -> np.ndarray:
    std = np.sqrt(np.diag(cov))
    # Avoid division by zero for assets with zero variance
    std[std == 0] = 1.0
    corr = cov / std[:, None] / std[None, :]
    np.fill_diagonal(corr, 1.0)
    return corr


def _single_linkage_order(dist: np.ndarray) -> list[int]:
    """Return asset order from a simple single-linkage clustering tree."""
    n = dist.shape[0]
    clusters: list[set[int]] = [
        {i} for i in range(n)
    ]  # each cluster represented by set of member indices
    nodes: list[object] = [i for i in range(n)]  # tree nodes (ints or tuples)
    while len(clusters) > 1:
        min_d = float("inf")
        pair: tuple[int, int] | None = None
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                d = min(dist[p, q] for p in clusters[i] for q in clusters[j])
                if d < min_d:
                    min_d = d
                    pair = (i, j)
        assert pair is not None
        i, j = pair
        new_set = clusters[i] | clusters[j]
        new_node = (nodes[i], nodes[j])
        for idx in sorted(pair, reverse=True):
            clusters.pop(idx)
            nodes.pop(idx)
        clusters.append(new_set)
        nodes.append(new_node)
    tree = nodes[0]

    def _get_order(node: object) -> list[int]:
        if isinstance(node, int):
            return [node]
        left, right = node  # type: ignore[misc]
        return _get_order(left) + _get_order(right)

    return _get_order(tree)


def _cluster_var(cov: np.ndarray, indices: Sequence[int]) -> float:
    sub = cov[np.ix_(indices, indices)]
    inv_diag = 1.0 / np.diag(sub)
    weights = inv_diag / inv_diag.sum()
    return float(weights @ sub @ weights)


def _recursive_bisection(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    weights = np.ones(n)
    clusters: list[list[int]] = [list(range(n))]
    while clusters:
        cluster = clusters.pop(0)
        if len(cluster) <= 1:
            continue
        split = len(cluster) // 2
        left = cluster[:split]
        right = cluster[split:]
        var_left = _cluster_var(cov, left)
        var_right = _cluster_var(cov, right)
        alpha = 1.0 - var_left / (var_left + var_right)
        weights[left] *= alpha
        weights[right] *= 1 - alpha
        clusters.extend([left, right])
    return weights / weights.sum()


def hrp_weights(cov_matrix: np.ndarray) -> np.ndarray:
    """Return hierarchical risk parity weights for ``cov_matrix``."""
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be square")
    corr = _cov_to_corr(cov)
    dist = np.sqrt(0.5 * (1.0 - corr))
    order = _single_linkage_order(dist)
    cov_sorted = cov[np.ix_(order, order)]
    w_sorted = _recursive_bisection(cov_sorted)
    weights = np.zeros(len(order))
    weights[order] = w_sorted
    return weights


@dataclass
class HRPOptimizer:
    """Compute portfolio weights using hierarchical risk parity."""

    weights: np.ndarray | None = field(default=None, init=False)

    def compute_weights(
        self, expected_returns: Sequence[float] | None, cov_matrix: np.ndarray
    ) -> np.ndarray:
        weights = hrp_weights(cov_matrix)
        self.weights = weights
        return weights

    def diversification_ratio(self) -> float:
        if self.weights is None or len(self.weights) == 0:
            return 0.0
        return float((1.0 / np.sum(self.weights ** 2)) / len(self.weights))
