"""Train :class:`models.graph_net.GraphNet` on multi-asset returns.

This lightweight training routine constructs node features from per-symbol
return series and obtains the graph structure from
``df.attrs['adjacency_matrices']``.  Each row of ``df`` represents a single
time step with columns containing the return for each symbol.  The
``adjacency_matrices`` attribute should be an iterable of ``(n,n)`` NumPy
arrays describing the directed edges between symbols for each time step.

The model is trained to predict next-step returns (or optionally provided
labels) for every symbol using mean squared error loss.
"""

from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import logging
import numpy as np
import pandas as pd
import torch
from torch.optim import Adam
from torch.nn import MSELoss

from models.graph_net import GraphNet
from models.graph_attention import GATNet


logger = logging.getLogger(__name__)


def build_node_features(df: pd.DataFrame, symbols: Sequence[str]) -> List[torch.Tensor]:
    """Return list of node feature tensors from per-symbol returns.

    Parameters
    ----------
    df:
        DataFrame where each column corresponds to the return series of a
        symbol.  Each row is a time step.
    symbols:
        Ordered sequence of symbols matching the columns in ``df``.
    """

    arr = df[symbols].to_numpy(dtype=np.float32)
    return [torch.from_numpy(row).view(len(symbols), 1) for row in arr]


def build_graph_examples(
    df: pd.DataFrame, symbols: Sequence[str]
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Create training examples ``(x, edge_index, y)`` for each time step.

    ``y`` consists of the future return for each symbol.  ``edge_index`` is
    derived from ``df.attrs['adjacency_matrices']`` using the standard COO
    format.
    """

    returns = df[symbols]
    X_df = returns.iloc[:-1]
    y_df = returns.shift(-1).dropna()

    adj_attr = df.attrs.get("adjacency_matrices")
    if adj_attr is None:
        raise ValueError("df.attrs['adjacency_matrices'] must be provided")
    if isinstance(adj_attr, dict):
        matrices = [adj_attr[idx] for idx in df.index]
    else:
        matrices = list(adj_attr)
    matrices = matrices[:-1]

    examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
    x_list = build_node_features(X_df, symbols)
    y_list = build_node_features(y_df, symbols)
    for x, y, mat in zip(x_list, y_list, matrices):
        edge_index = torch.tensor(np.array(mat).nonzero(), dtype=torch.long)
        examples.append((x, edge_index, y))
    return examples


def train_graphnet(
    df: pd.DataFrame,
    cfg: dict,
    *,
    return_losses: bool = False,
) -> Tuple[GraphNet, List[float]] | GraphNet:
    """Train :class:`~models.graph_net.GraphNet` on ``df``.

    Parameters
    ----------
    df:
        Wide DataFrame of per-symbol return series.  ``df.attrs`` must contain
        ``"adjacency_matrices"`` as described in :func:`build_graph_examples`.
    cfg:
        Configuration dictionary. Recognised keys include ``"symbols"``,
        ``"epochs"``, ``"lr"``, ``"hidden_channels"`` and ``"num_layers"``. Set
        ``"use_gat"`` to ``True`` to build a :class:`models.graph_attention.GATNet`
        instead of :class:`models.graph_net.GraphNet`. Additional GAT-specific
        options are ``"gat_heads"`` controlling the number of attention heads and
        ``"gat_dropout"`` which applies dropout to the attention weights.
    return_losses:
        If ``True`` the function returns a tuple ``(model, losses)`` where
        ``losses`` is a list of average epoch losses.
    """

    symbols = cfg.get("symbols")
    if symbols is None:
        symbols = list(df.columns)
    examples = build_graph_examples(df, symbols)

    use_gat = cfg.get("use_gat", False)
    if use_gat:
        model = GATNet(
            in_channels=examples[0][0].size(1),
            hidden_channels=cfg.get("hidden_channels", 32),
            out_channels=1,
            num_layers=cfg.get("num_layers", 2),
            heads=cfg.get("gat_heads", 1),
            dropout=cfg.get("gat_dropout", 0.0),
        )
    else:
        model = GraphNet(
            in_channels=examples[0][0].size(1),
            hidden_channels=cfg.get("hidden_channels", 32),
            out_channels=1,
            num_layers=cfg.get("num_layers", 2),
        )
    optimizer = Adam(model.parameters(), lr=cfg.get("lr", 0.01))
    loss_fn = MSELoss()

    epochs = int(cfg.get("epochs", 100))
    losses: List[float] = []
    for _ in range(epochs):
        total = 0.0
        for x, edge_index, y in examples:
            pred = model(x, edge_index)
            if use_gat and getattr(model, "last_attention", None) is not None:
                logger.debug("attention weights: %s", model.last_attention)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
        losses.append(total / len(examples))

    if return_losses:
        return model, losses
    return model


__all__ = ["build_node_features", "build_graph_examples", "train_graphnet"]
