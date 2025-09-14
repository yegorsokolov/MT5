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
from data.features import make_features
from utils import load_config


logger = logging.getLogger(__name__)


def build_node_features(
    df: pd.DataFrame, symbols: Sequence[str], feature_cols: Sequence[str]
) -> List[torch.Tensor]:
    """Return list of node feature tensors for each time step.

    ``df`` must be in long format with ``Timestamp`` and ``Symbol`` columns
    alongside numeric feature columns. ``feature_cols`` selects which columns to
    include as node features.
    """

    df = df.sort_values("Timestamp")
    grouped = df.groupby("Timestamp")
    tensors: List[torch.Tensor] = []
    for _, group in grouped:
        group = group.set_index("Symbol")
        feats = [group.loc[sym, feature_cols].to_list() for sym in symbols]
        tensors.append(torch.tensor(feats, dtype=torch.float32))
    return tensors


def build_graph_examples(
    df: pd.DataFrame, symbols: Sequence[str], feature_cols: Sequence[str]
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]]:
    """Create training examples ``(x, edge_index, edge_weight, y)`` per step."""

    adj_attr = df.attrs.get("adjacency_matrices")
    if adj_attr is None:
        raise ValueError("df.attrs['adjacency_matrices'] must be provided")
    if isinstance(adj_attr, dict):
        matrices = [adj_attr[ts] for ts in sorted(adj_attr.keys())]
    else:
        matrices = list(adj_attr)

    df = df.sort_values("Timestamp")
    ret_pivot = df.pivot(index="Timestamp", columns="Symbol", values="return").sort_index()
    X_all = build_node_features(df, symbols, feature_cols)

    start = len(ret_pivot.index) - len(matrices)
    matrices = matrices[:-1]
    X_tensors = X_all[start:-1]
    y_arr = ret_pivot.iloc[start + 1 :].to_numpy(dtype=np.float32)

    examples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor]] = []
    for x, y_row, mat in zip(X_tensors, y_arr, matrices):
        y = torch.tensor(y_row, dtype=torch.float32).view(len(symbols), 1)
        nz = np.nonzero(mat)
        edge_index = torch.tensor(np.vstack(nz), dtype=torch.long)
        edge_weight = torch.tensor(mat[nz], dtype=torch.float32)
        examples.append((x, edge_index, edge_weight, y))
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

    if "adjacency_matrices" not in df.attrs:
        try:
            feat_cfg = load_config()
            df = make_features(df, validate=feat_cfg.get("validate", False))
        except Exception:
            df = make_features(df)

    if not {"Symbol", "Timestamp"}.issubset(df.columns):
        raise ValueError("df must contain 'Symbol' and 'Timestamp' columns")

    symbols = cfg.get("symbols")
    if symbols is None:
        symbols = sorted(df["Symbol"].unique())

    feature_cols = [
        c
        for c in df.columns
        if c not in {"Symbol", "Timestamp"}
        and np.issubdtype(df[c].dtype, np.number)
    ]
    examples = build_graph_examples(df, symbols, feature_cols)

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
        for x, edge_index, edge_weight, y in examples:
            if use_gat:
                pred = model(x, edge_index)
                if getattr(model, "last_attention", None) is not None:
                    logger.info("attention weights: %s", model.last_attention)
            else:
                pred = model(x, edge_index, edge_weight=edge_weight)
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
