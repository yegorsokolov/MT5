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

from dataclasses import dataclass
from typing import List, Optional, Sequence

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


@dataclass
class GraphExample:
    """Container holding a single graph learning example."""

    x: torch.Tensor
    edge_index: torch.Tensor
    edge_weight: torch.Tensor | None
    y: torch.Tensor
    timestamp: Optional[pd.Timestamp] = None


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
) -> List[GraphExample]:
    """Create :class:`GraphExample` objects aligned with adjacency matrices."""

    adj_attr = df.attrs.get("adjacency_matrices")
    if adj_attr is None:
        raise ValueError("df.attrs['adjacency_matrices'] must be provided")

    if isinstance(adj_attr, dict):
        ordered = sorted(adj_attr.items(), key=lambda kv: kv[0])
        timestamps = [pd.to_datetime(ts) for ts, _ in ordered]
        matrices = [np.asarray(mat) for _, mat in ordered]
    else:
        matrices = [np.asarray(mat) for mat in adj_attr]
        timestamps = [None] * len(matrices)

    df = df.sort_values("Timestamp")
    ret_pivot = (
        df.pivot(index="Timestamp", columns="Symbol", values="return")
        .sort_index()
        .fillna(0.0)
    )
    X_all = build_node_features(df, symbols, feature_cols)

    if len(matrices) == 0:
        raise ValueError("adjacency_matrices must contain at least one matrix")

    start = len(ret_pivot.index) - len(matrices)
    X_tensors = X_all[start:-1]
    y_arr = ret_pivot.iloc[start + 1 :].to_numpy(dtype=np.float32)
    matrices = matrices[:-1]
    timestamps = timestamps[:-1]

    examples: List[GraphExample] = []
    for x, y_row, mat, ts in zip(X_tensors, y_arr, matrices, timestamps):
        y = torch.tensor(y_row, dtype=torch.float32).view(len(symbols), 1)
        nz = np.nonzero(mat)
        if nz[0].size == 0:
            # No edges above threshold, fall back to identity so that the
            # aggregation step has self information available.
            nz = np.nonzero(np.eye(mat.shape[0], dtype=mat.dtype))
            mat = np.eye(mat.shape[0], dtype=mat.dtype)
        mask = nz[0] != nz[1]
        if mask.any():
            rows = nz[0][mask]
            cols = nz[1][mask]
            weights = mat[rows, cols]
        else:
            rows = nz[0]
            cols = nz[1]
            weights = mat[rows, cols]
        edge_index = torch.tensor(np.vstack((rows, cols)), dtype=torch.long)
        edge_weight = torch.tensor(weights, dtype=torch.float32) if weights.size else None
        examples.append(
            GraphExample(
                x=x,
                edge_index=edge_index,
                edge_weight=edge_weight,
                y=y,
                timestamp=ts,
            )
        )
    return examples


def _format_timestamp(ts: Optional[pd.Timestamp]) -> str:
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    if ts is None:
        return "n/a"
    return str(ts)


def _log_top_edges(
    weights: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    symbols: Sequence[str],
    top_k: int,
    timestamp: Optional[pd.Timestamp],
    *,
    prefix: str,
) -> None:
    if weights is None or weights.numel() == 0 or top_k <= 0:
        return
    weights_cpu = weights.detach().cpu().reshape(-1)
    edge_cpu = edge_index.detach().cpu()
    k = min(top_k, weights_cpu.numel())
    values, idx = torch.topk(weights_cpu.abs(), k)
    entries: List[str] = []
    for order, edge_pos in enumerate(idx):
        src = symbols[int(edge_cpu[0, edge_pos])]
        dst = symbols[int(edge_cpu[1, edge_pos])]
        entries.append(
            f"{src}->{dst}:{weights_cpu[edge_pos].item():.3f} (|w|={values[order].item():.3f})"
        )
    logger.info("%s at %s: %s", prefix, _format_timestamp(timestamp), ", ".join(entries))


def _log_attention_summary(
    attn: torch.Tensor,
    edge_index: torch.Tensor,
    symbols: Sequence[str],
    top_k: int,
    timestamp: Optional[pd.Timestamp],
) -> None:
    attn_cpu = attn.detach().cpu()
    if attn_cpu.ndim == 2:
        mean_attn = attn_cpu.mean(dim=1)
        head_means = attn_cpu.mean(dim=0)
        logger.info(
            "attention head means: %s",
            ", ".join(f"h{idx}:{float(val):.3f}" for idx, val in enumerate(head_means)),
        )
    else:
        mean_attn = attn_cpu.reshape(-1)
    logger.info(
        "attention stats at %s -> mean %.4f std %.4f",
        _format_timestamp(timestamp),
        float(mean_attn.mean()),
        float(mean_attn.std(unbiased=False)),
    )
    resolved_top = top_k if top_k > 0 else min(5, mean_attn.numel())
    _log_top_edges(
        mean_attn,
        edge_index,
        symbols,
        resolved_top,
        timestamp,
        prefix="top attention weights",
    )


def train_graphnet(
    df: pd.DataFrame,
    cfg: dict,
    *,
    return_losses: bool = False,
) -> tuple[torch.nn.Module, List[float]] | torch.nn.Module:
    """Train :class:`~models.graph_net.GraphNet` on ``df``.

    Parameters
    ----------
    df:
        Long-form DataFrame containing at least ``Timestamp``, ``Symbol`` and a
        ``return`` column. ``df.attrs`` must provide ``"adjacency_matrices"`` as
        described in :func:`build_graph_examples`. When absent, the
        :mod:`data.features` pipeline is invoked to compute cross-asset features
        including rolling correlation adjacency matrices.
    cfg:
        Configuration dictionary. Top-level keys may specify ``"symbols"``. The
        optional ``"graph"`` sub-dictionary configures the model architecture and
        training hyper-parameters with the following recognised fields:

        - ``epochs`` / ``lr`` / ``hidden_channels`` / ``num_layers``: overrides
          for the optimiser and architecture.
        - ``use_gat`` (bool): build a :class:`models.graph_attention.GATNet`
          instead of :class:`models.graph_net.GraphNet`. ``heads`` and
          ``dropout`` further tune the attention mechanism.
        - ``log_top_k_edges`` (int) and ``log_attention`` (bool): enable logging
          of the strongest correlations/attentions for interpretability.
    return_losses:
        If ``True`` the function returns a tuple ``(model, losses)`` where
        ``losses`` is a list of average epoch losses.
    """

    graph_cfg = cfg.get("graph") or {}

    def _cfg_get(key: str, default=None):
        if isinstance(graph_cfg, dict) and key in graph_cfg:
            return graph_cfg[key]
        return cfg.get(key, default)

    if "adjacency_matrices" not in df.attrs:
        try:
            feat_cfg = load_config()
            df = make_features(df, validate=feat_cfg.get("validate", False))
        except Exception:
            df = make_features(df)

    if not {"Symbol", "Timestamp"}.issubset(df.columns):
        raise ValueError("df must contain 'Symbol' and 'Timestamp' columns")

    symbols = _cfg_get("symbols")
    if symbols is None:
        symbols = sorted(df["Symbol"].unique())
    symbols = list(symbols)

    feature_cols = [
        c
        for c in df.columns
        if c not in {"Symbol", "Timestamp"}
        and np.issubdtype(df[c].dtype, np.number)
    ]
    examples = build_graph_examples(df, symbols, feature_cols)
    if not examples:
        raise ValueError("Insufficient graph examples were generated")

    use_gat = bool(_cfg_get("use_gat", False))
    hidden_channels = int(_cfg_get("hidden_channels", 32))
    num_layers = int(_cfg_get("num_layers", 2))
    lr = float(_cfg_get("lr", 0.01))
    log_top_k = int(_cfg_get("log_top_k_edges", 0) or 0)
    log_attention = bool(_cfg_get("log_attention", False))

    if use_gat:
        heads = int(_cfg_get("heads", _cfg_get("gat_heads", 1)))
        dropout = float(_cfg_get("dropout", _cfg_get("gat_dropout", 0.0)))
        model: torch.nn.Module = GATNet(
            in_channels=examples[0].x.size(1),
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=num_layers,
            heads=heads,
            dropout=dropout,
        )
    else:
        model = GraphNet(
            in_channels=examples[0].x.size(1),
            hidden_channels=hidden_channels,
            out_channels=1,
            num_layers=num_layers,
        )
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = MSELoss()

    epochs = int(_cfg_get("epochs", 100))
    device = next(model.parameters()).device
    losses: List[float] = []
    model.train()
    for epoch in range(epochs):
        total = 0.0
        last_example: Optional[GraphExample] = None
        for example in examples:
            x = example.x.to(device)
            edge_index = example.edge_index.to(device)
            edge_weight = (
                example.edge_weight.to(device)
                if example.edge_weight is not None
                else None
            )
            y = example.y.to(device)
            if use_gat:
                pred = model(x, edge_index)
            else:
                pred = model(x, edge_index, edge_weight=edge_weight)
            loss = loss_fn(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.item())
            last_example = example
        losses.append(total / len(examples))

        if log_top_k and last_example is not None and not use_gat:
            _log_top_edges(
                last_example.edge_weight,
                last_example.edge_index,
                symbols,
                log_top_k,
                last_example.timestamp,
                prefix="top edge weights",
            )
        if use_gat and log_attention and getattr(model, "last_attention", None) is not None:
            attn = model.last_attention
            edge_index = getattr(model, "last_edge_index", last_example.edge_index if last_example else None)
            if edge_index is not None:
                _log_attention_summary(
                    attn,
                    edge_index,
                    symbols,
                    log_top_k,
                    last_example.timestamp if last_example else None,
                )

    if return_losses:
        return model, losses
    return model


__all__ = ["build_node_features", "build_graph_examples", "train_graphnet"]
