from . import register_feature
from utils import load_config
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from torch_geometric.utils import dense_to_sparse, degree
    from torch_geometric.data import Data
    from torch_geometric.nn import GATConv
except Exception:  # pragma: no cover - optional dependency
    torch = None
    dense_to_sparse = None
    degree = None
    Data = None


@register_feature
def add_graph_features(df: pd.DataFrame, adjacency_matrices: Dict[pd.Timestamp, np.ndarray] | None = None) -> pd.DataFrame:
    """Add simple graph degree features derived from adjacency matrices."""
    if adjacency_matrices is None or torch is None:
        return df

    if "Symbol" not in df.columns:
        return df

    symbols = sorted(df["Symbol"].unique())
    deg_rows = []
    for ts, mat in adjacency_matrices.items():
        try:
            tensor = torch.tensor(mat, dtype=torch.float)
            edge_index, edge_weight = dense_to_sparse(tensor)
            g = Data(edge_index=edge_index, edge_attr=edge_weight, num_nodes=len(symbols))
            deg = degree(g.edge_index[0], num_nodes=len(symbols), dtype=torch.float)
        except Exception as e:  # pragma: no cover - runtime errors
            logger.warning("Graph feature failed at %s: %s", ts, e)
            continue
        row = {"Timestamp": ts}
        for i, sym in enumerate(symbols):
            row[f"graph_deg_{sym}"] = float(deg[i])
        deg_rows.append(row)

    if not deg_rows:
        return df

    deg_df = pd.DataFrame(deg_rows)
    df = df.merge(deg_df, on="Timestamp", how="left")
    return df


@register_feature
def add_gat_embeddings(
    df: pd.DataFrame, adjacency_matrices: Dict[pd.Timestamp, np.ndarray] | None = None
) -> pd.DataFrame:
    """Generate node embeddings using a tiny GAT model for each timestamp."""
    cfg = load_config()
    if not cfg.get("use_gat_features", False):
        return df

    if adjacency_matrices is None or torch is None:
        return df

    if "Symbol" not in df.columns:
        return df

    symbols = sorted(df["Symbol"].unique())
    num_nodes = len(symbols)

    class SmallGAT(torch.nn.Module):
        def __init__(self, nodes: int) -> None:
            super().__init__()
            self.conv1 = GATConv(nodes, 8, add_self_loops=False)
            self.conv2 = GATConv(8, 4, add_self_loops=False)

        def forward(self, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
            x = torch.eye(num_nodes)
            x = self.conv1(x, edge_index, edge_weight)
            x = torch.relu(x)
            x = self.conv2(x, edge_index, edge_weight)
            return x

    model = SmallGAT(num_nodes)
    emb_rows = []
    for ts, mat in adjacency_matrices.items():
        try:
            tensor = torch.tensor(mat, dtype=torch.float)
            edge_index, edge_weight = dense_to_sparse(tensor)
            embeddings = model(edge_index, edge_weight)
        except Exception as e:  # pragma: no cover - runtime errors
            logger.warning("GAT embedding failed at %s: %s", ts, e)
            continue
        for i, sym in enumerate(symbols):
            row = {"Timestamp": ts, "Symbol": sym}
            for j in range(embeddings.shape[1]):
                row[f"gat_emb_{j}"] = float(embeddings[i, j])
            emb_rows.append(row)

    if not emb_rows:
        return df

    emb_df = pd.DataFrame(emb_rows)
    df = df.merge(emb_df, on=["Timestamp", "Symbol"], how="left")
    return df
