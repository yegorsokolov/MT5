from . import register_feature
import pandas as pd
import numpy as np
from typing import Dict
import logging

logger = logging.getLogger(__name__)

try:
    import torch
    from torch_geometric.utils import dense_to_sparse, degree
    from torch_geometric.data import Data
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
