import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd

# Provide a minimal sklearn.KMeans stub to avoid heavy dependency
cluster_mod = types.ModuleType("sklearn.cluster")

class KMeans:
    def __init__(self, n_clusters=8, n_init="auto", random_state=None):
        self.n_clusters = n_clusters

    def fit_predict(self, X):
        uniq = np.unique(X, axis=0)
        centers = uniq[: self.n_clusters]
        labels = [np.argmin(np.linalg.norm(centers - x, axis=1)) for x in X]
        self.cluster_centers_ = centers
        return np.array(labels)

cluster_mod.KMeans = KMeans
sklearn_mod = types.ModuleType("sklearn")
sklearn_mod.cluster = cluster_mod
sys.modules["sklearn"] = sklearn_mod
sys.modules["sklearn.cluster"] = cluster_mod

sys.path.append(str(Path(__file__).resolve().parents[1]))

from analysis.market_baskets import cluster_market_baskets


def test_merge_rare_basket(tmp_path):
    df = pd.DataFrame({
        "x": [0, 0, 0, 0, 0, 10],
        "y": [0, 0, 0, 0, 0, 10],
    })
    log_path = tmp_path / "basket_merges.csv"
    labels, _ = cluster_market_baskets(
        df,
        ["x", "y"],
        n_baskets=2,
        min_count=2,
        merge_log=log_path,
    )
    counts = labels.value_counts()
    assert counts.min() >= 2
    log_df = pd.read_csv(log_path)
    assert len(log_df) == 1
    assert log_df.loc[0, "source_count"] == 1


def test_no_endless_merge_loop(tmp_path):
    df = pd.DataFrame({
        "x": [0] * 6 + [10] + [20],
        "y": [0] * 6 + [10] + [20],
    })
    log_path = tmp_path / "basket_merges.csv"
    labels, _ = cluster_market_baskets(
        df,
        ["x", "y"],
        n_baskets=3,
        min_count=2,
        merge_log=log_path,
    )
    counts = labels.value_counts()
    assert counts.min() >= 2
    log_df = pd.read_csv(log_path)
    assert len(log_df) == 2
    assert log_df["source"].is_unique
