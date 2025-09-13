"""Nearest-neighbor features based on similar historical windows."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence, Tuple

import joblib
import numpy as np
import pandas as pd

try:  # pragma: no cover - optional sklearn dependency
    from sklearn.decomposition import PCA  # type: ignore
    from sklearn.neighbors import NearestNeighbors  # type: ignore
except Exception:  # pragma: no cover - fallback to minimal implementations
    PCA = None
    NearestNeighbors = None


class _FallbackPCA:
    def __init__(self, n_components: int):
        self.n_components = n_components
        self.components_: np.ndarray | None = None

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        cov = np.atleast_2d(np.cov(Xc, rowvar=False))
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1][: self.n_components]
        self.components_ = eigvecs[:, idx]
        return Xc @ self.components_

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float) - self.mean_
        return X @ self.components_


class _FallbackNN:
    def __init__(self, n_neighbors: int, algorithm: str | None = None):
        self.n_neighbors = n_neighbors
        self._data: np.ndarray | None = None

    def fit(self, X: np.ndarray):
        self._data = np.asarray(X, dtype=float)
        return self

    def kneighbors(self, X: np.ndarray, n_neighbors: int | None = None):
        n_neighbors = n_neighbors or self.n_neighbors
        X = np.asarray(X, dtype=float)
        dists = ((self._data[None, :, :] - X[:, None, :]) ** 2).sum(axis=2)
        idx = np.argsort(dists, axis=1)[:, :n_neighbors]
        dist = np.take_along_axis(dists, idx, axis=1)
        return dist, idx


if PCA is None:  # pragma: no cover - executed when sklearn missing
    PCA = _FallbackPCA  # type: ignore
if NearestNeighbors is None:  # pragma: no cover
    NearestNeighbors = _FallbackNN  # type: ignore


class SimilarDaysIndex:
    """Stores embeddings and returns for nearest-neighbor retrieval.

    The index is built using a PCA embedding followed by a sklearn
    :class:`~sklearn.neighbors.NearestNeighbors` structure.  It can be saved to
    disk via :meth:`save` and later reloaded with :meth:`load` for inference
    time feature augmentation.
    """

    def __init__(self, pca: PCA, embeddings: np.ndarray, returns: np.ndarray, k: int) -> None:
        self.pca = pca
        self.embeddings = embeddings
        self.returns = returns
        self.k = k
        self.nn = NearestNeighbors(n_neighbors=k, algorithm="auto")
        self.nn.fit(embeddings)

    def query(self, X: pd.DataFrame | np.ndarray, k: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
        """Return mean and volatility of neighbor returns for ``X``.

        Parameters
        ----------
        X:
            Feature matrix to embed and query.
        k:
            Optional override for the number of neighbors.  When ``None`` the
            value specified during index construction is used.
        """
        emb = self.pca.transform(np.asarray(X, dtype=float))
        k = k or self.k
        _dist, idx = self.nn.kneighbors(emb, n_neighbors=k)
        r = self.returns[idx]
        return r.mean(axis=1), r.std(axis=1)

    def save(self, path: Path) -> None:
        """Persist the index to ``path`` using :mod:`joblib`."""
        joblib.dump({"pca": self.pca, "embeddings": self.embeddings, "returns": self.returns, "k": self.k}, path)

    @classmethod
    def load(cls, path: Path) -> "SimilarDaysIndex":
        """Load a previously saved index from ``path``."""
        data = joblib.load(path)
        return cls(data["pca"], data["embeddings"], data["returns"], data["k"])


def add_similar_day_features(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    return_col: str = "return",
    k: int = 5,
    n_components: int = 5,
    index_path: Path | None = None,
) -> Tuple[pd.DataFrame, SimilarDaysIndex]:
    """Append nearest-neighbor summary statistics to ``df``.

    The function computes a PCA embedding over ``feature_cols`` and, for each
    row, retrieves the ``k`` most similar past observations.  The mean and
    standard deviation of their returns are appended as ``nn_return_mean`` and
    ``nn_vol`` respectively.  An ANN index suitable for inference-time queries
    is returned and optionally saved to ``index_path``.
    """
    required = set(feature_cols) | {return_col}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    X = df[list(feature_cols)].fillna(0.0).to_numpy(dtype=float)
    returns = df[return_col].to_numpy(dtype=float)
    n_comp = int(min(n_components, X.shape[1], len(df))) or 1
    pca = PCA(n_components=n_comp)
    embeddings = pca.fit_transform(X)

    nn_return_mean = np.zeros(len(df))
    nn_vol = np.zeros(len(df))
    for i in range(len(df)):
        if i == 0:
            continue
        past_emb = embeddings[:i]
        n_neighbors = min(k, len(past_emb))
        nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="auto")
        nbrs.fit(past_emb)
        _dist, idx = nbrs.kneighbors([embeddings[i]])
        r = returns[idx[0]]
        nn_return_mean[i] = r.mean()
        nn_vol[i] = r.std(ddof=0)

    df["nn_return_mean"] = nn_return_mean
    df["nn_vol"] = nn_vol

    index = SimilarDaysIndex(pca, embeddings, returns, k)
    if index_path is not None:
        index.save(index_path)
    return df, index
