from __future__ import annotations

"""Lightweight vector store for news event embeddings.

This module stores text embeddings using an ``AnnoyIndex`` and provides a
``similar_events`` helper used by the news impact model.  The index location can
be customised via environment variables:

``NEWS_VECTOR_INDEX``
    Path to the persisted Annoy index.  Defaults to
    ``analysis/news_vectors.ann`` relative to the project root.

``NEWS_VECTOR_META``
    JSON file storing the original event text for each vector in the index.
    Defaults to the same directory as the index with ``.json`` extension.

Embeddings are generated via a tiny hashing based encoder which avoids heavy
model dependencies while remaining deterministic.
"""

import json
import hashlib
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
from annoy import AnnoyIndex

# ---------------------------------------------------------------------------
# Storage paths
_DIM = 32
_INDEX_PATH = Path(
    os.getenv(
        "NEWS_VECTOR_INDEX",
        Path(__file__).resolve().parents[1] / "analysis" / "news_vectors.ann",
    )
)
_META_PATH = Path(
    os.getenv("NEWS_VECTOR_META", _INDEX_PATH.with_suffix(".json"))
)

_index: AnnoyIndex | None = None
_texts: List[str] = []


# ---------------------------------------------------------------------------
# Embedding helper

def embed(text: str, dim: int = _DIM) -> np.ndarray:
    """Return a simple normalised hashing based embedding for ``text``."""

    vec = np.zeros(dim, dtype=np.float32)
    for word in text.split():
        h = int(hashlib.md5(word.lower().encode("utf-8")).hexdigest(), 16)
        vec[h % dim] += 1.0
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    return vec


# ---------------------------------------------------------------------------
# Index loading and rebuilding

def _load() -> None:
    """Lazily load the Annoy index and metadata from disk."""

    global _index, _texts
    if _index is not None:
        return
    _index = AnnoyIndex(_DIM, metric="angular")
    if _INDEX_PATH.exists() and _META_PATH.exists():
        _index.load(str(_INDEX_PATH))
        with _META_PATH.open("r", encoding="utf-8") as fh:
            _texts = json.load(fh)
    else:
        _texts = []


def rebuild(texts: Iterable[str]) -> None:
    """Rebuild the index from ``texts``.

    ``texts`` should be an iterable of headline/event strings.  The function
    computes embeddings, builds a new Annoy index and persists both the index
    and accompanying metadata to ``_INDEX_PATH`` and ``_META_PATH``.
    """

    global _index, _texts
    _index = AnnoyIndex(_DIM, metric="angular")
    _texts = []
    for i, text in enumerate(texts):
        emb = embed(text)
        _index.add_item(i, emb)
        _texts.append(text)
    if _texts:
        _index.build(10)
        _INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
        _index.save(str(_INDEX_PATH))
        with _META_PATH.open("w", encoding="utf-8") as fh:
            json.dump(_texts, fh)


# ---------------------------------------------------------------------------
# Query API

def similar_events(text: str, k: int = 5) -> List[Tuple[str, float]]:
    """Return up to ``k`` events similar to ``text``.

    The function returns a list of ``(event_text, similarity)`` tuples where
    similarity is mapped to the ``[0, 1]`` range with larger values indicating
    greater similarity.
    """

    _load()
    if not _texts:
        return []
    k = min(k, len(_texts))
    query = embed(text)
    idx, dists = _index.get_nns_by_vector(  # type: ignore[union-attr]
        query.tolist(), k, include_distances=True
    )
    results: List[Tuple[str, float]] = []
    for i, dist in zip(idx, dists):
        similarity = 1.0 - dist / 2.0  # convert angular distance to [0,1]
        results.append((_texts[i], similarity))
    return results


__all__ = ["rebuild", "similar_events", "embed"]
