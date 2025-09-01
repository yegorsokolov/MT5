import importlib
import sys
from pathlib import Path

import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))


def test_vector_store_similarity(tmp_path, monkeypatch):
    idx = tmp_path / "vec.ann"
    meta = tmp_path / "vec.json"
    monkeypatch.setenv("NEWS_VECTOR_INDEX", str(idx))
    monkeypatch.setenv("NEWS_VECTOR_META", str(meta))
    vs = importlib.reload(importlib.import_module("news.vector_store"))
    texts = ["Fed raises rates", "ECB cuts rates", "Stocks rally on earnings"]
    vs.rebuild(texts)
    res = vs.similar_events("Fed hikes rates", k=1)
    assert res and res[0][0] == "Fed raises rates" and res[0][1] > 0


def test_vector_store_empty(tmp_path, monkeypatch):
    idx = tmp_path / "vec.ann"
    meta = tmp_path / "vec.json"
    monkeypatch.setenv("NEWS_VECTOR_INDEX", str(idx))
    monkeypatch.setenv("NEWS_VECTOR_META", str(meta))
    vs = importlib.reload(importlib.import_module("news.vector_store"))
    res = vs.similar_events("No index", k=5)
    assert res == []
