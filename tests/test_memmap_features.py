import numpy as np
import pandas as pd
import sys
import types
from pathlib import Path

# Stub heavy dependencies similar to other tests
sys.modules.setdefault("mlflow", types.SimpleNamespace(
    set_tracking_uri=lambda *a, **k: None,
    set_experiment=lambda *a, **k: None,
    start_run=lambda *a, **k: types.SimpleNamespace(__enter__=lambda self: None, __exit__=lambda self, *exc: None),
    log_dict=lambda *a, **k: None,
))
sys.modules.setdefault("utils.environment", types.SimpleNamespace(ensure_environment=lambda: None))
sklearn_stub = types.ModuleType("sklearn")
sklearn_stub.decomposition = types.SimpleNamespace(PCA=lambda *a, **k: None)
sys.modules.setdefault("sklearn", sklearn_stub)
sys.modules.setdefault("sklearn.decomposition", sklearn_stub.decomposition)
sys.modules.setdefault("duckdb", types.SimpleNamespace(connect=lambda *a, **k: None))
sys.modules.setdefault("networkx", types.SimpleNamespace())

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset
import data.features as F
import utils
from data.labels import triple_barrier


def test_memmap_features_same_as_inmemory(tmp_path, monkeypatch):
    monkeypatch.setattr(F, "get_events", lambda past_events=False: [])
    monkeypatch.setattr(F, "add_news_sentiment_features", lambda df: df.assign(news_sentiment=0.0))
    monkeypatch.setattr(
        F,
        "add_index_features",
        lambda df: df.assign(
            sp500_ret=0.0,
            sp500_vol=0.0,
            vix_ret=0.0,
            vix_vol=0.0,
        ),
    )
    monkeypatch.setattr(utils, "load_config", lambda: {"use_atr": True, "use_donchian": True})

    n = 300
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=n, freq="min"),
        "Bid": np.linspace(1, 2, n),
        "Ask": np.linspace(1.0001, 2.0001, n),
    })
    path = tmp_path / "hist.parquet"
    df.to_parquet(path)

    feats_inmem = dataset.make_features(df.copy())
    feats_memmap = dataset.make_features_memmap(path, chunk_size=100)

    pd.testing.assert_frame_equal(
        feats_memmap.reset_index(drop=True),
        feats_inmem.reset_index(drop=True),
    )

    labels_mem = triple_barrier(feats_inmem["mid"], 0.01, 0.01, 5)
    labels_mm = triple_barrier(feats_memmap["mid"], 0.01, 0.01, 5)
    pd.testing.assert_series_equal(
        labels_mm.reset_index(drop=True),
        labels_mem.reset_index(drop=True),
    )
