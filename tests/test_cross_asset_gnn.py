import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pandas as pd
import pytest

torch = pytest.importorskip("torch")


def _synthetic_df(n: int = 60) -> pd.DataFrame:
    rows = []
    rng = np.random.default_rng(0)
    timestamps = pd.date_range("2024-01-01", periods=n, freq="D")
    for ts in timestamps:
        a = rng.normal(scale=0.1)
        b = a + rng.normal(scale=0.1)
        rows.append({"Timestamp": ts, "Symbol": "A", "return": a})
        rows.append({"Timestamp": ts, "Symbol": "B", "return": b})
    return pd.DataFrame(rows)


def test_cross_asset_gnn_beats_independent():
    spec_ca = importlib.util.spec_from_file_location(
        "cross_asset", Path(__file__).resolve().parents[1] / "features" / "cross_asset.py"
    )
    ca = importlib.util.module_from_spec(spec_ca)
    spec_ca.loader.exec_module(ca)  # type: ignore

    spec_tg = importlib.util.spec_from_file_location(
        "train_graphnet", Path(__file__).resolve().parents[1] / "mt5" / "train_graphnet.py"
    )
    tg = importlib.util.module_from_spec(spec_tg)
    sys.path.append(str(Path(__file__).resolve().parents[1]))
    sys.modules.setdefault(
        "data",
        types.SimpleNamespace(features=types.SimpleNamespace(make_features=lambda df, **k: df)),
    )
    sys.modules.setdefault(
        "data.features", types.SimpleNamespace(make_features=lambda df, **k: df)
    )
    sys.modules.setdefault(
        "utils", types.SimpleNamespace(load_config=lambda: {"cross_asset": {"adjacency_window": 5}})
    )
    spec_tg.loader.exec_module(tg)  # type: ignore
    train_graphnet = tg.train_graphnet

    df = _synthetic_df()
    feat_df = ca.compute(df)

    cfg = {
        "symbols": ["A", "B"],
        "graph": {"epochs": 80, "lr": 0.1, "hidden_channels": 8},
    }
    _, losses = train_graphnet(feat_df, cfg, return_losses=True)

    eye = np.eye(2, dtype=np.float32)
    baseline_df = feat_df[["Timestamp", "Symbol", "return"]].copy()
    baseline_df.attrs["adjacency_matrices"] = [
        (ts, eye) for ts, _ in feat_df.attrs["adjacency_matrices"]
    ]
    _, base_losses = train_graphnet(baseline_df, cfg, return_losses=True)

    assert losses[-1] < base_losses[-1]

