import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset  # noqa: E402
import data.features as features  # noqa: E402


def _basic_patches(monkeypatch):
    monkeypatch.setattr(features, "add_economic_calendar_features", lambda df: df)
    monkeypatch.setattr(features, "add_news_sentiment_features", lambda df: df)
    monkeypatch.setattr(features, "add_index_features", lambda df: df)
    monkeypatch.setattr(features, "add_cross_asset_features", lambda df: df)
    cfg = {
        "use_atr": False,
        "use_donchian": False,
        "macro_series": ["GDP", "CPI"],
    }
    import types
    dummy_utils = types.SimpleNamespace(load_config=lambda: cfg)
    monkeypatch.setitem(sys.modules, "utils", dummy_utils)
    return cfg


def test_macro_merge_accuracy(monkeypatch):
    _basic_patches(monkeypatch)
    macro_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(["2020-01-01", "2020-03-01"]),
            "GDP": [1.0, 2.0],
            "CPI": [3.0, 4.0],
        }
    )
    monkeypatch.setattr(features, "load_macro_series", lambda symbols: macro_df)

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(
                ["2020-02-15", "2020-03-15", "2020-04-15"]
            ),
            "Bid": np.linspace(1, 1.2, 3),
            "Ask": np.linspace(1.0001, 1.2001, 3),
        }
    )

    out = dataset.make_features(df)
    assert list(out["macro_GDP"]) == [1.0, 2.0, 2.0]
    assert list(out["macro_CPI"]) == [3.0, 4.0, 4.0]


def test_macro_missing_data(monkeypatch):
    _basic_patches(monkeypatch)
    macro_df = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01"]), "GDP": [np.nan]})
    monkeypatch.setattr(features, "load_macro_series", lambda symbols: macro_df)

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2020-02-15", "2020-03-15"]),
            "Bid": [1.0, 1.1],
            "Ask": [1.0001, 1.1001],
        }
    )

    out = dataset.make_features(df)
    assert "macro_GDP" in out.columns
    assert "macro_CPI" in out.columns
    assert out["macro_GDP"].isna().all()
    assert out["macro_CPI"].isna().all()
