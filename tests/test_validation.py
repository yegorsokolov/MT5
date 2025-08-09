import pandas as pd
import numpy as np
import pandera as pa
import pytest
import types, sys

import dataset


def _basic_df(n: int = 10) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Timestamp": pd.date_range("2020-01-01", periods=n, freq="min"),
            "Bid": np.linspace(1, 2, n),
            "Ask": np.linspace(1.0001, 2.0001, n),
        }
    )


def test_load_history_validation_error(tmp_path):
    bad = pd.DataFrame({
        "Timestamp": ["20200101 00:00:00:000"],
        "Bid": [1.0],
    })
    path = tmp_path / "bad.csv"
    bad.to_csv(path, index=False)
    with pytest.raises(pa.errors.SchemaError):
        dataset.load_history(path, validate=True)


def test_make_features_validation_error(monkeypatch):
    monkeypatch.setattr(dataset, "get_events", lambda past_events=False: [])
    monkeypatch.setattr(
        dataset,
        "add_news_sentiment_features",
        lambda df: df.assign(news_sentiment=0.0),
    )
    monkeypatch.setattr(
        dataset,
        "add_index_features",
        lambda df: df.assign(
            sp500_ret=0.0,
            sp500_vol=0.0,
            vix_ret=0.0,
            vix_vol=0.0,
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "regime",
        types.SimpleNamespace(
            label_regimes=lambda df: df.assign(market_regime="bad")
        ),
    )
    df = _basic_df()
    with pytest.raises(pa.errors.SchemaError):
        dataset.make_features(df, validate=True)
