import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset


def test_tsfresh_features_plugin(monkeypatch):
    monkeypatch.setattr(dataset, "get_events", lambda past_events=False: [])
    monkeypatch.setattr(dataset, "add_news_sentiment_features", lambda df: df.assign(news_sentiment=0.0))
    monkeypatch.setattr(
        dataset,
        "add_index_features",
        lambda df: df.assign(sp500_ret=0.0, sp500_vol=0.0, vix_ret=0.0, vix_vol=0.0),
    )

    import utils
    cfg = {"use_tsfresh": True, "tsfresh_window": 5, "use_atr": False, "use_donchian": False}
    monkeypatch.setattr(utils, "load_config", lambda: cfg)

    import plugins.tsfresh_features as tsf
    monkeypatch.setattr(tsf, "load_config", lambda: cfg)

    n = 50
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=n, freq="min"),
        "Bid": np.linspace(1, 2, n),
        "Ask": np.linspace(1.0001, 2.0001, n),
    })

    out = dataset.make_features(df)
    expected = {
        "tsfresh_abs_change",
        "tsfresh_autocorr",
        "tsfresh_cid_ce",
        "tsfresh_kurtosis",
        "tsfresh_skewness",
    }
    assert expected.issubset(out.columns)
