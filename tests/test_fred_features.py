import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset


def test_fred_features_plugin(monkeypatch):
    monkeypatch.setattr(dataset, "get_events", lambda past_events=False: [])
    monkeypatch.setattr(dataset, "add_news_sentiment_features", lambda df: df.assign(news_sentiment=0.0))
    monkeypatch.setattr(
        dataset,
        "add_index_features",
        lambda df: df.assign(sp500_ret=0.0, sp500_vol=0.0, vix_ret=0.0, vix_vol=0.0),
    )

    import utils
    cfg = {
        "use_fred_features": True,
        "fred_series": ["DUMMY"],
        "use_atr": False,
        "use_donchian": False,
    }
    monkeypatch.setattr(utils, "load_config", lambda: cfg)

    import plugins.fred_features as ff
    monkeypatch.setattr(ff, "load_config", lambda: cfg)

    dummy = pd.DataFrame({"DUMMY": np.linspace(1, 3, 3)}, index=pd.date_range("2020-01-01", periods=3, freq="D"))
    monkeypatch.setitem(sys.modules, "pandas_datareader.data", type("mod", (), {"DataReader": lambda code, src, start, end: dummy}))

    n = 20
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=n, freq="H"),
        "Bid": np.linspace(1, 2, n),
        "Ask": np.linspace(1.0001, 2.0001, n),
    })

    out = dataset.make_features(df)
    assert "fred_dummy" in out.columns

