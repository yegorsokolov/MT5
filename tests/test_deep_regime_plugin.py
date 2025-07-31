import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset

def test_deep_regime_plugin(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "get_events", lambda past_events=False: [])
    monkeypatch.setattr(dataset, "add_news_sentiment_features", lambda df: df.assign(news_sentiment=0.0))
    monkeypatch.setattr(
        dataset,
        "add_index_features",
        lambda df: df.assign(sp500_ret=0.0, sp500_vol=0.0, vix_ret=0.0, vix_vol=0.0),
    )

    import utils
    cfg = {
        "use_deep_regime": True,
        "deep_regime_window": 5,
        "deep_regime_dim": 2,
        "deep_regime_states": 2,
        "use_atr": False,
        "use_donchian": False,
    }
    monkeypatch.setattr(utils, "load_config", lambda: cfg)

    import plugins.deep_regime as dr
    monkeypatch.setattr(dr, "load_config", lambda: cfg)
    monkeypatch.setattr(dr, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(dr, "EPOCHS", 1)

    n = 100
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=n, freq="min"),
        "Bid": np.linspace(1, 2, n),
        "Ask": np.linspace(1.0001, 2.0001, n),
    })

    out = dataset.make_features(df)
    assert "regime_dl" in out.columns
    assert out["regime_dl"].notna().sum() > 0
