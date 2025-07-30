import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset


def test_autoencoder_features(monkeypatch, tmp_path):
    monkeypatch.setattr(dataset, "get_events", lambda past_events=False: [])
    monkeypatch.setattr(dataset, "add_news_sentiment_features", lambda df: df.assign(news_sentiment=0.0))
    monkeypatch.setattr(
        dataset,
        "add_index_features",
        lambda df: df.assign(sp500_ret=0.0, sp500_vol=0.0, vix_ret=0.0, vix_vol=0.0),
    )

    import utils
    cfg = {
        "use_autoencoder_features": True,
        "autoencoder_window": 5,
        "autoencoder_dim": 2,
        "use_atr": False,
        "use_donchian": False,
    }
    monkeypatch.setattr(utils, "load_config", lambda: cfg)

    import plugins.autoencoder_features as aef
    monkeypatch.setattr(aef, "load_config", lambda: cfg)
    monkeypatch.setattr(aef, "MODELS_DIR", tmp_path)
    monkeypatch.setattr(aef, "EPOCHS", 1)

    n = 100
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=n, freq="min"),
        "Bid": np.linspace(1, 2, n),
        "Ask": np.linspace(1.0001, 2.0001, n),
    })

    out = dataset.make_features(df)
    assert {"ae_0", "ae_1"}.issubset(out.columns)
    assert out["ae_0"].notna().sum() > 0

