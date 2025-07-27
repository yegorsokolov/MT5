import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset


def _base_cfg():
    return {
        "use_pair_trading": True,
        "pair_z_window": 5,
        "pair_long_threshold": -1.0,
        "pair_short_threshold": 1.0,
        "use_atr": False,
        "use_donchian": False,
    }


def build_df():
    n = 300
    ts = pd.date_range("2020-01-01", periods=n, freq="min")
    price_a = np.linspace(1.0, 2.0, n)
    price_b = 2 * price_a
    price_b[250:260] += 2  # temporary divergence after initial burn-in
    df_a = pd.DataFrame({
        "Timestamp": ts,
        "Symbol": "A",
        "Bid": price_a,
        "Ask": price_a + 0.0001,
    })
    df_b = pd.DataFrame({
        "Timestamp": ts,
        "Symbol": "B",
        "Bid": price_b,
        "Ask": price_b + 0.0001,
    })
    return pd.concat([df_a, df_b], ignore_index=True)


def test_pair_trading_features(monkeypatch):
    monkeypatch.setattr(dataset, "get_events", lambda past_events=False: [])
    monkeypatch.setattr(dataset, "add_news_sentiment_features", lambda df: df.assign(news_sentiment=0.0))
    monkeypatch.setattr(dataset, "add_index_features", lambda df: df.assign(sp500_ret=0.0, sp500_vol=0.0, vix_ret=0.0, vix_vol=0.0))

    import utils
    monkeypatch.setattr(utils, "load_config", _base_cfg)

    df = build_df()
    out = dataset.make_features(df)
    assert any(col.startswith("pair_z_A_B") for col in out.columns)
    assert "pair_long" in out.columns
    assert "pair_short" in out.columns
    assert (out["pair_long"] + out["pair_short"]).notnull().all()

