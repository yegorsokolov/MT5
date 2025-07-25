import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dataset


def test_make_features_columns(monkeypatch):
    monkeypatch.setattr(dataset, "get_events", lambda past_events=False: [])
    monkeypatch.setattr(dataset, "add_news_sentiment_features", lambda df: df.assign(news_sentiment=0.0))
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
    import utils
    monkeypatch.setattr(utils, "load_config", lambda: {"use_atr": True, "use_donchian": True})

    n = 300
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=n, freq="min"),
        "Bid": np.linspace(1, 2, n),
        "Ask": np.linspace(1.0001, 2.0001, n),
    })

    result = dataset.make_features(df)
    expected = {
        "return",
        "ma_5",
        "ma_10",
        "ma_30",
        "ma_60",
        "ma_h4",
        "boll_upper",
        "boll_lower",
        "boll_break",
        "atr_14",
        "atr_stop_long",
        "atr_stop_short",
        "donchian_high",
        "donchian_low",
        "donchian_break",
        "volatility_30",
        "rsi_14",
        "spread",
        "mid_change",
        "spread_change",
        "trade_rate",
        "quote_revision",
        "hour",
        "hour_sin",
        "hour_cos",
        "ma_cross",
        "minutes_to_event",
        "minutes_from_event",
        "nearest_news_minutes",
        "upcoming_red_news",
        "news_sentiment",
        "sp500_ret",
        "sp500_vol",
        "vix_ret",
        "vix_vol",
        "market_regime",
    }
    assert expected.issubset(result.columns)
