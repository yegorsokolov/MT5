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


def test_train_test_split_multi_symbol():
    ts = pd.date_range("2020-01-01", periods=5, freq="min")
    df_a = pd.DataFrame({"Timestamp": ts, "Symbol": "A", "return": range(5)})
    df_b = pd.DataFrame({"Timestamp": ts, "Symbol": "B", "return": range(5)})
    df = pd.concat([df_a, df_b], ignore_index=True)

    train, test = dataset.train_test_split(df, n_train=3)
    assert len(train) == 6
    assert len(test) == 4
    assert list(train[train.Symbol == "A"]["return"]) == [0, 1, 2]
    assert list(test[test.Symbol == "B"]["return"]) == [3, 4]


def test_add_economic_calendar_features(monkeypatch):
    df = pd.DataFrame({"Timestamp": pd.date_range("2020-01-01", periods=3, freq="H")})
    events = [
        {"date": "2020-01-01T01:00:00Z", "impact": "High", "currency": "USD", "event": "news"}
    ]
    monkeypatch.setattr(dataset, "get_events", lambda past_events=True: events)
    out = dataset.add_economic_calendar_features(df.copy())
    assert "minutes_to_event" in out.columns
    assert out.loc[0, "minutes_to_event"] == 60
    assert out.loc[1, "minutes_to_event"] == 0


def test_save_load_history_parquet(tmp_path):
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=3, freq="min"),
        "Bid": [1.0, 1.1, 1.2],
        "Ask": [1.0001, 1.1001, 1.2001],
    })
    path = tmp_path / "hist.parquet"
    dataset.save_history_parquet(df, path)
    loaded = dataset.load_history_parquet(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_make_sequence_arrays():
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=6, freq="min"),
        "Symbol": "A",
        "return": [0, 0.1, -0.1, 0.2, -0.2, 0.3],
        "f1": [1, 2, 3, 4, 5, 6],
    })
    X, y = dataset.make_sequence_arrays(df, ["f1"], seq_len=2)
    assert X.shape == (3, 2, 1)
    assert y.tolist() == [1, 0, 1]


def test_keltner_and_regime_plugins(monkeypatch):
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
    monkeypatch.setattr(
        utils,
        "load_config",
        lambda: {"use_keltner": True, "use_regime_classifier": True},
    )
    import types, sys
    sys.modules['regime'] = types.SimpleNamespace(
        label_regimes=lambda df, n_states=3, column='regime_hmm': df.assign(**{column: 0})
    )
    import plugins.keltner as keltner
    import plugins.regime_plugin as regime_plugin
    monkeypatch.setattr(keltner, "load_config", lambda: {"use_keltner": True})
    monkeypatch.setattr(
        regime_plugin,
        "load_config",
        lambda: {"use_regime_classifier": True, "regime_states": 3},
    )

    n = 100
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=n, freq="min"),
        "Bid": np.linspace(1, 2, n),
        "Ask": np.linspace(1.0001, 2.0001, n),
    })

    result = dataset.make_features(df)
    assert {"keltner_high", "keltner_low", "keltner_break"}.issubset(result.columns)
    assert "regime_hmm" in result.columns

def test_qlib_features_plugin(monkeypatch):
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
    monkeypatch.setattr(utils, "load_config", lambda: {"use_qlib_features": True})

    import importlib
    import types, sys
    sys.modules['regime'] = types.SimpleNamespace(
        label_regimes=lambda df, n_states=3, column='regime_hmm': df.assign(**{column: 0})
    )
    import dataset as ds
    importlib.reload(ds)
    globals()['dataset'] = ds
    import plugins.qlib_features as qf
    monkeypatch.setattr(qf, "load_config", lambda: {"use_qlib_features": True})

    dummy_ta = types.SimpleNamespace(
        MA=lambda s, window=10: s.rolling(window).mean(),
        RSI=lambda s, window=14: s.rolling(window).mean(),
    )
    sys.modules.setdefault("qlib", types.SimpleNamespace())
    sys.modules["qlib.contrib"] = types.SimpleNamespace(ta=dummy_ta)

    n = 300
    df = pd.DataFrame({
        "Timestamp": pd.date_range("2020-01-01", periods=n, freq="min"),
        "Bid": np.linspace(1, 2, n),
        "Ask": np.linspace(1.0001, 2.0001, n),
    })

    result = ds.make_features(df)
    assert {"qlib_ma10", "qlib_ma30", "qlib_rsi14"}.issubset(result.columns)


def test_graph_plugin_receives_adjacency(monkeypatch):
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
    monkeypatch.setattr(utils, "load_config", lambda: {"use_atr": False, "use_donchian": False})

    captured = {}

    def dummy_plugin(df, adjacency_matrices=None):
        captured["count"] = len(adjacency_matrices) if adjacency_matrices else 0
        return df.assign(graph_dummy=1)

    monkeypatch.setattr(dataset, "FEATURE_PLUGINS", [dummy_plugin])

    n = 60
    ts = pd.date_range("2020-01-01", periods=n, freq="min")
    df = pd.DataFrame({
        "Timestamp": list(ts) * 2,
        "Symbol": ["A"] * n + ["B"] * n,
        "Bid": np.linspace(1, 2, 2 * n),
        "Ask": np.linspace(1.0001, 2.0001, 2 * n),
    })

    result = dataset.make_features(df)
    assert "graph_dummy" in result.columns
    assert captured.get("count", 0) > 0
