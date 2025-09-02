import pandas as pd
import sys, types

# Stub feature gate to avoid heavy dependencies during import
sys.modules.setdefault(
    "analysis.feature_gate",
    types.SimpleNamespace(select=lambda df, tier, regime_id, persist=False: (df, [])),
)

sys.modules["analytics.metrics_store"] = types.SimpleNamespace(
    record_metric=lambda *a, **k: None, TS_PATH=""
)

from data import features as feat
from rl.trading_env import TradingEnv
from analytics import decision_logger


def test_sentiment_features_and_env(monkeypatch):
    monkeypatch.setattr(feat, "get_feature_pipeline", lambda: [])
    from utils.resource_monitor import monitor
    monkeypatch.setattr(monitor, "capability_tier", "standard", raising=False)

    import news.sentiment_score as ss

    def fake_load_vectors(window: int = 3, cache_dir=None):
        return pd.DataFrame(
            {
                "symbol": ["AAPL", "AAPL"],
                "timestamp": pd.to_datetime(["2024-01-01", "2024-01-02"]),
                "news_sentiment_0": [0.1, 0.2],
                "news_sentiment_1": [0.0, 0.1],
                "news_impact_0": [0.5, 0.4],
                "news_impact_1": [0.0, 0.5],
            }
        )

    monkeypatch.setattr(ss, "load_vectors", fake_load_vectors)

    df = pd.DataFrame(
        {
            "Timestamp": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "Symbol": ["AAPL", "AAPL"],
            "mid": [100.0, 101.0],
        }
    )
    df = feat.make_features(df)
    assert "news_sentiment_0" in df.columns
    assert "news_impact_0" in df.columns
    env = TradingEnv(df, features=[], news_window=0)
    assert any(col.endswith("news_sentiment_0") for col in env.feature_cols)
    obs = env.reset()
    assert obs.shape[0] == len(env.feature_cols)


def test_decision_logger_includes_news(monkeypatch):
    captured = {}

    def fake_log_decision(df):
        captured["df"] = df

    monkeypatch.setattr(decision_logger, "log_decision", fake_log_decision)

    df = pd.DataFrame({"symbol": ["AAPL"], "action": [1]})
    news = [
        {
            "title": "AAPL rallies",
            "sentiment": 0.5,
            "impact": 0.2,
            "url": "http://example.com",
        }
    ]
    decision_logger.log(df, news=news)
    logged = captured["df"]
    assert "news" in logged.columns
    assert logged["news"].iloc[0][0]["title"] == "AAPL rallies"
    assert logged["news"].iloc[0][0]["sentiment"] == 0.5
